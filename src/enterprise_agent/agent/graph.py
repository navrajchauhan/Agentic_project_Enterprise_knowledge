"""Graph assembly.

The workflow is a small, explicit state machine rather than a free-form ReAct
loop, because enterprise QA has a known shape:

.. code-block:: text

    START
      |
    guard_input ---------------(blocked)--------------> END
      |
    route ----(tool)----> tool_agent <---> tools
      |                       |  \\--(budget spent)--> tool_finalize
      |                       \\----(answer ready)----> guard_output
      |--(retrieve)--> retrieve --> grade --(ok)--> generate --> guard_output
      |                             |
      |                             \\--(insufficient)--> rewrite --> retrieve
      \\--(direct)-----------------------------------> guard_output
                                                          |
                                                         END

Why this matters for hallucination control: an answer can only be produced by
``generate`` (which sees retrieved evidence) or by a tool result, and every path
converges on ``guard_output``.  There is no edge that lets generated prose skip
the guard.

Both loops are bounded:
* ``rewrite -> retrieve`` is bounded by ``MAX_QUERY_REWRITES``;
* ``tool_agent -> tools`` is bounded by ``MAX_TOOL_STEPS`` (plus LangGraph's
  ``recursion_limit`` as a backstop).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..config import Settings
from ..logging_setup import get_logger
from .nodes import AgentNodes
from .state import AgentState

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conditional-edge predicates (kept as small pure functions so they are testable)
# ---------------------------------------------------------------------------


def after_input_guard(state: AgentState) -> str:
    return "end" if state.get("blocked") else "route"


def after_route(state: AgentState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool_agent"
    if route == "direct":
        return "direct"
    return "retrieve"


def after_grade(state: AgentState, *, max_rewrites: int) -> str:
    if state.get("sufficient"):
        return "generate"
    if int(state.get("rewrite_count") or 0) >= max_rewrites:
        # Out of rewrite budget: generate anyway, with the low-confidence hint.
        return "generate"
    if not state.get("evidence"):
        # Nothing was retrieved, so a rewrite cannot help.
        return "generate"
    return "rewrite"


def after_tool_agent(state: AgentState, *, max_steps: int) -> str:
    messages = state.get("messages") or []
    calls = getattr(messages[-1], "tool_calls", None) if messages else None
    if not calls:
        return "guard_output"
    if int(state.get("tool_steps") or 0) >= max_steps:
        return "tool_finalize"
    return "tools"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_graph(
    nodes: AgentNodes,
    settings: Settings,
    *,
    checkpointer: object | None = None,
) -> CompiledStateGraph:
    """Compile the agent graph.

    ``checkpointer`` is what makes the workflow *stateful*: with it, every node
    transition is persisted per ``thread_id`` and the next question on the same
    thread resumes with the full message history.
    """
    builder = StateGraph(AgentState)

    builder.add_node("guard_input", nodes.guard_input)
    builder.add_node("route", nodes.route)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade", nodes.grade)
    builder.add_node("rewrite", nodes.rewrite)
    builder.add_node("generate", nodes.generate)
    builder.add_node("tool_agent", nodes.tool_agent)
    builder.add_node("tools", nodes.tools_node)
    builder.add_node("tool_finalize", nodes.finalize_tools)
    builder.add_node("direct", nodes.direct)
    builder.add_node("guard_output", nodes.guard_output)

    builder.add_edge(START, "guard_input")
    builder.add_conditional_edges(
        "guard_input", after_input_guard, {"route": "route", "end": END}
    )
    builder.add_conditional_edges(
        "route",
        after_route,
        {"retrieve": "retrieve", "tool_agent": "tool_agent", "direct": "direct"},
    )

    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        lambda state: after_grade(state, max_rewrites=settings.max_query_rewrites),
        {"generate": "generate", "rewrite": "rewrite"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", "guard_output")

    builder.add_conditional_edges(
        "tool_agent",
        lambda state: after_tool_agent(state, max_steps=settings.max_tool_steps),
        {"tools": "tools", "tool_finalize": "tool_finalize", "guard_output": "guard_output"},
    )
    builder.add_edge("tools", "tool_agent")
    builder.add_edge("tool_finalize", "guard_output")

    builder.add_edge("direct", "guard_output")
    builder.add_edge("guard_output", END)

    compiled = builder.compile(checkpointer=checkpointer, name="enterprise_rag_agent")
    logger.debug("Compiled agent graph with %d nodes", 11)
    return compiled


def graph_diagram(graph: CompiledStateGraph) -> str:
    """Mermaid source for the compiled graph (used by ``run.py graph``).

    Handy in a PR description or a design doc: it is generated from the code, so
    it cannot drift from the real topology.
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as exc:  # noqa: BLE001 - diagramming is optional
        return f"graph diagram unavailable: {exc}"
