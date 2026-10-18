"""The LangGraph state schema.

LangGraph passes a single mutable-state dictionary through the nodes, where each
node returns only the keys it changes.  Two special ``Annotated`` channels:

``messages``
    Uses ``add_messages``, which *appends* instead of replacing and de-duplicates
    by message id.  This is the conversation memory: combined with a
    checkpointer keyed by ``thread_id`` it gives multi-turn context for free.

``trace``
    Uses ``trace_reducer`` so every node can append its own line and the CLI can
    replay the decision path ("guard -> route=retrieve -> retrieve(6) ->
    grade=sufficient -> generate -> guard_output=pass").  A new turn sends
    ``TRACE_RESET`` first, so the trace always describes the current turn rather
    than the whole conversation.

Everything else is last-write-wins, which keeps the nodes trivial to reason
about.  ``total=False`` because no node sets the whole state.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

#: Sentinel that lets a node clear the trace when a new turn starts.
#: LangGraph reducers append, so without it the CLI would show every previous
#: turn's steps alongside the current one.
TRACE_RESET = "\x00reset-trace\x00"


def trace_reducer(existing: list[str] | None, update: list[str] | None) -> list[str]:
    """Append trace lines, honouring ``TRACE_RESET`` as "start again"."""
    lines = list(update or [])
    if lines and lines[0] == TRACE_RESET:
        return lines[1:]
    return [*(existing or []), *lines]


class AgentState(TypedDict, total=False):
    """State carried through one agent turn (and across turns per thread)."""

    # -- conversation ------------------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]

    # -- input -------------------------------------------------------------
    question: str  # normalised question for this turn
    original_question: str  # exactly what the user typed
    thread_id: str

    # -- guardrails --------------------------------------------------------
    guard_input: dict[str, Any]  # InputVerdict.to_dict()
    blocked: bool
    block_reason: str
    guard_output: dict[str, Any]  # OutputVerdict.to_dict()
    abstained: bool

    # -- routing / retrieval ----------------------------------------------
    route: str  # "retrieve" | "tool" | "direct"
    query: str  # the query actually used for search (may be rewritten)
    rewrite_count: int
    evidence: list[dict[str, Any]]  # serialised ScoredChunk list
    context: str  # numbered evidence block handed to the LLM
    sufficient: bool
    grade_confidence: float

    # -- tools -------------------------------------------------------------
    tool_steps: int
    tool_calls: list[dict[str, Any]]
    tool_provider: str

    # -- output ------------------------------------------------------------
    answer: str
    citations: list[dict[str, Any]]
    grounding_score: float
    #: True when the model could not be reached. The output guard then reports the
    #: real cause instead of rewriting the answer as "insufficient information".
    generation_failed: bool

    # -- observability -----------------------------------------------------
    trace: Annotated[list[str], trace_reducer]


def initial_state(question: str, thread_id: str) -> AgentState:
    """Fresh per-turn inputs.

    Turn-scoped counters (``rewrite_count``, ``tool_steps``) are reset here;
    the checkpointer restores everything else (``messages`` in particular) so
    the conversation continues where it left off.
    """
    return AgentState(
        question=question,
        original_question=question,
        thread_id=thread_id,
        rewrite_count=0,
        tool_steps=0,
        tool_calls=[],
        evidence=[],
        citations=[],
        blocked=False,
        abstained=False,
        sufficient=False,
        trace=[],
    )
