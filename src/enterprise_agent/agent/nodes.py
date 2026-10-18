"""Graph nodes - the actual behaviour of the agent.

Each node is a pure-ish function ``state -> partial update``:

===============  ==============================================================
``guard_input``  validate + sanitise the question; may end the turn immediately
``route``        pick retrieve / tool / direct (one cheap LLM call)
``retrieve``     hybrid search, render the numbered evidence block
``grade``        is the evidence sufficient? (one cheap LLM call)
``rewrite``      improve the query, then retrieve again (bounded loop)
``generate``     grounded answer with citations
``tool_agent``   LLM call with tools bound
``tools``        execute the requested tool calls, feed results back
``direct``       canned capabilities answer (greetings, meta questions)
``guard_output`` validate citations and grounding, then commit the answer
===============  ==============================================================

Design notes
------------
* **Every LLM failure degrades, never crashes.**  Routing falls back to a
  keyword heuristic, grading falls back to lexical overlap, generation falls
  back to an abstention.  An agent that returns "I couldn't reach the model" is
  better than a stack trace.
* **Only ``guard_output`` writes the final answer into ``messages``.**  That way
  the conversation memory can never contain an answer the guard rejected.
* **The tool loop is bounded twice** - by ``MAX_TOOL_STEPS`` in the conditional
  edge and by ``recursion_limit`` in the runner configuration.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from ..config import Settings
from ..guardrails.input_guard import InputGuard
from ..guardrails.output_guard import OutputGuard
from ..llm.base import bind_tools, invoke_json, message_text, tool_call_args, tool_call_name
from ..logging_setup import get_logger
from ..rag.models import ScoredChunk
from ..rag.retriever import HybridRetriever
from ..tools.models import ToolExecutor
from ..utils.text import content_tokens, looks_like_tool_request
from . import prompts
from .state import TRACE_RESET, AgentState

logger = get_logger(__name__)

#: How many prior messages are replayed into a prompt.  Enough for follow-ups
#: ("what about international?") without drowning the evidence in history.
HISTORY_WINDOW = 8

ABSTAIN_TEXT = "I don't have enough information in the knowledge base to answer that."

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|who are you|"
    r"what can you do|help)\b[\s!.?]*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Structured-output models for the cheap LLM calls
# ---------------------------------------------------------------------------


class RouteDecision(BaseModel):
    route: Literal["retrieve", "tool", "direct"] = "retrieve"
    reason: str = Field(default="", description="short justification")


class GradeDecision(BaseModel):
    sufficient: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    missing: list[str] = Field(default_factory=list)


class RewriteDecision(BaseModel):
    query: str = ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class AgentNodes:
    """Holds the dependencies and implements every graph node."""

    def __init__(
        self,
        *,
        settings: Settings,
        llm: BaseChatModel,
        fast_llm: BaseChatModel,
        retriever: HybridRetriever,
        tools: ToolExecutor,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.fast_llm = fast_llm
        self.retriever = retriever
        self.tools = tools
        self.input_guard = input_guard or InputGuard(settings)
        self.output_guard = output_guard or OutputGuard(settings)
        # Bind the tool schemas once at construction: binding per call would
        # rebuild the schema list on every step of the loop.
        self.tool_llm = bind_tools(llm, tools.openai_tools())

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _conversation(state: AgentState) -> list[BaseMessage]:
        """Recent conversation, excluding the system prompts we add ourselves."""
        return list(state.get("messages") or [])[-HISTORY_WINDOW:]

    def _history_text(self, state: AgentState) -> str:
        """Compact history block for the router prompt."""
        lines: list[str] = []
        for message in (state.get("messages") or [])[-4:]:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            text = message_text(message).strip().replace("\n", " ")
            if text:
                lines.append(f"{role}: {text[:180]}")
        return "\n".join(lines) or "(empty)"

    # -- 1. input guard ----------------------------------------------------
    def guard_input(self, state: AgentState) -> dict[str, Any]:
        verdict = self.input_guard.check(state.get("question", ""))
        update: dict[str, Any] = {
            "guard_input": verdict.to_dict(),
            "original_question": state.get("original_question") or state.get("question", ""),
        }

        if not verdict.allowed:
            logger.info("Blocked input: %s", verdict.reason)
            update.update(
                blocked=True,
                block_reason=verdict.reason,
                answer=verdict.refusal_message,
                abstained=True,
                trace=[TRACE_RESET, f"input_guard:blocked({verdict.reason})"],
                messages=[AIMessage(content=verdict.refusal_message)],
            )
            return update

        # The redacted, normalised query is what the rest of the turn sees.
        update.update(
            question=verdict.query,
            query=verdict.query,
            blocked=False,
            trace=[TRACE_RESET, f"input_guard:allow(risk={verdict.risk_score:.2f})"],
            messages=[HumanMessage(content=verdict.query)],
        )
        return update

    # -- 2. router ---------------------------------------------------------
    def route(self, state: AgentState) -> dict[str, Any]:
        question = state.get("question", "")

        # Cheap deterministic fast paths before spending an LLM call.
        if _GREETING_RE.match(question):
            return {"route": "direct", "trace": ["route:direct(heuristic:greeting)"]}

        decision = invoke_json(
            self.fast_llm,
            system=prompts.ROUTER_SYSTEM,
            human=prompts.ROUTER_HUMAN.format(
                question=question, history=self._history_text(state)
            ),
            schema=RouteDecision,
        )

        if decision is None:
            # Router unavailable: fall back to the imperative-phrase heuristic
            # (an action verb mid-sentence means "asks about", not "do it").
            route = "tool" if looks_like_tool_request(question) else "retrieve"
            trace = f"route:{route}(fallback:heuristic)"
        else:
            route = decision.route
            trace = f"route:{route}({decision.reason[:40]})"

        return {"route": route, "trace": [trace]}

    # -- 3. retrieval ------------------------------------------------------
    def retrieve(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state.get("question", "")
        hits = self.retriever.search(query, top_k=self.settings.top_k)

        # If a rewrite made things worse, fall back to what the user actually
        # asked: cheap insurance against a bad rewrite.
        if not hits and state.get("original_question"):
            logger.info("Rewrite returned no hits; retrying with the original question")
            hits = self.retriever.search(
                state["original_question"], top_k=self.settings.top_k
            )
            query = state["original_question"]

        context = self.retriever.context_block(hits) if hits else ""
        logger.debug("Retrieved %d chunk(s) for %r", len(hits), query)

        return {
            "query": query,
            "evidence": [hit.to_dict() for hit in hits],
            "context": context,
            "trace": [f"retrieve:{len(hits)} hits({query[:40]!r})"],
        }

    # -- 4. grade evidence -------------------------------------------------
    def grade(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence") or []
        question = state.get("question", "")

        if not evidence:
            return {
                "sufficient": False,
                "grade_confidence": 0.0,
                "trace": ["grade:insufficient(no_evidence)"],
            }

        decision = invoke_json(
            self.fast_llm,
            system=prompts.GRADER_SYSTEM,
            human=prompts.GRADER_HUMAN.format(
                question=question, context=state.get("context", "")
            ),
            schema=GradeDecision,
        )

        if decision is None:
            # Lexical fallback: does the evidence share enough vocabulary with
            # the question? Coarse, but better than blindly generating.
            question_tokens = content_tokens(question)
            evidence_tokens = content_tokens(state.get("context", ""))
            overlap = (
                len(question_tokens & evidence_tokens) / len(question_tokens)
                if question_tokens
                else 0.0
            )
            sufficient = overlap >= 0.2
            return {
                "sufficient": sufficient,
                "grade_confidence": round(overlap, 3),
                "trace": [f"grade:{'sufficient' if sufficient else 'insufficient'}(fallback)"],
            }

        return {
            "sufficient": decision.sufficient,
            "grade_confidence": decision.confidence,
            "trace": [
                f"grade:{'sufficient' if decision.sufficient else 'insufficient'}"
                f"({decision.confidence:.2f}: {decision.reason[:40]})"
            ],
        }

    # -- 5. query rewrite --------------------------------------------------
    def rewrite(self, state: AgentState) -> dict[str, Any]:
        question = state.get("question", "")
        previous = state.get("query", question)
        attempts = int(state.get("rewrite_count") or 0)

        if attempts >= self.settings.max_query_rewrites:
            return {"trace": [f"rewrite:skipped(max={self.settings.max_query_rewrites})"]}

        decision = invoke_json(
            self.fast_llm,
            system=prompts.REWRITE_SYSTEM,
            human=prompts.REWRITE_HUMAN.format(question=question, previous=previous),
            schema=RewriteDecision,
        )

        if decision is not None and decision.query.strip():
            query = decision.query.strip()
            source = "llm"
        else:
            # Deterministic fallback: drop stop-words so BM25 sees only content
            # terms. This alone fixes many vague queries.
            from ..utils.text import tokenize

            query = " ".join(dict.fromkeys(tokenize(question))) or question
            source = "keywords"

        return {
            "query": query,
            "rewrite_count": attempts + 1,
            "trace": [f"rewrite[{attempts + 1}/{self.settings.max_query_rewrites}]:{source}({query[:40]!r})"],
        }

    # -- 6. grounded generation -------------------------------------------
    def generate(self, state: AgentState) -> dict[str, Any]:
        question = state.get("question", "")
        evidence = state.get("evidence") or []
        context = state.get("context", "")

        if not evidence or not context:
            # No evidence -> no generation. Cheapest possible anti-hallucination.
            return {"answer": ABSTAIN_TEXT, "trace": ["generate:abstain(no_evidence)"]}

        human = prompts.ANSWER_HUMAN.format(question=question, context=context)
        if int(state.get("rewrite_count") or 0) > 0 and not state.get("sufficient"):
            human += prompts.LOW_CONFIDENCE_NOTE

        messages: list[BaseMessage] = [
            SystemMessage(content=prompts.ANSWER_SYSTEM),
            *self._conversation(state),
            HumanMessage(content=human),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = message_text(response).strip()
        except Exception as exc:  # noqa: BLE001 - provider outages are expected
            logger.error("Generation failed: %s", exc)
            return {
                "answer": (
                    "I couldn't reach the language model to answer that, so I can't ground a "
                    "response. The retrieved documents are listed below so you can check them "
                    "directly."
                ),
                "generation_failed": True,
                "trace": ["generate:error(llm_unavailable)"],
            }

        return {"answer": answer or ABSTAIN_TEXT, "trace": [f"generate:{len(answer)} chars"]}

    # -- 7. tool calling ---------------------------------------------------
    def tool_agent(self, state: AgentState) -> dict[str, Any]:
        """Ask the model what to do next, with tools bound.

        The question message is only appended on the *first* step of the turn.
        On later steps the conversation already ends with the tool results, and
        appending the question again would tell the model (and the deterministic
        double) that nothing has happened yet - which is how you get a tool loop.
        """
        question = state.get("question", "")
        messages: list[BaseMessage] = [
            SystemMessage(content=prompts.TOOL_SYSTEM),
            *self._conversation(state),
        ]
        if int(state.get("tool_steps") or 0) == 0:
            messages.append(HumanMessage(content=prompts.TOOL_HUMAN.format(question=question)))

        try:
            response = self.tool_llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool planning failed: %s", exc)
            return {
                "answer": "I couldn't reach the language model to perform that action.",
                "trace": ["tool_agent:error(llm_unavailable)"],
            }

        calls = getattr(response, "tool_calls", None) or []
        update: dict[str, Any] = {"messages": [response]}

        if calls:
            update["trace"] = [f"tool_agent:requested({len(calls)})"]
            update["tool_provider"] = type(self.tools).__name__
        else:
            # No more tool calls: the model's text *is* the answer.
            update["answer"] = message_text(response).strip()
            update["trace"] = ["tool_agent:final_answer"]
        return update

    def tools_node(self, state: AgentState) -> dict[str, Any]:
        """Execute every tool call from the last AI message."""
        messages = state.get("messages") or []
        if not messages:
            return {"trace": ["tools:skipped(no_messages)"]}

        calls = getattr(messages[-1], "tool_calls", None) or []
        if not calls:
            return {"trace": ["tools:skipped(no_calls)"]}

        thread_id = state.get("thread_id") or "-"
        tool_messages: list[ToolMessage] = []
        records: list[dict[str, Any]] = []
        traces: list[str] = []

        for index, call in enumerate(calls):
            name = tool_call_name(call)
            arguments = tool_call_args(call)
            call_id = call.get("id") or f"call_{index}"

            result = self.tools.execute(name, arguments, thread_id=thread_id)
            traces.append(f"tool:{name}({'ok' if result.ok else result.error or 'failed'})")
            records.append(
                {
                    "tool": name,
                    "arguments": arguments if isinstance(arguments, dict) else {"raw": arguments},
                    "ok": result.ok,
                    "content": result.content,
                    "error": result.error,
                }
            )

            # The model sees the structured JSON so it can quote concrete ids.
            tool_messages.append(
                ToolMessage(content=result.to_json(), tool_call_id=call_id, name=name)
            )

        return {
            "messages": tool_messages,
            "tool_steps": int(state.get("tool_steps") or 0) + 1,
            "tool_calls": [*(state.get("tool_calls") or []), *records],
            "trace": traces,
        }

    # -- 8. direct answer --------------------------------------------------
    def finalize_tools(self, state: AgentState) -> dict[str, Any]:
        """Compose an answer from tool records when the step budget runs out.

        This is the "budget exhausted" safety net: instead of ending with an
        empty answer (which the output guard would have to reject), report what
        the tools actually returned.  It is deterministic and quotes the tool
        output verbatim, so it cannot hallucinate.
        """
        records = state.get("tool_calls") or []
        if not records:
            return {
                "answer": "I couldn't complete that action.",
                "trace": ["tool_finalize:no_records"],
            }

        lines = []
        for record in records[-3:]:
            status = "ok" if record.get("ok") else f"failed ({record.get('error')})"
            summary = " ".join(str(record.get("content", "")).split())[:300]
            lines.append(f"- {record['tool']} [{status}]: {summary}")

        return {
            "answer": "I reached the tool-call limit. Here is what the tools returned:\n"
            + "\n".join(lines),
            "trace": ["tool_finalize:budget_exhausted"],
        }

    def direct(self, state: AgentState) -> dict[str, Any]:
        return {
            "answer": prompts.DIRECT_ANSWER,
            "evidence": [],
            "context": "",
            "trace": ["direct:capabilities"],
        }

    # -- 9. output guard ---------------------------------------------------
    def guard_output(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer") or ""
        hits = [ScoredChunk.from_dict(item) for item in (state.get("evidence") or [])]
        route = state.get("route", "retrieve")
        generation_failed = bool(state.get("generation_failed"))

        # Citations are only meaningful when the answer came from retrieved
        # documents. Action results are verified by the tool's own response.
        require_citations = (
            self.settings.require_citations and route == "retrieve" and not generation_failed
        )

        verdict = self.output_guard.review(
            answer, hits, require_citations=require_citations
        )

        if generation_failed:
            # Do not rewrite an infrastructure failure as "the documents don't say
            # that": the honest message is already in the answer, and the sources
            # are still shown so the user can read the policy themselves.
            verdict.issues.append("llm_unavailable")
            verdict.passed = True
            verdict.abstained = False
        elif not verdict.passed:
            logger.info("Output guard failed (%s); degrading to abstention", verdict.issues)
            verdict = self.output_guard.degrade_to_abstention(verdict, hits)

        return {
            "answer": verdict.answer,
            "citations": [citation.to_dict() for citation in verdict.citations],
            "grounding_score": verdict.grounding_score,
            "guard_output": verdict.to_dict(),
            "abstained": verdict.abstained,
            # Commit only the vetted answer to conversation memory.
            "messages": [AIMessage(content=verdict.answer)],
            "trace": [
                f"guard_output:{'pass' if verdict.passed else 'fail'}"
                f"(grounding={verdict.grounding_score:.2f},"
                f" citations={len(verdict.citations)})"
            ],
        }
