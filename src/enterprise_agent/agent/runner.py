"""Agent runner: one object that owns the whole wired-up system.

This is the entry point every other layer (CLI, evaluation harness, tests) uses.
It assembles:

    Settings -> Indexer -> HybridIndex -> HybridRetriever
             -> ActionStore -> tool provider (MCP or in-process)
             -> Groq models (or the offline double)
             -> AgentNodes -> compiled LangGraph with a checkpointer

and exposes ``ask()``, which returns a structured :class:`AgentAnswer` instead of
a bare string: callers need the citations, the guard verdict and the tool calls
to render a trustworthy answer.

Statefulness lives here: the checkpointer keeps one state per ``thread_id``, so
asking a follow-up question on the same thread replays the message history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

from ..config import Settings, get_settings
from ..guardrails.input_guard import InputGuard
from ..guardrails.output_guard import OutputGuard
from ..llm.base import build_chat_model, message_text
from ..logging_setup import configure_logging, get_logger
from ..rag.indexer import HybridIndex, Indexer
from ..rag.retriever import HybridRetriever
from ..tools.provider import build_tool_provider
from ..tools.store import ActionStore
from .graph import build_graph, graph_diagram
from .nodes import AgentNodes
from .state import initial_state

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class AgentAnswer:
    """Everything the caller needs to trust (or audit) one answer."""

    question: str
    answer: str
    route: str = "retrieve"
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    blocked: bool = False
    block_reason: str = ""
    trace: list[str] = field(default_factory=list)
    guard: dict[str, Any] = field(default_factory=dict)
    thread_id: str = "default"
    elapsed_seconds: float = 0.0
    mode: str = "live"

    # -- rendering helpers -------------------------------------------------
    @property
    def sources(self) -> list[str]:
        """Distinct human-readable sources, in citation order."""
        seen: list[str] = []
        for citation in self.citations:
            label = citation.get("label") or citation.get("doc_id", "?")
            if label not in seen:
                seen.append(label)
        return seen

    def sources_text(self) -> str:
        """``[1] hr_policy.md - Annual Leave`` style source list."""
        if not self.citations:
            return ""
        return "\n".join(
            f"[{citation['marker']}] {citation.get('label', citation.get('doc_id', '?'))}"
            for citation in self.citations
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "citations": self.citations,
            "tool_calls": self.tool_calls,
            "confidence": round(self.confidence, 3),
            "abstained": self.abstained,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "trace": self.trace,
            "guard": self.guard,
            "thread_id": self.thread_id,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "mode": self.mode,
        }


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------


def build_checkpointer(settings: Settings) -> Any:
    """Return the conversation-memory backend.

    ``memory`` keeps threads in-process (fine for a CLI session and for tests).
    ``sqlite`` persists them to disk so a restart keeps the history; it needs
    ``langgraph-checkpoint-sqlite``, so a missing extra downgrades with a warning
    instead of breaking the agent.
    """
    if settings.checkpoint_backend == "sqlite":
        try:
            import sqlite3  # noqa: PLC0415 - only needed for this backend

            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415

            settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
            logger.info("Using SQLite checkpointer at %s", settings.checkpoint_path)
            return SqliteSaver(connection)
        except ImportError as exc:
            logger.warning(
                "CHECKPOINT_BACKEND=sqlite requires langgraph-checkpoint-sqlite (%s); "
                "falling back to in-memory threads.",
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open the SQLite checkpointer (%s); using memory.", exc)

    # ``MemorySaver`` was renamed to ``InMemorySaver`` across LangGraph releases;
    # accept either so the project is not pinned to one minor version.
    try:
        from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415

        return MemorySaver()
    except ImportError:  # pragma: no cover - depends on LangGraph version
        from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

        return InMemorySaver()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AgentRunner:
    """Wired agent: retrieval + tools + graph + memory."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        index: HybridIndex | None = None,
        retriever: HybridRetriever | None = None,
        tools: Any | None = None,
        llm: Any | None = None,
        fast_llm: Any | None = None,
        build_index_if_missing: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level)
        self.settings.ensure_directories()

        # -- retrieval -----------------------------------------------------
        if retriever is not None:
            self.retriever = retriever
            self.index = index or retriever.index
        else:
            indexer = Indexer(self.settings)
            if build_index_if_missing:
                self.index = index or indexer.build()
            else:
                self.index = index or indexer.load()
            self.retriever = HybridRetriever(self.index, self.settings)

        # -- actions -------------------------------------------------------
        self.store = ActionStore(self.settings.action_db_path)
        self.tools = tools or build_tool_provider(
            self.settings, store=self.store, retriever=self.retriever
        )

        # -- models --------------------------------------------------------
        self.llm = llm or build_chat_model(self.settings, purpose="primary")
        self.fast_llm = fast_llm or build_chat_model(self.settings, purpose="fast")

        # -- graph ---------------------------------------------------------
        self.nodes = AgentNodes(
            settings=self.settings,
            llm=self.llm,
            fast_llm=self.fast_llm,
            retriever=self.retriever,
            tools=self.tools,
            input_guard=InputGuard(self.settings),
            output_guard=OutputGuard(self.settings),
        )
        self.checkpointer = build_checkpointer(self.settings)
        self.graph = build_graph(self.nodes, self.settings, checkpointer=self.checkpointer)

    # -- properties --------------------------------------------------------
    @property
    def mode(self) -> str:
        """``offline`` when running without a Groq key."""
        return "offline" if self.settings.is_offline else "live"

    @property
    def tool_transport(self) -> str:
        return type(self.tools).__name__

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "documents": len({chunk.doc_id for chunk in self.index.chunks}),
            "chunks": len(self.index),
            "embedder": self.index.embedder_name,
            "top_k": self.settings.top_k,
            "bm25_weight": self.settings.bm25_weight,
            "dense_weight": self.settings.dense_weight,
            "tools": self.tools.names,
            "tool_transport": self.tool_transport,
            "checkpoint_backend": self.settings.checkpoint_backend,
            "max_tool_steps": self.settings.max_tool_steps,
            "max_query_rewrites": self.settings.max_query_rewrites,
        }

    def diagram(self) -> str:
        return graph_diagram(self.graph)

    # -- main entry point --------------------------------------------------
    def ask(self, question: str, *, thread_id: str = "default") -> AgentAnswer:
        """Run one turn of the agent and return a structured answer."""
        started = time.perf_counter()
        state = initial_state(question, thread_id)
        config = {
            "configurable": {"thread_id": thread_id},
            # Backstop against a runaway loop; the graph's own counters are the
            # primary limit.
            "recursion_limit": 40,
        }

        try:
            final = self.graph.invoke(state, config=config)
        except Exception as exc:  # noqa: BLE001 - must never crash the caller
            logger.exception("Agent invocation failed")
            return AgentAnswer(
                question=question,
                answer=(
                    "Something went wrong while processing that request. "
                    f"({type(exc).__name__}: {exc})"
                ),
                abstained=True,
                thread_id=thread_id,
                elapsed_seconds=time.perf_counter() - started,
                mode=self.mode,
                trace=["error:graph_invocation_failed"],
            )

        elapsed = time.perf_counter() - started
        answer = AgentAnswer(
            question=question,
            answer=final.get("answer", "") or "",
            route=final.get("route", "retrieve"),
            citations=final.get("citations") or [],
            evidence=final.get("evidence") or [],
            tool_calls=final.get("tool_calls") or [],
            confidence=float(final.get("grounding_score") or 0.0),
            abstained=bool(final.get("abstained")),
            blocked=bool(final.get("blocked")),
            block_reason=final.get("block_reason", "") or "",
            trace=final.get("trace") or [],
            guard=final.get("guard_output") or final.get("guard_input") or {},
            thread_id=thread_id,
            elapsed_seconds=elapsed,
            mode=self.mode,
        )

        logger.info(
            "turn done in %.2fs route=%s citations=%d grounding=%.2f",
            elapsed,
            answer.route,
            len(answer.citations),
            answer.confidence,
        )
        return answer

    # -- conversation management ------------------------------------------
    def history(self, thread_id: str = "default") -> list[dict[str, str]]:
        """Return the stored conversation for a thread."""
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        messages: list[BaseMessage] = (snapshot.values or {}).get("messages", []) if snapshot else []
        history: list[dict[str, str]] = []
        for message in messages:
            role = getattr(message, "type", "unknown")
            if role in {"human", "ai"}:
                history.append({"role": role, "content": message_text(message)})
        return history

    def reset_thread(self, thread_id: str = "default") -> None:
        """Forget a thread's conversation.

        Uses the checkpointer's ``delete_thread`` when available (LangGraph 1.x).
        Older/custom checkpointers may not implement it, in which case callers
        should simply use a new ``thread_id`` - that is always equivalent.
        """
        delete = getattr(self.checkpointer, "delete_thread", None)
        if callable(delete):
            try:
                delete(thread_id)
                logger.info("Cleared conversation thread %s", thread_id)
                return
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning("Could not delete thread %s: %s", thread_id, exc)
        logger.info(
            "This checkpointer cannot delete threads; use a new thread_id to start fresh."
        )

    def close(self) -> None:
        """Release subprocess resources (MCP server, SQLite handles)."""
        close = getattr(self.tools, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> AgentRunner:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
