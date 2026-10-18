"""Command line interface.

Commands
--------
``ingest``     build / rebuild the hybrid index from the knowledge base
``ask``        one question, printed with citations and the guard verdict
``chat``       interactive multi-turn session (one LangGraph thread)
``evaluate``   run the golden set (retrieval metrics + optional LLM judge)
``tools``      list the registered tools, optionally probing the MCP server
``graph``      print the LangGraph topology as Mermaid
``info``       show the effective configuration and index statistics

Everything is reachable through ``python run.py <command>`` from the repo root.
The CLI is intentionally thin: it wires settings to the runner, prints results
and maps failures to exit codes, which keeps it trivial to replace with an HTTP
API later (the ``AgentRunner`` API is the real interface).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .config import Settings, get_settings
from .console import (
    error,
    header,
    info,
    paint,
    render_answer,
    render_retrieval,
    rule,
    set_color,
    success,
    warn,
)
from .logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Build the hybrid index (idempotent unless ``--force``)."""
    from .rag.indexer import Indexer

    settings.ensure_directories()
    indexer = Indexer(settings)
    index = indexer.build(force=args.force)

    header("Index ready")
    info(f"  documents : {len({chunk.doc_id for chunk in index.chunks})}")
    info(f"  chunks    : {len(index)}")
    info(f"  embedder  : {index.embedder_name}")
    info(f"  location  : {settings.index_path}")
    manifest = index.manifest
    if manifest.get("built_at"):
        info(f"  built at  : {manifest['built_at']}")
    success("Run 'python run.py ask \"...\"' to query it.")
    return 0


def _build_runner(settings: Settings) -> Any:
    """Construct the agent runner (expensive: index + models)."""
    from .agent.runner import AgentRunner

    return AgentRunner(settings)


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    """Answer a single question."""
    question = " ".join(args.question).strip()
    if not question:
        error("Please provide a question.")
        return 2

    runner = _build_runner(settings)
    try:
        if args.explain:
            render_retrieval(runner.retriever.search(question, top_k=settings.top_k))
            print()

        result = runner.ask(question, thread_id=args.thread)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            render_answer(result, show_trace=not args.no_trace)
        return 0 if not result.blocked else 1
    finally:
        runner.close()


def cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    """Interactive session; every question shares one LangGraph thread."""
    runner = _build_runner(settings)
    thread = args.thread

    rule("=")
    header("Enterprise document agent")
    info(
        f"  mode={runner.mode}  tools={len(runner.tools.names)}  "
        f"chunks={len(runner.index)}  thread={thread}"
    )
    info("  commands: /exit  /new  /history  /trace  /help")
    rule("=")

    show_trace = True
    try:
        while True:
            try:
                raw = input(paint("\nyou> ", "bold", "green")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                continue
            if raw.startswith("/"):
                command = raw.lower()
                if command in {"/exit", "/quit"}:
                    break
                if command == "/new":
                    thread = f"thread-{len(runner.history(thread))}-{abs(hash(raw)) % 1000}"
                    info(f"Started a new conversation thread: {thread}")
                    continue
                if command == "/history":
                    history = runner.history(thread)
                    if not history:
                        info("(empty)")
                    for message in history:
                        who = "you" if message["role"] == "human" else "agent"
                        print(f"  {paint(who, 'cyan')}: {message['content'][:200]}")
                    continue
                if command == "/trace":
                    show_trace = not show_trace
                    info(f"trace display {'on' if show_trace else 'off'}")
                    continue
                warn("Commands: /exit /new /history /trace /help")
                continue

            result = runner.ask(raw, thread_id=thread)
            render_answer(result, show_trace=show_trace)
    finally:
        runner.close()

    success("Goodbye.")
    return 0


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    """Run the golden evaluation set."""
    from .evaluation.runner import EvaluationRunner

    if args.no_judge:
        settings = settings.model_copy(update={"enable_llm_judge": False})

    runner = EvaluationRunner(settings)
    report = runner.run(retrieval_only=args.retrieval_only, limit=args.limit)

    header("Retrieval quality")
    for key, value in report["retrieval"].items():
        info(f"  {key:<14} {value:.3f}")

    if not args.retrieval_only:
        header("Answer quality")
        for key, value in report["answer"].items():
            info(f"  {key:<20} {value:.3f}")
        judge = report.get("judge", {})
        if judge.get("cases_scored"):
            header("LLM judge (1-5)")
            info(f"  faithfulness       {judge['faithfulness']:.2f}")
            info(f"  relevance          {judge['relevance']:.2f}")
            info(f"  citation support   {judge['citation_support']:.2f}")

    if args.no_save:
        info("\n(report not saved: --no-save)")
        return 0

    json_path, markdown_path = runner.write_report(report)
    print()
    success(f"JSON report     : {json_path}")
    success(f"Markdown report : {markdown_path}")
    return 0


def cmd_tools(args: argparse.Namespace, settings: Settings) -> int:
    """List the tool catalogue, or probe the MCP server."""
    if args.probe_mcp:
        from .mcp.client import probe

        header("MCP tool catalogue (live probe)")
        try:
            print(probe(settings))
        except Exception as exc:  # noqa: BLE001 - CLI reporting
            error(f"MCP probe failed: {exc}")
            return 1
        return 0

    from .rag.indexer import Indexer
    from .rag.retriever import HybridRetriever
    from .tools.provider import build_tool_provider
    from .tools.store import ActionStore

    index = Indexer(settings).build()
    retriever = HybridRetriever(index, settings)
    store = ActionStore(settings.action_db_path)
    provider = build_tool_provider(settings, store=store, retriever=retriever)

    header(f"Tools ({type(provider).__name__})")
    for name in provider.names:
        print(f"  - {paint(name, 'cyan')}")
    info(f"\n{len(provider.names)} tool(s); transport={type(provider).__name__}")

    close = getattr(provider, "close", None)
    if callable(close):
        close()
    return 0


def cmd_graph(args: argparse.Namespace, settings: Settings) -> int:
    """Print the compiled graph as Mermaid."""
    runner = _build_runner(settings)
    try:
        print(runner.diagram())
    finally:
        runner.close()
    return 0


def cmd_info(args: argparse.Namespace, settings: Settings) -> int:
    """Show effective configuration and index stats."""
    from .rag.indexer import Indexer

    header(f"Enterprise RAG Agent v{__version__}")
    info(f"  project root   : {settings.knowledge_base_path.parent}")
    info(f"  knowledge base : {settings.knowledge_base_path}")
    info(f"  index dir      : {settings.index_path}")
    info(f"  offline mode   : {settings.is_offline}")

    indexer = Indexer(settings)
    if indexer.loaded_is_fresh():
        index = indexer.load()
        info(f"  index          : {len(index)} chunks / "
             f"{len({chunk.doc_id for chunk in index.chunks})} documents")
        info(f"  embedder       : {index.embedder_name}")
    else:
        warn("  index          : stale or missing (run: python run.py ingest)")

    info(f"  LLM model      : {settings.llm_model}")
    info(f"  fast model     : {settings.llm_fast_model}")
    info(f"  judge model    : {settings.llm_judge_model}")
    info(f"  tool provider  : {settings.tool_provider}")
    info(f"  checkpoints    : {settings.checkpoint_backend}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Enterprise document agent with hybrid RAG, MCP tools and guardrails.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  python run.py ingest\n'
            '  python run.py ask "What is the hotel cap for domestic travel?"\n'
            "  python run.py ask \"How long are logs kept?\" --explain\n"
            "  python run.py chat\n"
            "  python run.py evaluate --retrieval-only\n"
            "  python run.py tools --probe-mcp\n"
            "  python run.py graph\n"
        ),
    )
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="build or rebuild the hybrid index")
    ingest.add_argument("--force", action="store_true", help="rebuild even if the index is fresh")
    ingest.set_defaults(func=cmd_ingest)

    ask = subparsers.add_parser("ask", help="answer a single question")
    ask.add_argument("question", nargs="+", help="the question to ask")
    ask.add_argument("--thread", default="cli", help="conversation thread id (default: cli)")
    ask.add_argument("--json", action="store_true", help="print the full result as JSON")
    ask.add_argument("--explain", action="store_true", help="show the retrieval breakdown")
    ask.add_argument("--no-trace", action="store_true", help="hide the node trace")
    ask.set_defaults(func=cmd_ask)

    chat = subparsers.add_parser("chat", help="interactive multi-turn session")
    chat.add_argument("--thread", default="cli", help="conversation thread id (default: cli)")
    chat.set_defaults(func=cmd_chat)

    evaluate = subparsers.add_parser("evaluate", help="run the golden evaluation set")
    evaluate.add_argument(
        "--retrieval-only", action="store_true", help="skip answer generation and judging"
    )
    evaluate.add_argument("--limit", type=int, default=None, help="evaluate only the first N cases")
    evaluate.add_argument("--no-judge", action="store_true", help="skip the LLM-as-a-judge step")
    evaluate.add_argument("--no-save", action="store_true", help="do not write report files")
    evaluate.set_defaults(func=cmd_evaluate)

    tools = subparsers.add_parser("tools", help="list tools or probe the MCP server")
    tools.add_argument(
        "--probe-mcp", action="store_true", help="start the MCP server and list its tools"
    )
    tools.set_defaults(func=cmd_tools)

    graph = subparsers.add_parser("graph", help="print the LangGraph topology as Mermaid")
    graph.set_defaults(func=cmd_graph)

    info_parser = subparsers.add_parser("info", help="show configuration and index statistics")
    info_parser.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_color:
        set_color(False)

    settings = get_settings()
    configure_logging(args.log_level or settings.log_level)

    try:
        return int(args.func(args, settings))
    except FileNotFoundError as exc:
        error(str(exc))
        return 1
    except ValueError as exc:
        error(f"Configuration or data problem: {exc}")
        return 1
    except KeyboardInterrupt:
        print()
        warn("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level guard for a friendly message
        error(f"Unexpected error: {type(exc).__name__}: {exc}")
        if args.log_level == "DEBUG":
            raise
        logger.debug("Details", exc_info=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
