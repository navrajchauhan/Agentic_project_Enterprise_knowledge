"""Terminal output helpers.

Kept dependency-free (no rich/textual) so the CLI works in any environment.
Colour is opt-out via ``--no-color`` or the ``NO_COLOR`` environment variable,
and is disabled automatically when stdout is redirected, so piping JSON or logs
stays clean.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
}


def set_color(enabled: bool) -> None:
    """Force colour on/off (used by ``--no-color``)."""
    global _USE_COLOR
    _USE_COLOR = enabled


def paint(text: str, *styles: str) -> str:
    """Wrap ``text`` in ANSI styles when colour is enabled."""
    if not _USE_COLOR or not styles:
        return text
    prefix = "".join(_CODES.get(style, "") for style in styles)
    return f"{prefix}{text}{_CODES['reset']}"


def width(maximum: int = 100) -> int:
    """Usable terminal width, capped for readability."""
    return min(shutil.get_terminal_size((100, 24)).columns, maximum)


def rule(character: str = "-") -> str:
    return paint(character * width(), "grey")


def header(text: str) -> None:
    print(paint(text, "bold", "cyan"))


def info(text: str) -> None:
    print(paint(text, "grey"))


def success(text: str) -> None:
    print(paint(text, "green"))


def warn(text: str) -> None:
    print(paint(text, "yellow"))


def error(text: str) -> None:
    print(paint(text, "red"))


def bullet(text: str, marker: str = "-") -> None:
    print(f"  {paint(marker, 'grey')} {text}")


def render_answer(result: Any, *, show_trace: bool = True) -> None:
    """Pretty-print an ``AgentAnswer``.

    Structure is deliberate: the answer first, then the evidence, then the
    guard verdict.  A reader must be able to see *why* they should believe the
    answer without scrolling past the machinery.
    """
    rule("=")
    header(f"Q: {result.question}")
    confidence = "n/a" if result.abstained else f"{result.confidence:.2f}"
    status = (
        f"mode={result.mode}  route={result.route}  "
        f"{result.elapsed_seconds:.2f}s  confidence={confidence}"
    )
    info(status)
    rule()

    print(result.answer.strip() or paint("(no answer produced)", "red"))

    if result.citations:
        print()
        header("Sources")
        for citation in result.citations:
            label = citation.get("label") or citation.get("doc_id", "?")
            bullet(f"[{citation['marker']}] {label}", marker="*")
    elif result.evidence:
        # No vetted citation (tool route, generation failure or a rejected
        # answer) but documents were retrieved: show them as unverified leads
        # rather than pretending nothing was found.
        print()
        header("Candidate sources (not verified)")
        seen: list[str] = []
        for item in result.evidence:
            label = item.get("citation") or item.get("doc_id", "?")
            if label not in seen:
                seen.append(label)
            if len(seen) == 3:
                break
        for label in seen:
            bullet(label, marker="~")
    elif result.abstained:
        print()
        info("No citation: the knowledge base does not contain this answer.")

    if result.tool_calls:
        print()
        header("Actions taken")
        for call in result.tool_calls:
            state = paint("ok", "green") if call.get("ok") else paint("failed", "red")
            bullet(f"{call['tool']} ({state}): {call['content'][:160]}", marker=">")

    print()
    guard = result.guard or {}
    verdict = "pass" if guard.get("passed") else "review"
    colour = "green" if guard.get("passed") else "yellow"
    issues = guard.get("issues") or []
    issues_text = ", ".join(str(issue) for issue in issues) if issues else "none"
    info(
        f"guard={paint(verdict, colour)}  "
        f"grounding={guard.get('grounding_score', 0):.2f}  "
        f"issues={issues_text}"
    )

    if show_trace and result.trace:
        info("trace: " + " -> ".join(result.trace))


def render_retrieval(hits: list[Any]) -> None:
    """Print a retrieval breakdown (``ask --explain``)."""
    header("Retrieval breakdown")
    if not hits:
        warn("  no hits")
        return
    for rank, hit in enumerate(hits, start=1):
        print(
            f"  {rank}. {paint(hit.chunk.chunk_id, 'cyan')} "
            f"score={hit.score:.4f} "
            f"bm25={hit.bm25_score:.2f}@{hit.bm25_rank} "
            f"dense={hit.dense_score:.2f}@{hit.dense_rank}"
        )
        print(f"     {paint(hit.chunk.citation_label, 'grey')}")
