"""Shared, dependency-free helpers used across the package."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

#: Small English stop-word list.  We deliberately keep it short: enterprise
#: policy questions contain a lot of "how long", "when must" type phrasing whose
#: function words add noise to BM25 without adding meaning.
STOP_WORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with
    without into over under about as is are was were be been being do does did doing have
    has had having i you he she it we they me him her them my your his its our their who
    whom whose which what when where why how can could should would may might must will
    shall not no nor so such there here also any all both each few more most other some
    only own same too very s t just don now
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:[._'-][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_text(text: str) -> str:
    """Lowercase, unify quotes/dashes and collapse whitespace."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, *, drop_stop_words: bool = True, stem: bool = True) -> list[str]:
    """Tokenise text for BM25 / lexical overlap.

    ``stem`` applies a deliberately tiny suffix normaliser (plurals and common
    verb endings).  A full Porter stemmer is overkill here and makes debug
    output hard to read; this captures the 90% case ("days" -> "day",
    "approved" -> "approv").

    Identifiers such as ``security@company.example`` or ``99.9%`` survive as
    single tokens because the regex allows internal ``._'-``.
    """
    tokens = _WORD_RE.findall(normalize_text(text).lower())
    out: list[str] = []
    for token in tokens:
        if drop_stop_words and token in STOP_WORDS:
            continue
        if stem and token.isalpha() and len(token) > 3:
            # Tiny suffix normaliser. Both corpus and query go through it, so
            # even an imperfect stem ("bonus" -> "bonu") stays symmetric and
            # therefore harmless for matching.
            if token.endswith("ies") and len(token) > 5:
                token = token[:-3] + "y"
            elif token.endswith("ing") and len(token) > 6:
                token = token[:-3]
            elif token.endswith("ed") and len(token) > 5:
                token = token[:-2]
            elif token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
        out.append(token)
    return out


def content_tokens(text: str) -> set[str]:
    """Unique content tokens; used for lexical grounding checks."""
    return set(tokenize(text))


def numbers_in(text: str) -> set[str]:
    """Numeric literals in a text, normalised ("4,000" -> "4000").

    Used by the output guard: numbers are the most common hallucination vector
    in policy answers ("99.9%" vs "99%"), so they get an extra check.
    """
    return {match.replace(",", "") for match in _NUMBER_RE.findall(text)}


def overlap_ratio(reference: set[str], candidate: set[str]) -> float:
    """Share of ``reference`` tokens present in ``candidate`` (0..1)."""
    if not reference:
        return 0.0
    return len(reference & candidate) / len(reference)


# ---------------------------------------------------------------------------
# Hashing / ids
# ---------------------------------------------------------------------------


def stable_hash(text: str, *, length: int = 12) -> str:
    """Deterministic short hash, used for ids and idempotency keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def short_digest(*parts: str) -> str:
    """Hash several strings together (order-sensitive)."""
    return stable_hash("\x1f".join(parts))


# ---------------------------------------------------------------------------
# JSON helpers used for LLM structured output
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from an LLM response.

    LLMs do not always honour "reply with JSON only": they wrap it in code
    fences, add a sentence of preamble, or append a trailing note.  This helper
    tries, in order:

    1. the whole string as JSON,
    2. the contents of a ```json fenced block,
    3. the span from the first ``{`` to the last ``}``.

    Returns ``None`` when nothing parses, so callers can retry or degrade
    gracefully rather than crash.
    """
    if not raw:
        return None

    candidates: list[str] = [raw.strip()]

    fenced = _FENCE_RE.search(raw)
    if fenced:
        candidates.append(fenced.group(1).strip())

    first, last = raw.find("{"), raw.rfind("}")
    if first != -1 and last > first:
        candidates.append(raw[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def truncate(text: str, limit: int, suffix: str = " ...") -> str:
    """Truncate on a word boundary so log/report output stays readable."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut + suffix


# ---------------------------------------------------------------------------
# Routing heuristic (fallback only - the LLM router is the primary decider)
# ---------------------------------------------------------------------------

#: Imperative action request, e.g. "create a P2 ticket", "submit a 250 USD expense".
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|i\s+(?:want|need)\s+(?:you\s+)?to\s+)?"
    r"(?:create|raise|file|open|submit|send|notify|escalate|book|claim|log|look\s?up|reset)\b",
    re.IGNORECASE,
)

#: Phrasings that are unambiguous information requests about a record or person.
_INFO_REQUEST_RE = re.compile(
    r"\b(?:who\s+is|who's|status\s+of\s+(?:ticket\s+)?TCK-?\d+|"
    r"(?:create|raise|file|open)\s+(?:a\s+|the\s+)?(?:p[1-4]\s+)?(?:support\s+)?ticket)\b",
    re.IGNORECASE,
)


def looks_like_tool_request(text: str) -> bool:
    """Heuristic: is this an action request rather than a policy question?

    The trap this avoids: "how long do I have to *submit* an expense report?" is
    a question *about* a policy, not a request to submit one.  So an action verb
    only counts when it opens the sentence (imperative), and explicit
    record/person lookups are matched by phrase.

    Used only as a fallback when the LLM router is unavailable - the router
    prompt is the primary decision maker.
    """
    if _INFO_REQUEST_RE.search(text):
        return True
    return bool(_IMPERATIVE_RE.match(text))
