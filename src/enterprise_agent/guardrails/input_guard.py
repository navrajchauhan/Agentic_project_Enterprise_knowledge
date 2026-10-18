"""Input guard: what may reach the agent.

Order of checks (cheapest and most severe first):

1. empty / oversized input,
2. prompt injection and jailbreak attempts,
3. data-exfiltration requests,
4. PII redaction (does not block - it sanitises).

Every decision is recorded in an ``InputVerdict`` so the CLI can show *why* a
request was refused and the evaluation harness can assert on it.  Refusals are
deliberately explicit: a silent "I can't help with that" teaches the user
nothing and hides the guardrail from operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..logging_setup import get_logger
from ..utils.text import normalize_text
from .policy import EXFILTRATION_PATTERNS, INJECTION_PATTERNS, PII_PATTERNS

logger = get_logger(__name__)


def redact_pii(text: str, *, preserve_domains: tuple[str, ...] = ()) -> tuple[str, dict[str, int]]:
    """Replace PII-looking spans with placeholders.

    Returns the sanitised text and a count per detector (never the values
    themselves - the counts are safe to log, the values are not).

    ``preserve_domains`` keeps email addresses on the company's own domain
    intact.  Internal contact addresses are the organisation's own information,
    not user PII - and in a policy answer "report it to security@company.example"
    the address *is* the answer, so redacting it would destroy the content.
    Input is always redacted in full; only trusted output paths pass a domain.
    """
    counts: dict[str, int] = {}
    sanitised = text
    for name, pattern, replacement in PII_PATTERNS:
        if name == "email" and preserve_domains:
            redacted_here = 0

            def _keep_internal(match: re.Match[str]) -> str:
                nonlocal redacted_here
                address = match.group(0).lower()
                if any(address.endswith(domain.lower()) for domain in preserve_domains):
                    return match.group(0)  # internal contact detail: keep it
                redacted_here += 1
                return replacement

            sanitised = pattern.sub(_keep_internal, sanitised)
            hits = redacted_here
        else:
            sanitised, hits = pattern.subn(replacement, sanitised)
        if hits:
            counts[name] = counts.get(name, 0) + hits
    return sanitised, counts


@dataclass
class InputVerdict:
    """Result of checking one user request."""

    allowed: bool
    query: str  # normalised, PII-redacted query that the pipeline will use
    original: str
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)
    risk_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "flags": self.flags,
            "redactions": self.redactions,
            "risk_score": round(self.risk_score, 3),
        }

    @property
    def refusal_message(self) -> str:
        """User-facing explanation for a blocked request."""
        if self.reason == "prompt_injection":
            return (
                "I can't follow instructions that try to change my behaviour or reveal my "
                "configuration. Ask me a question about the company policies and I'll answer "
                "from the documents."
            )
        if self.reason == "data_exfiltration_attempt":
            return (
                "I can't bulk-export documents or personal data. I can look up a specific "
                "policy section, or a named colleague in the directory."
            )
        if self.reason == "input_too_long":
            return (
                "That request is too long for me to process. Please shorten it to the actual "
                "question - one or two sentences is plenty."
            )
        if self.reason == "empty_query":
            return "Please ask a question, for example: 'What is the hotel cap for domestic travel?'"
        return "I can't process that request."


class InputGuard:
    """Validates and sanitises incoming user text."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check(self, text: str) -> InputVerdict:
        original = text or ""
        normalised = normalize_text(original)

        # 1. Shape checks ---------------------------------------------------
        if not normalised or not re.search(r"[A-Za-z0-9]", normalised):
            return InputVerdict(
                allowed=False,
                query=normalised,
                original=original,
                reason="empty_query",
            )

        if len(normalised) > self.settings.max_input_chars:
            return InputVerdict(
                allowed=False,
                query=normalised[: self.settings.max_input_chars],
                original=original,
                reason="input_too_long",
                risk_score=0.2,
            )

        flags: list[str] = []
        risk = 0.0

        # 2. Injection / jailbreak -----------------------------------------
        if self.settings.block_prompt_injection:
            for pattern, label in INJECTION_PATTERNS:
                if pattern.search(normalised):
                    flags.append(f"injection:{label}")
                    risk = max(risk, 0.9)
            if risk >= 0.9:
                logger.warning("Blocked prompt injection attempt (flags=%s)", flags)
                return InputVerdict(
                    allowed=False,
                    query=normalised,
                    original=original,
                    reason="prompt_injection",
                    flags=flags,
                    risk_score=risk,
                )

        # 3. Exfiltration ---------------------------------------------------
        for pattern, label in EXFILTRATION_PATTERNS:
            if pattern.search(normalised):
                flags.append(f"exfiltration:{label}")
                risk = max(risk, 0.8)
        if "exfiltration:bypass_request" in flags:
            logger.warning("Blocked bypass/exfiltration attempt (flags=%s)", flags)
            return InputVerdict(
                allowed=False,
                query=normalised,
                original=original,
                reason="data_exfiltration_attempt",
                flags=flags,
                risk_score=risk,
            )
        if any(flag.startswith("exfiltration:") for flag in flags):
            logger.warning("Blocked bulk data request (flags=%s)", flags)
            return InputVerdict(
                allowed=False,
                query=normalised,
                original=original,
                reason="data_exfiltration_attempt",
                flags=flags,
                risk_score=risk,
            )

        # 4. PII redaction ---------------------------------------------------
        query = normalised
        redactions: dict[str, int] = {}
        if self.settings.redact_pii:
            query, redactions = redact_pii(normalised)
            if redactions:
                flags.append("pii_redacted")
                risk = max(risk, 0.3)
                logger.info("Redacted PII from input: %s", redactions)

        return InputVerdict(
            allowed=True,
            query=query,
            original=original,
            flags=flags,
            redactions=redactions,
            risk_score=risk,
        )
