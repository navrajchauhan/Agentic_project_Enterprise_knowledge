"""Output guard: what may reach the user.

This is the anti-hallucination layer.  It runs on every generated answer and
performs four checks:

1. **Citation resolution** - every ``[n]`` must map to a passage that was
   actually retrieved.  Markers that point nowhere are removed (they are the
   cheapest form of fabricated source).
2. **Citation presence** - a factual answer with no citation fails when
   ``REQUIRE_CITATIONS`` is on.  Abstentions are exempt: "I don't have enough
   information" needs no source.
3. **Sentence grounding** - each answer sentence must share content tokens (and,
   where relevant, numbers) with the evidence it cites.  The share of supported
   sentences is the ``grounding_score``.
4. **PII redaction** - model output is sanitised before display.

The grounding metric is intentionally lexical, not another LLM call: it is
deterministic, free, fast, and hard for a model to talk its way past.  The
LLM-as-a-judge in ``evaluation/`` is the complementary, semantic check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..logging_setup import get_logger
from ..rag.models import Citation, ScoredChunk
from ..utils.text import content_tokens, numbers_in
from .input_guard import redact_pii
from .policy import (
    ABSTENTION_PHRASES,
    SENTENCE_NUMBER_SUPPORT_THRESHOLD,
    SENTENCE_SUPPORT_THRESHOLD,
    UNSUPPORTED_CERTAINTY,
)

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
#: Sentence splitter that survives "99.9%", "USD 1,000." and "e.g.".
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class OutputVerdict:
    """Result of reviewing one generated answer."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounding_score: float = 0.0
    unsupported_sentences: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)
    abstained: bool = False
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "grounding_score": round(self.grounding_score, 3),
            "abstained": self.abstained,
            "issues": self.issues,
            "redactions": self.redactions,
            "unsupported_sentences": self.unsupported_sentences[:5],
            "citations": [citation.to_dict() for citation in self.citations],
        }


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences, keeping list items separate."""
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Bullet/numbered list items are already sentence-sized units.
        if re.match(r"^(?:[-*\u2022]|\d+[.)])\s+", line):
            chunks.append(re.sub(r"^(?:[-*\u2022]|\d+[.)])\s+", "", line))
            continue
        chunks.extend(part for part in _SENTENCE_RE.split(line) if part.strip())
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 15]


def is_abstention(answer: str) -> bool:
    """True when the answer says the documents do not contain the answer."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


class OutputGuard:
    """Validates a generated answer against the evidence it was given."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def review(
        self,
        answer: str,
        hits: list[ScoredChunk],
        *,
        require_citations: bool | None = None,
    ) -> OutputVerdict:
        """Validate ``answer`` against ``hits``.

        ``require_citations`` overrides ``REQUIRE_CITATIONS`` for this call.  The
        agent passes ``False`` for answers produced by the tool route: an action
        result ("Created TCK-1001") is verified by the tool's own response, not
        by a document citation.
        """
        require_citations = (
            self.settings.require_citations if require_citations is None else require_citations
        )
        issues: list[str] = []
        answer = (answer or "").strip()

        # -- 1. PII in the output ------------------------------------------
        # Internal addresses are preserved: they are the company's own contact
        # details and often part of the answer (e.g. where to report an incident).
        redactions: dict[str, int] = {}
        if self.settings.redact_pii:
            answer, redactions = redact_pii(
                answer, preserve_domains=(self.settings.internal_email_domain,)
            )
            if sum(redactions.values()):
                issues.append("pii_redacted_from_output")

        # -- 2. Abstention handling ----------------------------------------
        # An abstention is a correct outcome; it needs no citation.
        if is_abstention(answer):
            return OutputVerdict(
                answer=answer,
                citations=[],
                grounding_score=1.0,
                issues=issues,
                redactions=redactions,
                abstained=True,
                passed=True,
            )

        if not answer:
            return OutputVerdict(
                answer="I could not produce an answer from the retrieved documents.",
                citations=[],
                grounding_score=0.0,
                issues=[*issues, "empty_answer"],
                redactions=redactions,
                abstained=True,
                passed=False,
            )

        # -- 3. No evidence to check against -------------------------------
        # Happens on the tool route or when retrieval found nothing. There is
        # nothing to ground the text in, so it can only pass when citations are
        # not required for this route.
        if not hits:
            issues.append("no_evidence_to_verify")
            return OutputVerdict(
                answer=answer,
                citations=[],
                grounding_score=0.0 if require_citations else 1.0,
                issues=issues,
                redactions=redactions,
                abstained=False,
                passed=not require_citations,
            )

        # -- 4. Resolve citations ------------------------------------------
        valid_markers = {
            int(match.group(1))
            for match in _CITATION_RE.finditer(answer)
            if 1 <= int(match.group(1)) <= len(hits)
        }
        referenced = {
            int(match.group(1)) for match in _CITATION_RE.finditer(answer)
        }
        invalid = referenced - valid_markers
        if invalid:
            issues.append(f"invalid_citations:{sorted(invalid)}")
            # Remove the bogus markers so no fabricated source survives.
            for marker in invalid:
                answer = answer.replace(f"[{marker}]", "")

        citations = self._build_citations(valid_markers, hits)

        # -- 5. Citation presence ------------------------------------------
        if require_citations and not valid_markers:
            issues.append("missing_citations")

        # -- 6. Sentence grounding -----------------------------------------
        grounding, unsupported = self._grounding(answer, hits, valid_markers)
        if grounding < self.settings.min_grounding_score:
            issues.append(f"low_grounding:{grounding:.2f}")

        # -- 7. Over-certain claims ----------------------------------------
        lowered = answer.lower()
        for phrase in UNSUPPORTED_CERTAINTY:
            if phrase in lowered:
                issues.append(f"unsupported_certainty:{phrase}")

        failed_hard = "missing_citations" in issues or "empty_answer" in issues
        low_grounding = grounding < self.settings.min_grounding_score
        passed = not failed_hard and not low_grounding

        if not passed:
            logger.info(
                "Output guard rejected answer (grounding=%.2f, issues=%s)", grounding, issues
            )

        return OutputVerdict(
            answer=answer,
            citations=citations,
            grounding_score=grounding,
            unsupported_sentences=unsupported,
            issues=issues,
            redactions=redactions,
            abstained=False,
            passed=passed,
        )

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _build_citations(markers: set[int], hits: list[ScoredChunk]) -> list[Citation]:
        """Turn validated marker numbers into ``Citation`` objects."""
        citations: list[Citation] = []
        for marker in sorted(markers):
            chunk = hits[marker - 1].chunk
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    section=chunk.section,
                )
            )
        return citations

    def _grounding(
        self,
        answer: str,
        hits: list[ScoredChunk],
        valid_markers: set[int],
    ) -> tuple[float, list[str]]:
        """Score the share of answer sentences supported by the evidence.

        A sentence is supported when it shares enough content tokens with the
        passage(s) it cites.  When a sentence cites nothing, we fall back to the
        whole retrieved context: that is weaker evidence, so it is still scored,
        and a global shortfall drags ``grounding_score`` down (which is what
        makes the guard reject confidently-stated but unretrieved claims).
        """
        sentences = split_sentences(answer)
        if not sentences:
            return 0.0, []

        all_context_tokens: set[str] = set()
        all_context_numbers: set[str] = set()
        tokens_by_hit: dict[int, set[str]] = {}
        numbers_by_hit: dict[int, set[str]] = {}

        for index, hit in enumerate(hits, start=1):
            hit_tokens = content_tokens(hit.context_text)
            hit_numbers = numbers_in(hit.context_text)
            tokens_by_hit[index] = hit_tokens
            numbers_by_hit[index] = hit_numbers
            all_context_tokens |= hit_tokens
            all_context_numbers |= hit_numbers

        supported = 0
        unsupported: list[str] = []

        for sentence in sentences:
            markers = {
                int(match.group(1))
                for match in _CITATION_RE.finditer(sentence)
                if int(match.group(1)) in valid_markers
            }

            if markers:
                reference_tokens: set[str] = set()
                reference_numbers: set[str] = set()
                for marker in markers:
                    reference_tokens |= tokens_by_hit.get(marker, set())
                    reference_numbers |= numbers_by_hit.get(marker, set())
            else:
                reference_tokens = all_context_tokens
                reference_numbers = all_context_numbers

            # Strip the citation markers before tokenising: "[2]" is a pointer to
            # evidence, not a numeric claim, and treating it as one would flag
            # every correctly cited sentence as unsupported.
            claim = _CITATION_RE.sub(" ", sentence)
            sentence_tokens = content_tokens(claim)
            sentence_numbers = numbers_in(claim)

            if not sentence_tokens:
                continue

            overlap = len(sentence_tokens & reference_tokens) / len(sentence_tokens)
            numbers_ok = sentence_numbers.issubset(reference_numbers) if sentence_numbers else True

            # A sentence that introduces a number absent from the evidence is
            # the single most common hallucination shape, so it is scored
            # strictly even when the wording overlaps.
            if not numbers_ok:
                unsupported.append(sentence)
                continue

            if overlap >= SENTENCE_SUPPORT_THRESHOLD or (
                sentence_numbers
                and overlap >= SENTENCE_NUMBER_SUPPORT_THRESHOLD
                and numbers_ok
            ):
                supported += 1
            else:
                unsupported.append(sentence)

        return supported / len(sentences), unsupported

    def degrade_to_abstention(
        self, verdict: OutputVerdict, hits: list[ScoredChunk]
    ) -> OutputVerdict:
        """Replace a failed answer with an honest abstention plus pointers.

        Used by the runner when the guard rejects an answer: the user still gets
        the retrieved sources (so they can read the policy themselves) but no
        unsupported prose.
        """
        if not hits:
            message = (
                "I don't have enough information in the knowledge base to answer that. "
                "No relevant documents were found."
            )
        else:
            labels = sorted({hit.chunk.citation_label for hit in hits[:3]})
            sources = "; ".join(labels)
            message = (
                "I don't have enough information in the knowledge base to answer that "
                f"reliably. The closest documents I found were: {sources}. "
                "Please refine the question or check those sections directly."
            )

        return OutputVerdict(
            answer=message,
            citations=[],
            grounding_score=verdict.grounding_score,
            unsupported_sentences=verdict.unsupported_sentences,
            issues=[*verdict.issues, "degraded_to_abstention"],
            redactions=verdict.redactions,
            abstained=True,
            passed=True,
        )
