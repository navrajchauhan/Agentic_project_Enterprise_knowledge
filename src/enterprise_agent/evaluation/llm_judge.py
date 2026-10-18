"""LLM-as-a-judge for answer quality.

Retrieval metrics only measure whether the right *documents* came back.  They say
nothing about whether the answer was faithful to them.  This judge scores three
things on a 1-5 scale:

``faithfulness``
    Is every claim supported by the retrieved evidence? (the hallucination axis)
``relevance``
    Does the answer actually address the question?
``citation_support``
    Do the cited passages really contain the claims they are attached to?

Two implementation choices worth noting:

* **Separate judge model.**  A model grading its own output is systematically
  generous.  ``LLM_JUDGE_MODEL`` defaults to a different (usually stronger) model
  than ``LLM_FAST_MODEL``, and can be pointed elsewhere entirely.
* **Evidence + answer are both in the prompt**, and the judge is asked for
  *unsupported claims by name*.  Naming the specific sentence is much harder to
  fake than emitting a number, and it is the actionable output for debugging.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..config import Settings
from ..llm.base import build_chat_model, invoke_json
from ..logging_setup import get_logger

logger = get_logger(__name__)

JUDGE_SYSTEM = """[[INTENT:judge]]
You are a strict evaluator of enterprise question-answering systems.

Score the ANSWER against the EVIDENCE on three axes, each 1-5:
- faithfulness: 5 = every claim is directly supported by the evidence; 1 = claims are
  invented or contradicted. If the answer says it lacks information and the evidence indeed
  lacks it, faithfulness = 5.
- relevance: 5 = fully answers the question asked; 1 = off topic.
- citation_support: 5 = each cited passage really contains the claim it is attached to;
  1 = citations do not support the claims. If there are no citations and none were needed, 3.

Also list the specific claims that the evidence does not support (empty list if none).
Be harsh: a plausible-sounding number that is absent from the evidence is unfaithful.

Reply with ONLY this JSON object:
{"faithfulness": 1-5, "relevance": 1-5, "citation_support": 1-5,
 "unsupported_claims": ["..."], "rationale": "<max 40 words>"}
"""

JUDGE_HUMAN = """QUESTION: {question}

ANSWER:
{answer}

EVIDENCE:
{evidence}

Score the answer.
"""


class JudgeScores(BaseModel):
    """Validated judge output."""

    faithfulness: int = Field(default=3, ge=1, le=5)
    relevance: int = Field(default=3, ge=1, le=5)
    citation_support: int = Field(default=3, ge=1, le=5)
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str = ""

    @property
    def mean(self) -> float:
        return (self.faithfulness + self.relevance + self.citation_support) / 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "citation_support": self.citation_support,
            "mean": round(self.mean, 2),
            "unsupported_claims": self.unsupported_claims,
            "rationale": self.rationale,
        }


class LLMJudge:
    """Scores answers with a judge model (Groq in live mode, heuristics offline)."""

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self.settings = settings
        self.model = model or build_chat_model(settings, purpose="judge")

    def score(
        self,
        *,
        question: str,
        answer: str,
        evidence_text: str,
    ) -> JudgeScores | None:
        """Return judge scores, or ``None`` when the judge could not answer.

        ``None`` is deliberate: a failed judge must be reported as "unscored"
        rather than silently counted as a pass.
        """
        if not answer.strip():
            return JudgeScores(
                faithfulness=1,
                relevance=1,
                citation_support=1,
                rationale="empty answer",
            )

        # Truncate the evidence: the judge only needs enough to check the claims,
        # and long contexts make small judges lazy.
        evidence = evidence_text[:6000] if evidence_text else "(no evidence retrieved)"

        scores = invoke_json(
            self.model,
            system=JUDGE_SYSTEM,
            human=JUDGE_HUMAN.format(question=question, answer=answer[:2500], evidence=evidence),
            schema=JudgeScores,
        )
        if scores is None:
            logger.warning("Judge returned no valid JSON for question %r", question[:60])
        return scores
