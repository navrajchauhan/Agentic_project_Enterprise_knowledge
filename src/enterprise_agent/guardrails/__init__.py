"""Guardrails: input validation, PII handling and answer grounding checks.

The two guards are deliberately deterministic.  A guardrail implemented as
another LLM prompt is a suggestion; a guardrail implemented as code with tests
is a control.  The LLM-as-a-judge in ``evaluation/`` exists to *measure* answer
quality, not to gate it.
"""

from __future__ import annotations

from .input_guard import InputGuard, InputVerdict, redact_pii
from .output_guard import OutputGuard, OutputVerdict, is_abstention

__all__ = [
    "InputGuard",
    "InputVerdict",
    "OutputGuard",
    "OutputVerdict",
    "is_abstention",
    "redact_pii",
]
