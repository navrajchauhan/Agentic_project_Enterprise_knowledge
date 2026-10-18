"""Guardrail policy: patterns, thresholds and shared constants.

Keeping the rules in one module means a security review has exactly one file to
read, and tests can assert against the same constants the runtime uses.

Threat model for an enterprise document agent
---------------------------------------------
1. **Prompt injection** - text inside a question (or inside a retrieved
   document) tries to override the system prompt: "ignore previous
   instructions", "you are now DAN", "print your system prompt".
2. **Data exfiltration** - questions that ask the agent to dump the corpus or
   the employee directory rather than answer a business question.
3. **PII leakage** - user input and model output are scanned for emails, phone
   numbers, card numbers and national ids; input is redacted before it reaches
   the LLM, output is redacted before it reaches the user.
4. **Ungrounded answers** - the output guard requires citations that resolve to
   retrieved chunks and measures lexical support for each sentence.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Prompt injection / jailbreak
# ---------------------------------------------------------------------------

#: ``(pattern, label)`` pairs.  Patterns are deliberately narrow: a broad
#: "ignore" rule would block legitimate questions such as "which approvals can
#: I ignore during an emergency?".
INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+"
                r"(?:instructions?|prompts?|rules?|context)", re.I), "ignore_instructions"),
    (re.compile(r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+"
                r"(?:instructions?|prompts?|rules?)", re.I), "ignore_instructions"),
    (re.compile(r"\byou\s+are\s+now\s+(?:a|an|the|in)\b", re.I), "role_override"),
    (re.compile(r"\b(?:act|behave|respond)\s+as\s+(?:a|an|the)\s+(?:unrestricted|unfiltered|"
                r"developer|admin|root|jailbroken)", re.I), "role_override"),
    (re.compile(r"\b(?:developer|god|dan|jailbreak|unrestricted)\s+mode\b", re.I), "jailbreak_mode"),
    (re.compile(r"\b(?:reveal|show|print|repeat|leak|output)\s+(?:me\s+)?(?:your|the)\s+"
                r"(?:system|developer|hidden|initial)\s+(?:prompt|instructions?|message)", re.I),
     "prompt_extraction"),
    (re.compile(r"\bwhat\s+(?:are|is)\s+your\s+(?:system\s+prompt|instructions?)\b", re.I),
     "prompt_extraction"),
    (re.compile(r"\b(?:pretend|imagine)\s+(?:that\s+)?you\s+(?:have|are)\s+no\s+"
                r"(?:rules|restrictions|guardrails)", re.I), "guardrail_bypass"),
    (re.compile(r"</?(?:system|assistant|tool)\s*>", re.I), "role_tag_injection"),
    (re.compile(r"\bBEGIN\s+SYSTEM\s+PROMPT\b", re.I), "prompt_extraction"),
]

# ---------------------------------------------------------------------------
# Data exfiltration
# ---------------------------------------------------------------------------

EXFILTRATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:dump|export|exfiltrate|download)\s+(?:the\s+)?(?:entire|whole|all|full)\s+"
                r"(?:database|corpus|knowledge\s*base|document\s+store)", re.I), "dump_corpus"),
    (re.compile(r"\b(?:list|show|print|give)\s+(?:me\s+)?(?:all|every)\s+(?:the\s+)?"
                r"(?:salaries|salary|compensation|payroll|passwords?|api\s*keys?|secrets?)", re.I),
     "sensitive_data_request"),
    (re.compile(r"\b(?:all|every)\s+employees?'?\s+(?:salary|salaries|compensation|passwords?)", re.I),
     "sensitive_data_request"),
    (re.compile(r"\b(?:bypass|skip)\s+(?:the\s+)?(?:approval|authentication|mfa|guardrail)", re.I),
     "bypass_request"),
]

# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------

#: ``name -> (compiled pattern, replacement token)``.  Order matters: the card
#: pattern runs before the phone pattern so a 16-digit card number is not
#: partially matched as a phone number.
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<EMAIL>"),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<CARD>"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    ("password_assignment", re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", re.I), "<PASSWORD>"),
    ("phone", re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{6,12}(?!\w)"),
     "<PHONE>"),
    ("api_key", re.compile(r"\b(?:gsk_|sk-|AKIA)[A-Za-z0-9_\-]{8,}\b"), "<API_KEY>"),
]

# ---------------------------------------------------------------------------
# Output-side constants
# ---------------------------------------------------------------------------

#: Phrases that mean "the documents do not answer this".  When the answer
#: contains one of these, the output guard accepts a citation-free response
#: instead of failing it - abstaining is a *success* for a grounded agent.
ABSTENTION_PHRASES: tuple[str, ...] = (
    "do not have enough information",
    "don't have enough information",
    "no information",
    "not covered in",
    "not specified in",
    "could not find",
    "cannot find",
    "no relevant",
    "insufficient evidence",
    "not mentioned in",
)

#: Claims that must never be made from policy documents alone.
UNSUPPORTED_CERTAINTY = (
    "guaranteed approved",
    "approval is guaranteed",
)

#: Minimum lexical overlap between an answer sentence and its cited evidence.
SENTENCE_SUPPORT_THRESHOLD = 0.5
#: Looser threshold used when the sentence also matches the evidence's numbers.
SENTENCE_NUMBER_SUPPORT_THRESHOLD = 0.3
