"""Deterministic offline chat model - the project's test double.

Why does this exist?
    1. **Testability.**  The whole LangGraph workflow, the guardrails and the
       evaluation harness can be exercised in CI with no API key, no network and
       no flakiness.  Graph tests assert on *behaviour* (did the tool run, is
       the answer cited, did the guard reject it), which is what matters.
    2. **Demoability.**  Cloning the repo without a Groq key still gives a
       working agent instead of an error screen.

How it works
    Every prompt template carries a machine-readable marker, e.g.
    ``[[INTENT:route]]``.  This class reads that marker and implements a small,
    deterministic policy for each intent:

    ==================  =======================================================
    ``route``           keyword heuristic -> retrieve / tool / direct
    ``grade``           lexical overlap between question and evidence
    ``rewrite``         keyword extraction (stop-words removed)
    ``answer``          *extractive*: quotes the best-matching evidence sentence
    ``tools``           keyword -> tool mapping with sensible arguments
    ``judge``           lexical faithfulness/relevance scoring
    ==================  =======================================================

    The extractive answerer is intentionally dumb but **grounded by
    construction**: it can only quote text it was given, so it never invents a
    number.  That is exactly what makes it useful for testing the citation and
    grounding paths.

This is a test double, not a fallback product: with a real key the agent uses
Groq for every step.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ..utils.text import content_tokens, looks_like_tool_request, numbers_in, tokenize

_QUESTION_RE = re.compile(r"QUESTION:\s*(.+?)(?:\n\n|$)", re.DOTALL)
_EVIDENCE_RE = re.compile(r"EVIDENCE:\s*(.*)", re.DOTALL)
_BLOCK_RE = re.compile(
    r"^\[(\d+)\]\s*source:\s*(.+?)\s*\(chunk_id=([^,]+),\s*fused_score=([\d.]+)\)\s*$",
    re.MULTILINE,
)
_ABSTAIN = (
    "I don't have enough information in the knowledge base to answer that. "
    "The retrieved policy sections do not state it."
)
#: Minimum best-sentence overlap before the extractive answerer commits to an
#: answer. Below this the evidence is about a different topic, so it abstains -
#: which is exactly what a well-prompted real model is asked to do.
_ABSTAIN_THRESHOLD = 0.15
#: Share of the question's content words that must appear somewhere in the
#: retrieved evidence before an answer is attempted at all.
_GLOBAL_COVERAGE_THRESHOLD = 0.25
#: Share of the question a *single* evidence block must cover to be answerable.
_BLOCK_COVERAGE_THRESHOLD = 0.4
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|who are you|"
    r"what can you do|help)\b[\s!.?]*$",
    re.IGNORECASE,
)


def _parse_question(text: str) -> str:
    match = _QUESTION_RE.search(text)
    return match.group(1).strip() if match else text.strip()[:400]


def _parse_evidence(text: str) -> list[dict[str, Any]]:
    """Split a rendered context block into ``{marker, source, body}`` records.

    Falls back to treating the whole block as a single unnamed passage when the
    caller used a different rendering (the evaluation harness does), so the
    heuristic paths below still work with any prompt that carries an
    ``EVIDENCE:`` section.
    """
    match = _EVIDENCE_RE.search(text)
    block = match.group(1) if match else ""
    # Drop the trailing instruction line ("Answer in ...") if present.
    block = re.split(r"\n\s*(?:Answer|Decide|Reply|Score)\b", block)[0]

    headers = list(_BLOCK_RE.finditer(block))
    if not headers:
        body = block.strip()
        return [{"marker": 1, "source": "evidence", "body": body}] if body else []

    evidence: list[dict[str, Any]] = []
    for index, header in enumerate(headers):
        start = header.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(block)
        evidence.append(
            {
                "marker": int(header.group(1)),
                "source": header.group(2).strip(),
                "body": block[start:end].strip(),
            }
        )
    return evidence


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 12]


def _best_sentence(question: str, body: str) -> tuple[str, float]:
    """Pick the evidence sentence that best answers the question."""
    question_tokens = content_tokens(question)
    question_numbers = numbers_in(question)
    if not question_tokens:
        sentences = _split_sentences(body)
        return (sentences[0] if sentences else body, 0.0)

    best_sentence, best_score = "", 0.0
    for sentence in _split_sentences(body):
        sentence_tokens = content_tokens(sentence)
        if not sentence_tokens:
            continue
        overlap = len(sentence_tokens & question_tokens) / len(question_tokens)
        # A sentence sharing the question's numbers is almost always the answer
        # ("180" for a hotel-cap question), so give it a decisive bonus.
        if question_numbers and question_numbers & numbers_in(sentence):
            overlap += 0.5
        if overlap > best_score:
            best_sentence, best_score = sentence, overlap
    return best_sentence, best_score


class OfflineChatModel(BaseChatModel):
    """Deterministic, dependency-free stand-in for a chat model."""

    response_style: str = "extractive"

    @property
    def _llm_type(self) -> str:
        return "offline-extractive"

    # ------------------------------------------------------------------ tools
    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        """Store the schemas so ``_generate`` can emit tool calls.

        Real providers convert schemas to their own format; here the raw schema
        list is passed through ``bind`` and read back in ``_generate``.
        """
        return self.bind(tools=tools, tool_choice=tool_choice, **kwargs)

    # --------------------------------------------------------------- generate
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = "\n".join(
            str(message.content) for message in messages if getattr(message, "content", None)
        )
        intent = self._intent(text)
        tools = kwargs.get("tools") or []

        # --- tool loop ----------------------------------------------------
        if tools:
            # Only look at messages *after* the latest user turn: earlier turns in
            # the same thread may already contain tool calls and results, and
            # reusing those would make the second action in a thread a no-op.
            human_positions = [
                index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
            ]
            turn_start = human_positions[-1] if human_positions else 0
            trailing = messages[turn_start + 1 :]
            tool_messages = [m for m in trailing if isinstance(m, ToolMessage)]
            already_called = any(
                getattr(message, "tool_calls", None)
                for message in trailing
                if isinstance(message, AIMessage)
            )
            if not tool_messages and not already_called:
                question = _parse_question(text)
                call = self._choose_tool(question, tools)
                if call is not None:
                    return self._result(
                        AIMessage(content="", tool_calls=[call])
                    )
            if tool_messages:
                summary = "\n".join(
                    str(message.content) for message in tool_messages[-2:]
                )
                return self._result(
                    AIMessage(content=f"Tool result:\n{summary[:900]}")
                )

        # --- JSON intents ------------------------------------------------
        if intent == "route":
            question = _parse_question(text)
            route = self._route(question)
            return self._result(
                AIMessage(
                    content=json.dumps(
                        {"route": route, "reason": f"offline heuristic -> {route}"}
                    )
                )
            )

        if intent == "grade":
            question = _parse_question(text)
            evidence = _parse_evidence(text)
            payload = self._grade(question, evidence)
            return self._result(AIMessage(content=json.dumps(payload)))

        if intent == "rewrite":
            question = _parse_question(text)
            keywords = tokenize(question)
            return self._result(
                AIMessage(content=json.dumps({"query": " ".join(dict.fromkeys(keywords))}))
            )

        if intent == "judge":
            question = _parse_question(text)
            evidence = _parse_evidence(text)
            context = " ".join(item["body"] for item in evidence)
            payload = self._judge(question, context, text)
            return self._result(AIMessage(content=json.dumps(payload)))

        if intent == "answer":
            question = _parse_question(text)
            evidence = _parse_evidence(text)
            return self._result(AIMessage(content=self._answer(question, evidence)))

        # --- unknown intent ----------------------------------------------
        return self._result(
            AIMessage(
                content=(
                    "Offline model: I can answer questions about the indexed enterprise "
                    "documents. Ask about HR, IT security, travel expenses, support SLAs or "
                    "data retention."
                )
            )
        )

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _result(message: AIMessage) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _intent(text: str) -> str:
        match = re.search(r"\[\[INTENT:(\w+)\]\]", text)
        return match.group(1).lower() if match else "answer"

    @staticmethod
    def _route(question: str) -> str:
        """Keyword routing heuristic (stands in for the LLM router)."""
        lowered = question.lower()
        if _GREETING_RE.match(lowered):
            return "direct"
        if looks_like_tool_request(question):
            return "tool"
        return "retrieve"

    @staticmethod
    def _grade(question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        if not evidence:
            return {
                "sufficient": False,
                "confidence": 0.0,
                "reason": "no evidence retrieved",
                "missing": ["relevant documents"],
            }

        question_tokens = content_tokens(question)
        question_numbers = numbers_in(question)

        best_overlap = 0.0
        for item in evidence:
            item_tokens = content_tokens(item["body"])
            if not question_tokens:
                continue
            overlap = len(question_tokens & item_tokens) / len(question_tokens)
            if question_numbers & numbers_in(item["body"]):
                overlap += 0.15
            best_overlap = max(best_overlap, overlap)

        sufficient = best_overlap >= 0.25
        return {
            "sufficient": sufficient,
            "confidence": round(min(1.0, best_overlap), 3),
            "reason": (
                "evidence covers the question"
                if sufficient
                else "evidence shares too little with the question"
            ),
            "missing": [] if sufficient else ["specific policy detail"],
        }

    @staticmethod
    def _answer(question: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return _ABSTAIN

        question_tokens = content_tokens(question)
        all_evidence_tokens: set[str] = set()
        for item in evidence:
            all_evidence_tokens |= content_tokens(item["body"])

        # Coverage gate: if almost none of the question's content words appear
        # anywhere in the retrieved evidence, the documents are about a different
        # topic and no sentence can honestly answer it. This is what stops the
        # extractive double from quoting an unrelated sentence at an out-of-scope
        # question (e.g. equity vesting, which no policy document covers).
        global_coverage = (
            len(question_tokens & all_evidence_tokens) / len(question_tokens)
            if question_tokens
            else 0.0
        )
        if global_coverage < _GLOBAL_COVERAGE_THRESHOLD:
            return _ABSTAIN

        # Score every block: the best sentence gives most of the signal, and a
        # bonus for how much of the question the whole block covers picks the
        # *right* document when two blocks both contain a matching sentence
        # (a compound "receipts ... submit by when" question matches both the
        # travel policy and the retention policy lexically; only one answers it).
        best: tuple[float, float, int, str, str, str] | None = None
        best_block_coverage = 0.0
        for item in evidence:
            block_tokens = content_tokens(item["body"])
            block_coverage = (
                len(question_tokens & block_tokens) / len(question_tokens)
                if question_tokens
                else 0.0
            )
            sentence, overlap = _best_sentence(question, item["body"])
            block_score = overlap + 0.3 * block_coverage
            if best is None or block_score > best[0]:
                best = (
                    block_score,
                    overlap,
                    item["marker"],
                    item["source"],
                    sentence,
                    item["body"],
                )
                best_block_coverage = block_coverage

        assert best is not None  # evidence is non-empty
        _block_score, overlap, marker, source, sentence, body = best

        # A single block must cover a real share of the question, and the best
        # sentence must actually be about it. Otherwise: abstain.
        if best_block_coverage < _BLOCK_COVERAGE_THRESHOLD or overlap < _ABSTAIN_THRESHOLD:
            return _ABSTAIN

        # Real answers often need the sentence *after* the headline fact (carry-over
        # deadlines, approval thresholds).  Append the runner-up from the same block
        # when it is clearly relevant, which mirrors how an LLM quotes two clauses.
        extras = [
            candidate
            for candidate in _split_sentences(body)
            if candidate != sentence
            and question_tokens
            and len(content_tokens(candidate) & question_tokens) / len(question_tokens)
            >= max(0.2, overlap * 0.5)
        ]

        if extras:
            return f"According to [{marker}] ({source}): {sentence} {extras[0]}"
        return f"According to [{marker}] ({source}): {sentence}"

    @staticmethod
    def _judge(question: str, context: str, text: str) -> dict[str, Any]:
        """Coarse lexical judge used only when running evaluation offline.

        Faithfulness = share of the answer's content words that the evidence
        contains; relevance = share of the *question's* content words the answer
        covers. Both are weak proxies for a real judge, which is exactly why the
        project ships a Groq-backed judge for live runs.
        """
        answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n\n|$)", text, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""
        answer_tokens = content_tokens(answer)
        context_tokens = content_tokens(context)
        question_tokens = content_tokens(question)

        if not answer_tokens:
            faithfulness = relevance = 1
        else:
            supported = (
                len(answer_tokens & context_tokens) / len(answer_tokens)
                if context_tokens
                else 0.0
            )
            covered_question = (
                len(answer_tokens & question_tokens) / len(question_tokens)
                if question_tokens
                else 0.0
            )
            faithfulness = max(1, min(5, 1 + round(4 * supported)))
            relevance = max(1, min(5, 1 + round(4 * covered_question)))

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "citation_support": faithfulness,
            "rationale": "offline lexical judge (no LLM call)",
            "unsupported_claims": [],
        }

    @staticmethod
    def _choose_tool(question: str, tools: list[Any]) -> dict[str, Any] | None:
        """Pick a tool and build arguments from the question."""
        names = {
            (tool.get("function", {}).get("name") if isinstance(tool, dict) else None)
            for tool in tools
        }
        names.discard(None)
        lowered = question.lower()

        ticket_match = re.search(r"TCK-\d+", question, re.IGNORECASE)
        expense_amount = re.search(
            r"(?:\$|usd\s*)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*(?:usd|dollars)", lowered
        )
        amount = 0.0
        if expense_amount:
            raw = expense_amount.group(1) or expense_amount.group(2) or "0"
            amount = float(raw.replace(",", ""))

        chosen: tuple[str, dict[str, Any]] | None = None

        if "get_ticket_status" in names and ticket_match and "status" in lowered:
            chosen = ("get_ticket_status", {"ticket_id": ticket_match.group(0).upper()})
        elif "submit_expense_report" in names and re.search(
            r"expense|reimburs|claim", lowered
        ):
            category = "other"
            for keyword, value in (
                ("hotel", "hotel"),
                ("stay", "hotel"),
                ("flight", "airfare"),
                ("airfare", "airfare"),
                ("meal", "meals"),
                ("dinner", "meals"),
                ("lunch", "meals"),
                ("taxi", "ground_transport"),
                ("uber", "ground_transport"),
                ("cab", "ground_transport"),
                ("internet", "internet"),
            ):
                if keyword in lowered:
                    category = value
                    break
            chosen = (
                "submit_expense_report",
                {
                    "employee_email": "employee@company.example",
                    "amount_usd": amount or 50.0,
                    "category": category,
                    "description": question[:300],
                    "receipts_attached": "no receipt" not in lowered,
                },
            )
        elif "create_support_ticket" in names and re.search(
            r"ticket|incident|outage|bug|issue|defect", lowered
        ):
            priority = "P3"
            explicit = re.search(r"\bp([1-4])\b", lowered)
            if explicit:
                priority = f"P{explicit.group(1)}"
            elif re.search(r"critical|outage|down", lowered):
                priority = "P1"
            elif "major" in lowered:
                priority = "P2"
            elif "cosmetic" in lowered:
                priority = "P4"
            chosen = (
                "create_support_ticket",
                {
                    "subject": question[:150],
                    "description": question[:500],
                    "priority": priority,
                    "requester_email": "employee@company.example",
                },
            )
        elif "send_notification" in names and re.search(
            r"notify|notification|send|announce|status page|slack|email the", lowered
        ):
            channel = "slack" if "slack" in lowered else "email"
            if "status page" in lowered:
                channel = "status_page"
            recipient = "public" if channel == "status_page" else "#engineering"
            if channel == "email":
                recipient = "team@company.example"
            chosen = (
                "send_notification",
                {"channel": channel, "recipient": recipient, "message": question[:500]},
            )
        elif "lookup_employee" in names and re.search(
            r"who\s+is|who's|directory|manager|contact|look ?up", lowered
        ):
            chosen = ("lookup_employee", {"query": question[:80]})
        elif "search_knowledge_base" in names:
            chosen = ("search_knowledge_base", {"query": question, "top_k": 3})

        if chosen is None or chosen[0] not in names:
            return None

        name, args = chosen
        return {
            "name": name,
            "args": args,
            "id": f"call_{abs(hash(name + question)) % 10000}",
            "type": "tool_call",
        }
