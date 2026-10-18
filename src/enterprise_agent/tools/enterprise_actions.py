"""The six enterprise actions the agent can perform.

Each action is a plain, typed function returning a ``ToolResult``.  Wiring (LLM
binding, MCP publication, audit logging) is the registry's job - see
``registry.py``.

The actions are chosen to exercise every pattern an enterprise agent needs:

============================  ====================================================
``search_knowledge_base``     read-only retrieval, i.e. RAG *as a tool*
``lookup_employee``           read-only structured lookup
``get_ticket_status``         read-only follow-up on a previous action
``create_support_ticket``     write, with idempotency (no duplicate tickets)
``submit_expense_report``     write, with **policy validation** - the business
                              rules come from the documents the agent retrieves
``send_notification``         write, with a channel allow-list and PII redaction
============================  ====================================================

The expense tool is the interesting one: it applies the same thresholds the
travel policy document states (director approval above USD 1,000, VP approval
above USD 5,000, receipts required above USD 25, no alcohol).  Retrieval and
action therefore agree, which is the whole point of grounding an agent in
documents instead of letting it invent approval rules.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..logging_setup import get_logger
from ..utils.text import truncate
from .models import (
    CreateTicketArgs,
    LookupEmployeeArgs,
    SearchKnowledgeBaseArgs,
    SendNotificationArgs,
    SubmitExpenseArgs,
    TicketStatusArgs,
    ToolResult,
    ToolSpec,
)
from .store import ActionStore

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

#: A miniature employee directory.  In production this would be an HRIS lookup;
#: the shape is what matters here.  Note there is no salary field: an agent that
#: can read salaries invites data-exfiltration attempts, which is exactly what
#: the input guard blocks.
EMPLOYEE_DIRECTORY: list[dict[str, str]] = [
    {
        "name": "Priya Raghavan",
        "title": "Engineering Manager",
        "department": "Platform Engineering",
        "email": "priya.raghavan@company.example",
        "location": "Bengaluru",
    },
    {
        "name": "Arjun Mehta",
        "title": "Senior Software Engineer",
        "department": "Platform Engineering",
        "email": "arjun.mehta@company.example",
        "location": "Bengaluru",
    },
    {
        "name": "Sofia Almeida",
        "title": "Security Engineer",
        "department": "Security",
        "email": "sofia.almeida@company.example",
        "location": "Lisbon",
    },
    {
        "name": "Daniel Okafor",
        "title": "Customer Support Lead",
        "department": "Customer Support",
        "email": "daniel.okafor@company.example",
        "location": "Lagos",
    },
    {
        "name": "Mei Ling Chen",
        "title": "Director of Engineering",
        "department": "Platform Engineering",
        "email": "meiling.chen@company.example",
        "location": "Singapore",
    },
    {
        "name": "Tomas Novak",
        "title": "People Operations Partner",
        "department": "People",
        "email": "tomas.novak@company.example",
        "location": "Prague",
    },
    {
        "name": "Asha Nair",
        "title": "Finance Analyst",
        "department": "Finance",
        "email": "asha.nair@company.example",
        "location": "Kochi",
    },
    {
        "name": "Liam O'Connor",
        "title": "VP of Engineering",
        "department": "Engineering",
        "email": "liam.oconnor@company.example",
        "location": "Dublin",
    },
]

#: Approval ladder from travel_expense_policy.md.  Ordered high -> low so the
#: first matching threshold wins.
APPROVAL_THRESHOLDS: list[tuple[float, str]] = [
    (5000.0, "VP"),
    (1000.0, "Director"),
    (0.0, "auto"),
]

#: Receipt requirement from travel_expense_policy.md.
RECEIPT_REQUIRED_ABOVE_USD = 25.0

#: Categories that are never reimbursable (travel_expense_policy.md).
NON_REIMBURSABLE_KEYWORDS = ("alcohol", "wine", "beer", "spirits", "minibar")

INTERNAL_DOMAIN = "company.example"


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------


def search_knowledge_base(
    *,
    query: str,
    top_k: int = 3,
    retriever: Any,
) -> ToolResult:
    """Retrieve passages from the indexed knowledge base.

    Exposing retrieval as a *tool* (not only as a graph node) is what lets the
    LLM ask a follow-up question mid-task, e.g. "check the SLA before you set
    the priority".  The passages come back with citation labels so the final
    answer can still be source-grounded.
    """
    hits = retriever.search(query, top_k=top_k)
    if not hits:
        return ToolResult.failure(
            "search_knowledge_base",
            "no_results",
            content=f"No documents matched the query: {query!r}",
        )

    passages = [
        {
            "citation": hit.chunk.citation_label,
            "chunk_id": hit.chunk.chunk_id,
            "score": round(hit.score, 5),
            "text": truncate(" ".join(hit.context_text.split()), 700),
        }
        for hit in hits
    ]
    content = "\n\n".join(
        f"[{index}] {p['citation']}\n{p['text']}" for index, p in enumerate(passages, start=1)
    )
    return ToolResult.success(
        "search_knowledge_base",
        f"Found {len(passages)} passage(s):\n\n{content}",
        passages=passages,
        query=query,
    )


def lookup_employee(*, query: str) -> ToolResult:
    """Search the employee directory by name, email, department or title."""
    needle = query.lower().strip()
    matches = [
        person
        for person in EMPLOYEE_DIRECTORY
        if needle in person["name"].lower()
        or needle in person["email"].lower()
        or needle in person["department"].lower()
        or needle in person["title"].lower()
        or any(token in person["name"].lower() for token in needle.split())
    ]

    if not matches:
        return ToolResult.failure(
            "lookup_employee",
            "not_found",
            content=f"No directory entry matched {query!r}. Try a name, team or department.",
        )

    lines = [
        f"- {p['name']} | {p['title']} | {p['department']} | {p['email']} | {p['location']}"
        for p in matches[:10]
    ]
    return ToolResult.success(
        "lookup_employee",
        f"{len(matches)} match(es):\n" + "\n".join(lines),
        employees=matches[:10],
    )


def get_ticket_status(*, ticket_id: str, store: ActionStore) -> ToolResult:
    """Look up an existing ticket."""
    ticket = store.get_ticket(ticket_id)
    if ticket is None:
        return ToolResult.failure(
            "get_ticket_status",
            "not_found",
            content=f"No ticket found with id {ticket_id}. Ticket ids look like TCK-1001.",
        )
    return ToolResult.success(
        "get_ticket_status",
        f"{ticket['ticket_id']} is {ticket['status']} (priority {ticket['priority']}, "
        f"raised by {ticket['requester_email']}): {ticket['subject']}",
        ticket=ticket,
    )


# ---------------------------------------------------------------------------
# Write tools
# ---------------------------------------------------------------------------


def create_support_ticket(
    *,
    subject: str,
    description: str,
    priority: str,
    requester_email: str,
    store: ActionStore,
) -> ToolResult:
    """Create (or reuse) a support ticket."""
    ticket = store.create_ticket(
        subject=subject.strip(),
        description=description.strip(),
        priority=priority,
        requester_email=requester_email.strip(),
    )

    if ticket["deduplicated"]:
        return ToolResult.success(
            "create_support_ticket",
            f"An open ticket already covers this: {ticket['ticket_id']} "
            f"(priority {ticket['priority']}, status {ticket['status']}).",
            ticket=ticket,
            deduplicated=True,
        )

    # Priority P1/P2 carry an SLA that the support team must see immediately.
    sla_note = ""
    if priority in {"P1", "P2"}:
        sla_note = " 24x7 on-call paged per product_sla.md (P1: 15 min response, P2: 1 hour)."

    return ToolResult.success(
        "create_support_ticket",
        f"Created {ticket['ticket_id']} with priority {ticket['priority']} for "
        f"{ticket['requester_email']}.{sla_note}",
        ticket=ticket,
        deduplicated=False,
    )


def submit_expense_report(
    *,
    employee_email: str,
    amount_usd: float,
    category: str,
    description: str,
    receipts_attached: bool,
    store: ActionStore,
) -> ToolResult:
    """Submit an expense, applying the travel policy rules.

    Returns ``ok=False`` for policy rejections (``error`` names the rule) so the
    agent relays the reason instead of pretending the claim succeeded.
    """
    lowered = description.lower()

    # Rule 1: alcohol is never reimbursable.
    if any(keyword in lowered for keyword in NON_REIMBURSABLE_KEYWORDS):
        return ToolResult.failure(
            "submit_expense_report",
            "policy_non_reimbursable",
            content=(
                "Rejected: alcohol is not reimbursable under travel_expense_policy.md "
                "(section 'Non-Reimbursable Items'). Submit the remaining amount instead."
            ),
            amount_usd=amount_usd,
            category=category,
        )

    # Rule 2: itemised receipts above USD 25.
    if amount_usd > RECEIPT_REQUIRED_ABOVE_USD and not receipts_attached:
        return ToolResult.failure(
            "submit_expense_report",
            "policy_receipts_required",
            content=(
                f"Rejected: itemised receipts are required for any single expense above "
                f"USD {RECEIPT_REQUIRED_ABOVE_USD:.0f} (travel_expense_policy.md). "
                "Attach the receipts and resubmit."
            ),
            amount_usd=amount_usd,
            category=category,
        )

    approval_level = next(
        level for threshold, level in APPROVAL_THRESHOLDS if amount_usd > threshold
    )
    decision = "auto_approved" if approval_level == "auto" else f"pending_{approval_level.lower()}_approval"

    expense = store.record_expense(
        employee_email=employee_email,
        amount_usd=amount_usd,
        category=category,
        description=description,
        receipts_attached=receipts_attached,
        decision=decision,
        approval_level=approval_level,
    )

    # Expenses needing approval also need a paper trail the approver can see.
    ticket_id: str | None = None
    if approval_level != "auto":
        ticket = store.create_ticket(
            subject=f"Expense approval ({approval_level}) - USD {amount_usd:,.2f} {category}",
            description=(
                f"Expense {expense['expense_id']} submitted by {employee_email} "
                f"for USD {amount_usd:,.2f} ({category}): {description}"
            ),
            priority="P4",
            requester_email=employee_email,
            note=f"Approver level: {approval_level}",
        )
        ticket_id = ticket["ticket_id"]

    content = (
        f"Recorded {expense['expense_id']}: USD {amount_usd:,.2f} ({category}) -> {decision}."
    )
    if ticket_id:
        content += (
            f" Approval ticket {ticket_id} created for the {approval_level} "
            "(travel_expense_policy.md approval ladder)."
        )

    return ToolResult.success(
        "submit_expense_report",
        content,
        expense=expense,
        decision=decision,
        approval_level=approval_level,
        approval_ticket=ticket_id,
    )


def send_notification(
    *,
    channel: str,
    recipient: str,
    message: str,
    store: ActionStore,
    settings: Settings,
) -> ToolResult:
    """Send a notification (simulated) to an internal channel.

    Two guardrails are enforced here rather than in the prompt, because prompts
    are suggestions and code is a guarantee:

    1. **Channel allow-list** - only email / slack / status_page exist.
    2. **Recipient domain check** - external email addresses are refused, which
       stops an agent from mailing internal policy text to the outside world.
       PII is redacted from the body before it is persisted.
    """
    from ..guardrails.input_guard import redact_pii  # local import: avoids a cycle

    # Slack channels (``#engineering``) are internal by construction; email
    # recipients must be on the company domain.
    is_internal_channel = channel == "slack" and recipient.strip().startswith("#")
    if channel == "email" and INTERNAL_DOMAIN not in recipient.lower():
        return ToolResult.failure(
            "send_notification",
            "external_recipient_blocked",
            content=(
                f"Refused: email notifications may only go to internal "
                f"@{INTERNAL_DOMAIN} recipients. Got {recipient!r}."
            ),
        )
    if channel == "slack" and not is_internal_channel:
        return ToolResult.failure(
            "send_notification",
            "external_recipient_blocked",
            content=(
                "Refused: slack notifications must target an internal channel such as "
                f"'#engineering'. Got {recipient!r}."
            ),
        )
    if channel == "status_page" and recipient.lower() != "public":
        return ToolResult.failure(
            "send_notification",
            "invalid_recipient",
            content="Refused: status_page notifications take recipient='public'.",
        )

    safe_message, redactions = redact_pii(message) if settings.redact_pii else (message, {})
    record = store.record_notification(
        channel=channel, recipient=recipient, message=safe_message
    )

    note = f" Redacted: {', '.join(sorted(redactions))}." if redactions else ""
    return ToolResult.success(
        "send_notification",
        f"Queued {record['notification_id']} via {channel} to {recipient}.{note}",
        notification=record,
        redactions=redactions,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_tool_specs(
    *,
    store: ActionStore,
    retriever: Any,
    settings: Settings,
) -> list[ToolSpec]:
    """Create the tool catalogue with its dependencies bound in.

    ``functools.partial`` keeps the handlers plain functions (easy to unit test
    in isolation) while still injecting the store/retriever/settings.
    """
    from functools import partial

    return [
        ToolSpec(
            name="search_knowledge_base",
            description=(
                "Search the enterprise knowledge base (HR, IT security, travel, support SLA, "
                "data retention policies). Use this when you need a policy fact, a limit or a "
                "definition. Returns numbered passages you can cite."
            ),
            args_model=SearchKnowledgeBaseArgs,
            func=partial(search_knowledge_base, retriever=retriever),
        ),
        ToolSpec(
            name="lookup_employee",
            description=(
                "Look up a colleague in the employee directory by name, email, department or "
                "job title. Returns name, title, department, email and location."
            ),
            args_model=LookupEmployeeArgs,
            func=lookup_employee,
        ),
        ToolSpec(
            name="get_ticket_status",
            description="Fetch the current status of a support ticket by id (e.g. TCK-1004).",
            args_model=TicketStatusArgs,
            func=partial(get_ticket_status, store=store),
        ),
        ToolSpec(
            name="create_support_ticket",
            description=(
                "Create a support ticket and return its id. Use P1 for a critical outage, "
                "P2 for major impact, P3 for minor issues and P4 for cosmetic ones. "
                "Duplicate open tickets for the same subject are reused."
            ),
            args_model=CreateTicketArgs,
            func=partial(create_support_ticket, store=store),
            writes=True,
        ),
        ToolSpec(
            name="submit_expense_report",
            description=(
                "Submit an expense claim. The tool applies the travel policy: receipts are "
                "required above USD 25, alcohol is never reimbursable, above USD 1,000 needs "
                "director approval and above USD 5,000 needs VP approval."
            ),
            args_model=SubmitExpenseArgs,
            func=partial(submit_expense_report, store=store),
            writes=True,
        ),
        ToolSpec(
            name="send_notification",
            description=(
                "Send a notification to an internal email address, a Slack channel, or the "
                "public status page. External recipients are refused."
            ),
            args_model=SendNotificationArgs,
            func=partial(send_notification, store=store, settings=settings),
            writes=True,
        ),
    ]
