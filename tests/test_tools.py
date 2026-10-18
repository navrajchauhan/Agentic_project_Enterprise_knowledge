"""Tool registry: schemas, validation, policy rules, idempotency and audit."""

from __future__ import annotations

import pytest


def test_catalogue_exposes_all_tools(registry) -> None:
    expected = {
        "search_knowledge_base",
        "lookup_employee",
        "get_ticket_status",
        "create_support_ticket",
        "submit_expense_report",
        "send_notification",
    }
    assert set(registry.names) == expected
    assert len(registry) == 6


def test_openai_tool_schemas_are_well_formed(registry) -> None:
    schemas = registry.openai_tools()
    assert len(schemas) == 6
    for schema in schemas:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] and function["description"]
        assert function["parameters"]["type"] == "object"
        assert "properties" in function["parameters"]


def test_unknown_tool_returns_failure_not_exception(registry) -> None:
    result = registry.execute("delete_everything", {})
    assert result.ok is False
    assert result.error == "unknown_tool"
    assert "Available tools" in result.content


def test_invalid_arguments_are_rejected_before_execution(registry) -> None:
    result = registry.execute(
        "create_support_ticket",
        {"subject": "hi", "description": "short", "priority": "P9", "requester_email": "a@b.com"},
    )
    assert result.ok is False
    assert result.error == "invalid_arguments"
    # The message tells the model how to fix it.
    assert "priority" in result.content


def test_arguments_may_arrive_as_a_json_string(registry) -> None:
    result = registry.execute(
        "lookup_employee", '{"query": "Platform Engineering"}'
    )
    assert result.ok is True
    assert "priya.raghavan@company.example" in result.content


def test_create_ticket_is_idempotent(registry) -> None:
    first = registry.execute(
        "create_support_ticket",
        {
            "subject": "Checkout API returns 500 for EU tenants",
            "description": "EU checkout calls fail with a 500 since 09:10 UTC.",
            "priority": "P2",
            "requester_email": "daniel.okafor@company.example",
        },
    )
    assert first.ok is True
    ticket_id = first.data["ticket"]["ticket_id"]
    assert ticket_id.startswith("TCK-")

    second = registry.execute(
        "create_support_ticket",
        {
            "subject": "Checkout API returns 500 for EU tenants",
            "description": "Same issue reported again.",
            "priority": "P2",
            "requester_email": "daniel.okafor@company.example",
        },
    )
    assert second.ok is True
    assert second.data["deduplicated"] is True
    assert second.data["ticket"]["ticket_id"] == ticket_id


def test_ticket_status_lookup(registry) -> None:
    created = registry.execute(
        "create_support_ticket",
        {
            "subject": "Status lookup test",
            "description": "Ticket used to verify the status tool.",
            "priority": "P3",
            "requester_email": "asha.nair@company.example",
        },
    )
    ticket_id = created.data["ticket"]["ticket_id"]

    result = registry.execute("get_ticket_status", {"ticket_id": ticket_id})
    assert result.ok is True
    assert "open" in result.content

    missing = registry.execute("get_ticket_status", {"ticket_id": "TCK-9999"})
    assert missing.ok is False
    assert missing.error == "not_found"


def test_expense_above_five_thousand_needs_vp_approval(registry) -> None:
    result = registry.execute(
        "submit_expense_report",
        {
            "employee_email": "arjun.mehta@company.example",
            "amount_usd": 6200,
            "category": "airfare",
            "description": "Business class flight to the Singapore customer summit.",
            "receipts_attached": True,
        },
    )
    assert result.ok is True
    assert result.data["approval_level"] == "VP"
    assert result.data["decision"] == "pending_vp_approval"
    assert result.data["approval_ticket"].startswith("TCK-")


def test_expense_below_threshold_is_auto_approved(registry) -> None:
    result = registry.execute(
        "submit_expense_report",
        {
            "employee_email": "arjun.mehta@company.example",
            "amount_usd": 120,
            "category": "ground_transport",
            "description": "Taxi from the airport to the customer office.",
            "receipts_attached": True,
        },
    )
    assert result.ok is True
    assert result.data["decision"] == "auto_approved"
    assert result.data["approval_level"] == "auto"


def test_expense_without_receipts_is_rejected(registry) -> None:
    result = registry.execute(
        "submit_expense_report",
        {
            "employee_email": "arjun.mehta@company.example",
            "amount_usd": 250,
            "category": "hotel",
            "description": "Hotel night during the customer visit.",
            "receipts_attached": False,
        },
    )
    assert result.ok is False
    assert result.error == "policy_receipts_required"
    assert "25" in result.content


def test_alcohol_is_not_reimbursable(registry) -> None:
    result = registry.execute(
        "submit_expense_report",
        {
            "employee_email": "arjun.mehta@company.example",
            "amount_usd": 90,
            "category": "meals",
            "description": "Dinner with the customer including two bottles of wine.",
            "receipts_attached": True,
        },
    )
    assert result.ok is False
    assert result.error == "policy_non_reimbursable"


def test_notification_to_external_address_is_refused(registry) -> None:
    result = registry.execute(
        "send_notification",
        {
            "channel": "email",
            "recipient": "someone@gmail.com",
            "message": "Here is the internal travel policy.",
        },
    )
    assert result.ok is False
    assert result.error == "external_recipient_blocked"


def test_notification_to_internal_address_succeeds(registry) -> None:
    result = registry.execute(
        "send_notification",
        {
            "channel": "slack",
            "recipient": "#engineering",
            "message": "P1 incident resolved; post-incident review scheduled.",
        },
    )
    assert result.ok is True
    assert result.data["notification"]["notification_id"].startswith("NTF-")


def test_knowledge_base_tool_returns_citable_passages(registry) -> None:
    result = registry.execute(
        "search_knowledge_base", {"query": "annual leave carry over", "top_k": 2}
    )
    assert result.ok is True
    passages = result.data["passages"]
    assert passages
    assert passages[0]["citation"]
    assert passages[0]["chunk_id"]


def test_every_call_is_audited(registry, store) -> None:
    registry.execute("lookup_employee", {"query": "Security"})
    rows = store.recent_audit(limit=10)
    assert rows
    latest = rows[0]
    assert latest["tool"] == "lookup_employee"
    assert latest["ok"] == 1


def test_tool_exceptions_are_converted_to_failures(settings, store) -> None:
    """A buggy tool must not break the graph."""
    from pydantic import BaseModel

    from enterprise_agent.tools.models import ToolResult, ToolSpec
    from enterprise_agent.tools.registry import ToolRegistry

    class Args(BaseModel):
        value: int = 1

    def explode(value: int) -> ToolResult:  # noqa: ARG001
        raise RuntimeError("boom")

    registry = ToolRegistry(store=store)
    registry.register(ToolSpec(name="explode", description="fails", args_model=Args, func=explode))

    result = registry.execute("explode", {"value": 2})
    assert result.ok is False
    assert result.error == "internal_error"
    assert "boom" in result.content
