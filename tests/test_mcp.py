"""MCP integration test: the agent talking to the tool server over stdio.

Skipped by default because it spawns a subprocess (slow, and blocked in some
sandboxes).  Run it explicitly:

    RUN_MCP_TESTS=1 pytest tests/test_mcp.py -q

What it proves that the unit tests cannot: the server publishes the same six
tools, the client completes the handshake, and a real tool call round-trips
through JSON-RPC and comes back with the same ``ToolResult`` envelope the
in-process registry produces.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.mcp,
    pytest.mark.skipif(
        os.environ.get("RUN_MCP_TESTS") != "1",
        reason="spawns the MCP server subprocess; set RUN_MCP_TESTS=1 to enable",
    ),
]


@pytest.fixture(scope="module")
def client(settings):
    from enterprise_agent.mcp.client import MCPToolClient

    mcp_settings = settings.model_copy(update={"tool_provider": "mcp"})
    mcp_client = MCPToolClient.from_settings(mcp_settings)
    mcp_client.start()
    yield mcp_client
    mcp_client.close()


def test_server_publishes_the_same_tool_catalogue(client, registry) -> None:
    assert sorted(client.names) == sorted(registry.names)


def test_schemas_are_exposed_for_llm_binding(client) -> None:
    schemas = client.openai_tools()
    assert len(schemas) == 6
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"


def test_read_tool_round_trip(client) -> None:
    result = client.execute("lookup_employee", {"query": "Security"})
    assert result.ok is True
    assert "sofia.almeida@company.example" in result.content


def test_write_tool_round_trip(client) -> None:
    result = client.execute(
        "create_support_ticket",
        {
            "subject": "MCP round-trip ticket",
            "description": "Created by the MCP integration test.",
            "priority": "P3",
            "requester_email": "asha.nair@company.example",
        },
    )
    assert result.ok is True
    assert result.data["ticket"]["ticket_id"].startswith("TCK-")


def test_validation_errors_survive_the_transport(client) -> None:
    result = client.execute("create_support_ticket", {"subject": "x"})
    assert result.ok is False
    assert result.error == "invalid_arguments"


def test_unknown_tool_is_rejected(client) -> None:
    result = client.execute("not_a_tool", {})
    assert result.ok is False
    assert result.error == "unknown_tool"


def test_agent_uses_mcp_tools_end_to_end(settings) -> None:
    from enterprise_agent.agent.runner import AgentRunner

    mcp_settings = settings.model_copy(update={"tool_provider": "mcp"})
    runner = AgentRunner(mcp_settings)
    try:
        assert runner.tool_transport == "MCPToolClient"
        answer = runner.ask(
            "Who is the VP of Engineering?", thread_id="mcp-e2e"
        )
        assert answer.route == "tool"
        assert any(call["tool"] == "lookup_employee" for call in answer.tool_calls)
        assert any(call["ok"] for call in answer.tool_calls)
    finally:
        runner.close()
