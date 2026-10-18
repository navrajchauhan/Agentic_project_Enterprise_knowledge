"""MCP server: exposes the enterprise tools over the Model Context Protocol.

Run standalone (stdio transport)::

    python -m enterprise_agent.mcp.server

The agent can either talk to this server (``TOOL_PROVIDER=mcp``) or call the same
:class:`~enterprise_agent.tools.registry.ToolRegistry` in-process
(``TOOL_PROVIDER=direct``).  Both paths go through the same ``ToolSpec`` objects,
so behaviour cannot diverge.

Why MCP at all?
    It is the standard seam between an agent and enterprise systems.  Once the
    tools live behind MCP, the same server can be consumed by this agent, by a
    desktop client, or by a different orchestrator - and the tool implementation
    can move to another host without touching the agent.

Protocol hygiene
    MCP speaks JSON-RPC on **stdout**, so nothing in this process may print to
    stdout.  Logging goes to stderr (see ``logging_setup``), and ``.env`` loading
    happens before the server starts.
"""

from __future__ import annotations

import json
from typing import Any

# The MCP Python SDK renamed `FastMCP` to `MCPServer` in v2 (module
# `mcp.server.mcpserver`).  Both expose the same decorator API, so support either
# version rather than pinning the whole project to one SDK major.
try:  # pragma: no cover - depends on the installed SDK major
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp < 2
    from mcp.server.fastmcp import FastMCP

from ..config import get_settings
from ..logging_setup import configure_logging, get_logger
from ..rag.indexer import Indexer
from ..rag.retriever import HybridRetriever
from ..tools.registry import ToolRegistry, build_registry
from ..tools.store import ActionStore

logger = get_logger(__name__)

#: Server identity shown to clients during the MCP handshake.
SERVER_NAME = "enterprise-knowledge-agent"

mcp = FastMCP(SERVER_NAME)

# Lazily-initialised dependencies.  Building the index on import would make the
# handshake slow and would crash the server before the client can see the error.
_state: dict[str, Any] = {}


def _registry() -> ToolRegistry:
    """Build (once) the tool registry used by every MCP tool handler."""
    if "registry" not in _state:
        settings = get_settings()
        configure_logging(settings.log_level)
        settings.ensure_directories()
        logger.info("MCP server bootstrapping index and tools")

        index = Indexer(settings).build()  # loads the persisted index when fresh
        retriever = HybridRetriever(index, settings)
        store = ActionStore(settings.action_db_path)

        _state["settings"] = settings
        _state["index"] = index
        _state["retriever"] = retriever
        _state["registry"] = build_registry(store=store, retriever=retriever, settings=settings)
    return _state["registry"]


def _execute(tool: str, arguments: dict[str, Any]) -> str:
    """Run a tool and return its JSON envelope (never raises)."""
    result = _registry().execute(tool, arguments, thread_id="mcp")
    return result.to_json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# The docstrings and signatures here mirror the pydantic argument models in
# enterprise_agent.tools.models. ToolSpec remains the source of truth for
# validation; these wrappers only adapt the transport.


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search the enterprise policy knowledge base.

    Args:
        query: Natural-language search query, e.g. "hotel cap domestic travel".
        top_k: Number of passages to return (1-10).
    """
    return _execute("search_knowledge_base", {"query": query, "top_k": top_k})


@mcp.tool()
def lookup_employee(query: str) -> str:
    """Look up a colleague by name, email, department or job title.

    Args:
        query: Name, email, department or title to search for.
    """
    return _execute("lookup_employee", {"query": query})


@mcp.tool()
def get_ticket_status(ticket_id: str) -> str:
    """Fetch the status of a support ticket.

    Args:
        ticket_id: Ticket id such as TCK-1004.
    """
    return _execute("get_ticket_status", {"ticket_id": ticket_id})


@mcp.tool()
def create_support_ticket(
    subject: str,
    description: str,
    priority: str = "P3",
    requester_email: str = "employee@company.example",
) -> str:
    """Create a support ticket (P1 critical, P2 major, P3 minor, P4 cosmetic).

    Args:
        subject: Short subject line.
        description: What happened, with enough detail to act on.
        priority: One of P1, P2, P3, P4.
        requester_email: Email of the person raising the ticket.
    """
    return _execute(
        "create_support_ticket",
        {
            "subject": subject,
            "description": description,
            "priority": priority,
            "requester_email": requester_email,
        },
    )


@mcp.tool()
def submit_expense_report(
    employee_email: str,
    amount_usd: float,
    category: str,
    description: str,
    receipts_attached: bool = True,
) -> str:
    """Submit an expense claim, applying the travel policy rules.

    Args:
        employee_email: Email of the employee claiming the expense.
        amount_usd: Amount in USD.
        category: airfare, hotel, meals, ground_transport, internet or other.
        description: Business purpose of the expense.
        receipts_attached: Whether itemised receipts are attached (required above USD 25).
    """
    return _execute(
        "submit_expense_report",
        {
            "employee_email": employee_email,
            "amount_usd": amount_usd,
            "category": category,
            "description": description,
            "receipts_attached": receipts_attached,
        },
    )


@mcp.tool()
def send_notification(channel: str, recipient: str, message: str) -> str:
    """Send an internal notification (email, slack or status_page).

    Args:
        channel: email, slack or status_page.
        recipient: Internal email address, Slack channel, or 'public' for the status page.
        message: Message body.
    """
    return _execute(
        "send_notification", {"channel": channel, "recipient": recipient, "message": message}
    )


# ---------------------------------------------------------------------------
# Resources (read-only context a client can attach)
# ---------------------------------------------------------------------------


@mcp.resource("kb://documents")
def list_documents() -> str:
    """List the indexed documents with their chunk counts."""
    _registry()  # ensures the index is loaded
    index = _state["index"]
    documents: dict[str, dict[str, Any]] = {}
    for chunk in index.chunks:
        entry = documents.setdefault(
            chunk.doc_id, {"doc_id": chunk.doc_id, "title": chunk.title, "chunks": 0}
        )
        entry["chunks"] += 1

    return json.dumps(
        {
            "document_count": len(documents),
            "chunk_count": len(index),
            "embedder": index.embedder_name,
            "documents": sorted(documents.values(), key=lambda item: item["doc_id"]),
        },
        indent=2,
    )


def main() -> None:
    """Entry point: start the MCP server on stdio."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting MCP server %s (stdio)", SERVER_NAME)
    try:
        mcp.run(transport="stdio")
    except TypeError:  # older/newer signature without the transport argument
        mcp.run()


if __name__ == "__main__":
    main()
