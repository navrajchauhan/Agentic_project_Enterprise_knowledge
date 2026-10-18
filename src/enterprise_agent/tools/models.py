"""Tool contracts.

One rule drives this module: **a tool is defined exactly once**.  The same
``ToolSpec`` (name + description + pydantic argument model + handler) is used to

* bind the tool to the LLM (JSON schema is generated from the argument model),
* execute it in-process, and
* publish it over MCP by the server in ``enterprise_agent.mcp.server``.

If each of those had its own definition they would drift, and drift in tool
schemas is the classic source of "the model passes the wrong argument" bugs.

Argument models are pydantic models rather than loose dicts because validation
happens *before* a side effect: a malformed ``amount`` never reaches the
expense table.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@dataclass
class ToolResult:
    """Uniform result envelope for every tool.

    ``content`` is what the LLM sees (short, factual, no stack traces);
    ``data`` is the structured payload the CLI/audit log can render.
    """

    ok: bool
    tool: str
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> str:
        """Serialise for the LangChain tool boundary / MCP response."""
        payload = {
            "ok": self.ok,
            "tool": self.tool,
            "content": self.content,
            "data": self.data,
        }
        if self.error:
            payload["error"] = self.error
        return json.dumps(payload, ensure_ascii=False, default=str)

    @classmethod
    def success(cls, tool: str, content: str, **data: Any) -> ToolResult:
        return cls(ok=True, tool=tool, content=content, data=data)

    @classmethod
    def failure(cls, tool: str, error: str, *, content: str | None = None, **data: Any) -> ToolResult:
        return cls(
            ok=False,
            tool=tool,
            content=content or f"Tool {tool} failed: {error}",
            data=data,
            error=error,
        )


@dataclass(frozen=True)
class ToolSpec:
    """Everything needed to expose and execute one tool."""

    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., ToolResult]
    #: True for tools with side effects.  Used by the registry to log an audit
    #: row and by the agent to require explicit user intent for actions.
    writes: bool = False

    def json_schema(self) -> dict[str, Any]:
        """OpenAI/Groq-style function schema derived from the pydantic model."""
        schema = self.args_model.model_json_schema()
        # Groq expects an object schema; pydantic already provides one, but the
        # "title" of the top-level model is noise for the model, so drop it.
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }


# ---------------------------------------------------------------------------
# Argument models
# ---------------------------------------------------------------------------


class SearchKnowledgeBaseArgs(BaseModel):
    query: str = Field(..., min_length=2, description="Natural-language search query.")
    top_k: int = Field(default=3, ge=1, le=10, description="How many passages to return.")


class CreateTicketArgs(BaseModel):
    subject: str = Field(..., min_length=5, max_length=160, description="Short ticket subject.")
    description: str = Field(
        ..., min_length=10, max_length=2000, description="What happened, with enough detail to act."
    )
    priority: str = Field(
        default="P3",
        description="Incident priority: P1 (critical outage), P2 (major), P3 (minor), P4 (cosmetic).",
        pattern="^P[1-4]$",
    )
    requester_email: str = Field(..., description="Email of the person raising the ticket.")


class TicketStatusArgs(BaseModel):
    ticket_id: str = Field(..., description="Ticket id such as TCK-1004.", pattern="^TCK-\\d+$")


class SubmitExpenseArgs(BaseModel):
    employee_email: str = Field(..., description="Email of the employee claiming the expense.")
    amount_usd: float = Field(..., gt=0, le=100000, description="Total amount in USD.")
    category: str = Field(
        ...,
        description="One of: airfare, hotel, meals, ground_transport, internet, other.",
        pattern="^(airfare|hotel|meals|ground_transport|internet|other)$",
    )
    description: str = Field(..., min_length=5, max_length=500, description="Business purpose.")
    receipts_attached: bool = Field(
        default=True, description="Whether itemised receipts above USD 25 are attached."
    )


class LookupEmployeeArgs(BaseModel):
    query: str = Field(
        ..., min_length=2, description="Name, email, department or team to search the directory for."
    )


class SendNotificationArgs(BaseModel):
    channel: str = Field(
        ..., description="One of: email, slack, status_page.", pattern="^(email|slack|status_page)$"
    )
    recipient: str = Field(
        ..., min_length=3, description="Internal email address, Slack channel, or 'public'."
    )
    message: str = Field(..., min_length=5, max_length=1500, description="Message body to send.")


@runtime_checkable
class ToolExecutor(Protocol):
    """Interface the agent graph depends on.

    Both the in-process ``ToolRegistry`` and the MCP client adapter satisfy it,
    which is what lets the agent switch between local tools and a remote MCP
    server without a single change in the graph code.  Depending on this narrow
    protocol instead of a concrete class is the seam that makes the two
    transports interchangeable (and the graph testable with a stub).
    """

    @property
    def names(self) -> list[str]: ...

    def openai_tools(self) -> list[dict[str, Any]]:
        """Function schemas for LLM tool binding."""
        ...

    def execute(
        self, name: str, arguments: dict[str, Any] | str | None, *, thread_id: str = "-"
    ) -> ToolResult:
        """Run one tool and always return a ``ToolResult`` (never raise)."""
        ...


__all__ = [
    "ToolExecutor",
    "CreateTicketArgs",
    "LookupEmployeeArgs",
    "SearchKnowledgeBaseArgs",
    "SendNotificationArgs",
    "SubmitExpenseArgs",
    "TicketStatusArgs",
    "ToolResult",
    "ToolSpec",
]
