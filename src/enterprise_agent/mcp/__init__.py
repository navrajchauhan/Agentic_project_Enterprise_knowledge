"""MCP integration: a stdio tool server and its synchronous client adapter.

The server (``server``) publishes the enterprise actions as MCP tools; the
client (``client``) lets the synchronous LangGraph nodes call them.  Importing
this package is cheap - the MCP SDK is imported lazily inside the implementation
modules - so ``TOOL_PROVIDER=direct`` never pays for it.
"""

from __future__ import annotations

__all__ = ["MCPToolClient"]


def __getattr__(name: str) -> object:
    """Import the client lazily so ``import enterprise_agent.mcp`` stays light."""
    if name == "MCPToolClient":
        from .client import MCPToolClient

        return MCPToolClient
    raise AttributeError(name)
