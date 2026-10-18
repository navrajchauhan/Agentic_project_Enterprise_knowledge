"""Tool provider selection: MCP when available, in-process otherwise.

The agent only ever sees the ``ToolExecutor`` protocol, so this module is the
single place that decides *where tools come from*.

``TOOL_PROVIDER=auto`` (default)
    Try the MCP stdio server.  If it cannot start (SDK missing, subprocess
    refused, protocol error) log a warning and fall back to the in-process
    registry.  "Demo works, production can harden" beats "demo fails".
``TOOL_PROVIDER=mcp``
    Require MCP; raise if it fails.  Use this in an environment where the MCP
    server is the supported integration path.
``TOOL_PROVIDER=direct``
    Always use the in-process registry (fastest, no subprocess).
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..logging_setup import get_logger
from .models import ToolExecutor
from .registry import ToolRegistry, build_registry
from .store import ActionStore

logger = get_logger(__name__)


def build_tool_provider(
    settings: Settings,
    *,
    store: ActionStore,
    retriever: Any,
) -> ToolExecutor:
    """Return a ready-to-use tool executor for the configured transport."""
    mode = settings.tool_provider

    if mode == "direct":
        logger.info("Using in-process tool registry (TOOL_PROVIDER=direct)")
        return build_registry(store=store, retriever=retriever, settings=settings)

    # Imported lazily: the MCP SDK pulls in a sizable dependency tree and the
    # direct path should not pay for it.
    from ..mcp.client import MCPToolClient

    try:
        client = MCPToolClient.from_settings(settings)
        client.start()
        logger.info("Connected to MCP server with %d tool(s)", len(client.names))
        return client
    except Exception as exc:  # noqa: BLE001 - any transport failure means fallback
        message = f"Could not start the MCP tool server: {exc}"
        if mode == "mcp":
            raise RuntimeError(message) from exc
        logger.warning("%s - falling back to the in-process registry", message)
        return build_registry(store=store, retriever=retriever, settings=settings)


__all__ = ["ToolRegistry", "ToolExecutor", "build_tool_provider"]
