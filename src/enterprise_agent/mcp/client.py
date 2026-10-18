"""MCP client: consume the tool server from synchronous agent code.

The MCP Python SDK is asyncio-native, while LangGraph nodes in this project are
plain synchronous functions.  Bridging the two safely is the whole job of this
module:

* a **dedicated thread** owns an asyncio event loop for the lifetime of the
  client (a session cannot be driven from a different loop than it was created
  in),
* the session is kept **open** between calls - spawning a server per tool call
  would cost ~1s of process startup each time,
* synchronous callers use ``asyncio.run_coroutine_threadsafe(...).result()``,
  which is the documented way to cross that boundary.

Failures never escape as exceptions into the graph: every problem becomes a
``ToolResult`` with ``ok=False``, and the provider builder decides whether to
fall back to the in-process registry.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Settings, get_settings
from ..logging_setup import get_logger
from ..tools.models import ToolResult

logger = get_logger(__name__)


def _coerce_arguments(arguments: dict[str, Any] | str | None) -> dict[str, Any]:
    """Normalise tool arguments to a dict (models sometimes emit a JSON string)."""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return dict(arguments)


def _first_field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present field, tolerating MCP v1/v2 naming.

    The SDK renamed fields from camelCase to snake_case in v2
    (``inputSchema`` -> ``input_schema``, ``isError`` -> ``is_error``), so the
    client accepts either rather than pinning the project to one major version.
    """
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _describe(exc: BaseException) -> str:
    """Flatten an ExceptionGroup into a readable one-line message."""
    parts: list[str] = []
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop(0)
        sub = getattr(current, "exceptions", None)
        if sub:
            stack.extend(sub)
            continue
        parts.append(f"{type(current).__name__}: {current}")
    return "; ".join(parts) or repr(exc)


class MCPToolClient:
    """Synchronous adapter around an MCP stdio session."""

    def __init__(
        self,
        command: str,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd
        self.timeout = timeout

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._session: Any = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._tools: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> MCPToolClient:
        """Build a client that launches this project's MCP server.

        ``MCP_SERVER_COMMAND`` may be a venv-relative path (the default); it is
        resolved against the project root and falls back to the current
        interpreter if it does not exist, so a fresh clone still works.
        """
        settings = settings or get_settings()

        command = Path(settings.mcp_server_command)
        if not command.is_absolute():
            command = PROJECT_ROOT / command
        if not command.exists():
            logger.warning(
                "MCP_SERVER_COMMAND %s not found; using the current interpreter (%s)",
                settings.mcp_server_command,
                sys.executable,
            )
            command = Path(sys.executable)

        # The server runs as `python -m enterprise_agent.mcp.server`, so src/ must
        # be importable in the child process even when the package is not
        # installed into the venv.
        src_dir = str(PROJECT_ROOT / "src")
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        return cls(
            command=str(command),
            args=["-m", settings.mcp_server_module],
            env=env,
            cwd=PROJECT_ROOT,
            timeout=settings.mcp_call_timeout_seconds,
        )

    def start(self) -> None:
        """Start the server and complete the MCP handshake."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._run_loop, name="mcp-client", daemon=True)
        self._thread.start()

        if not self._ready.wait(timeout=self.timeout + 15.0):
            raise RuntimeError(
                f"MCP server did not respond within {self.timeout + 15.0:.0f}s "
                f"({self.command} {' '.join(self.args)})"
            )
        if self._start_error is not None:
            raise RuntimeError(f"MCP session failed: {self._start_error}") from self._start_error

        logger.info("MCP session established with %d tool(s)", len(self._tools))

    def close(self) -> None:
        """Ask the session to shut down and stop the client thread."""
        if self._loop is not None and self._stop_event is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:  # loop already closed
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> MCPToolClient:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------- async internals
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._lifecycle())
        except BaseException as exc:  # noqa: BLE001 - reported through _start_error
            self._start_error = exc
            logger.error("MCP client loop terminated: %s", exc)
        finally:
            self._ready.set()  # never leave the caller waiting
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _lifecycle(self) -> None:
        """Open the stdio session and keep it alive until ``close()``."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.command, args=self.args, env=self.env, cwd=self.cwd
        )
        self._stop_event = asyncio.Event()

        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self._tools = [
                        {
                            "name": tool.name,
                            "description": _first_field(tool, "description", default="") or "",
                            "inputSchema": _first_field(
                                tool,
                                "input_schema",
                                "inputSchema",
                                default={"type": "object", "properties": {}},
                            ),
                        }
                        for tool in listing.tools
                    ]
                    self._session = session
                    self._ready.set()
                    await self._stop_event.wait()
        except Exception as exc:  # noqa: BLE001
            self._start_error = RuntimeError(_describe(exc))
            logger.error("MCP session error: %s", _describe(exc))
        finally:
            self._session = None
            self._ready.set()

    def _submit(self, coroutine: Any) -> Any:
        """Run a coroutine on the client loop and block for its result."""
        if self._loop is None:
            raise RuntimeError("MCP client loop is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result(timeout=self.timeout)

    # ---------------------------------------------------------- transport API
    @property
    def names(self) -> list[str]:
        return [tool["name"] for tool in self._tools]

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def openai_tools(self) -> list[dict[str, Any]]:
        """Expose MCP tool schemas in the shape the LLM expects."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                },
            }
            for tool in self._tools
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | str | None,
        *,
        thread_id: str = "-",
    ) -> ToolResult:
        """Call a tool over MCP and normalise the response."""
        if self._session is None:
            return ToolResult.failure(
                name,
                "mcp_not_connected",
                content=f"MCP session is not connected; cannot call {name}.",
            )
        if name not in self.names:
            return ToolResult.failure(
                name,
                "unknown_tool",
                content=f"MCP server does not expose {name!r}. Available: {', '.join(self.names)}.",
            )

        payload = _coerce_arguments(arguments)
        try:
            result = self._submit(self._session.call_tool(name, payload))
        except Exception as exc:  # noqa: BLE001 - timeout/protocol errors
            logger.error("MCP call %s failed: %s", name, exc)
            return ToolResult.failure(
                name, "mcp_call_failed", content=f"MCP call {name} failed: {exc}"
            )

        text = "\n".join(
            block.text
            for block in (_first_field(result, "content", default=[]) or [])
            if _first_field(block, "type", default="") == "text"
            and _first_field(block, "text")
        ).strip()
        is_error = bool(_first_field(result, "is_error", "isError", default=False))

        # The server returns the ToolResult JSON envelope; unwrap it when we can
        # so callers see the same structure as the in-process registry.
        try:
            envelope = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            envelope = None

        if isinstance(envelope, dict) and "ok" in envelope:
            return ToolResult(
                ok=bool(envelope.get("ok")) and not is_error,
                tool=str(envelope.get("tool", name)),
                content=str(envelope.get("content", text)),
                data=envelope.get("data") or {},
                error=envelope.get("error"),
            )

        # Non-envelope response: the SDK itself rejected the request (for example
        # a missing required argument, validated by the server's tool signature
        # before our registry runs). Normalise the error code so callers can
        # branch on it, while keeping the detailed message for the model.
        error_code = None
        if is_error:
            lowered = text.lower()
            error_code = (
                "invalid_arguments"
                if "validation error" in lowered or "field required" in lowered
                else "mcp_tool_error"
            )

        return ToolResult(
            ok=not is_error,
            tool=name,
            content=text or f"{name} returned no content",
            data={},
            error=error_code,
        )

    # ---------------------------------------------------------------- helpers
    def describe(self) -> str:
        """Human-readable tool catalogue (``run.py mcp-tools``)."""
        return "\n".join(
            f"- {tool['name']}: {tool['description'][:120]}" for tool in self._tools
        )


def probe(settings: Settings | None = None) -> str:
    """Connect, list tools, disconnect - used by the CLI to verify MCP wiring."""
    client = MCPToolClient.from_settings(settings)
    try:
        client.start()
        return client.describe()
    finally:
        client.close()
