"""Tool registry: validation, dispatch, audit and schema export.

Responsibilities
----------------
1. **Validate before acting.**  Arguments are parsed into the tool's pydantic
   model first, so an invalid amount never reaches the database.
2. **Never raise into the graph.**  Every failure becomes ``ToolResult.ok=False``
   with a machine-readable ``error`` code.  A tool error must not kill the
   conversation - the agent should explain what went wrong and continue.
3. **Audit everything.**  Each invocation is written to ``audit_log`` with the
   arguments and outcome, giving the enterprise the "what did the agent do"
   record.
4. **Export schemas.**  ``openai_tools()`` produces the JSON-function schemas
   the LLM is bound with, derived from the same pydantic models, so the model
   can never see a schema that disagrees with validation.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ..logging_setup import get_logger
from .models import ToolResult, ToolSpec
from .store import ActionStore

logger = get_logger(__name__)


class ToolRegistry:
    """Holds the tool catalogue and executes tools safely."""

    def __init__(self, *, store: ActionStore | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self.store = store

    # -- catalogue ---------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def register_all(self, specs: list[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    @property
    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in self.names]

    def __len__(self) -> int:
        return len(self._specs)

    def describe(self) -> str:
        """Human-readable catalogue (``run.py tools``)."""
        lines = []
        for spec in self.specs:
            marker = "write" if spec.writes else "read "
            fields = ", ".join(spec.args_model.model_fields)
            lines.append(f"[{marker}] {spec.name}({fields})\n           {spec.description}")
        return "\n".join(lines)

    # -- schema export -----------------------------------------------------
    def openai_tools(self) -> list[dict[str, Any]]:
        """Function schemas in the shape Groq/OpenAI expect for tool calling."""
        return [{"type": "function", "function": spec.json_schema()} for spec in self.specs]

    # -- execution ---------------------------------------------------------
    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | str | None,
        *,
        thread_id: str = "-",
    ) -> ToolResult:
        """Validate arguments, run the tool and record an audit row."""
        spec = self._specs.get(name)
        if spec is None:
            available = ", ".join(self.names)
            return ToolResult.failure(
                name,
                "unknown_tool",
                content=f"Unknown tool {name!r}. Available tools: {available}.",
                available_tools=self.names,
            )

        # 1. Normalise the argument payload.  Models sometimes emit a JSON
        #    string instead of an object, so accept both shapes.
        payload: dict[str, Any]
        if arguments is None:
            payload = {}
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return self._audited(
                    spec,
                    {"raw": arguments},
                    ToolResult.failure(name, "invalid_arguments", content=f"Arguments must be JSON: {exc}"),
                    thread_id,
                )
            if not isinstance(parsed, dict):
                return self._audited(
                    spec,
                    {"raw": arguments},
                    ToolResult.failure(name, "invalid_arguments", content="Arguments must be a JSON object."),
                    thread_id,
                )
            payload = parsed
        else:
            payload = dict(arguments)

        # 2. Validate with pydantic.  This is the guardrail against hallucinated
        #    or malformed arguments and produces a message the LLM can act on.
        try:
            validated = spec.args_model.model_validate(payload)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
                for error in exc.errors()
            )
            return self._audited(
                spec,
                payload,
                ToolResult.failure(
                    name,
                    "invalid_arguments",
                    content=f"Invalid arguments for {name}: {problems}",
                    validation_errors=exc.errors(include_url=False),
                ),
                thread_id,
            )

        # 3. Run it.  Tool bugs are captured, logged with the traceback in the
        #    server log, and surfaced as a generic failure to the model.
        try:
            result = spec.func(**validated.model_dump())
        except Exception as exc:  # noqa: BLE001 - the graph must survive tool bugs
            logger.exception("Tool %s raised an exception", name)
            result = ToolResult.failure(name, "internal_error", content=f"{name} failed: {exc}")

        return self._audited(spec, validated.model_dump(), result, thread_id)

    def _audited(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        result: ToolResult,
        thread_id: str,
    ) -> ToolResult:
        logger.info(
            "tool=%s ok=%s thread=%s args=%s",
            spec.name,
            result.ok,
            thread_id,
            {key: (str(value)[:60]) for key, value in arguments.items()},
        )
        if self.store is not None:
            self.store.log_tool_call(
                thread_id=thread_id,
                tool=spec.name,
                arguments=arguments,
                ok=result.ok,
                message=result.content[:500],
            )
        return result


def build_registry(*, store: ActionStore, retriever: Any, settings: Any) -> ToolRegistry:
    """Convenience builder used by the agent runner and the MCP server."""
    from .enterprise_actions import build_tool_specs

    registry = ToolRegistry(store=store)
    registry.register_all(build_tool_specs(store=store, retriever=retriever, settings=settings))
    return registry
