"""Chat-model plumbing shared by every node.

Two decisions worth calling out:

**One factory, three purposes.**  ``primary`` answers questions, ``fast`` does
the cheap classification work (routing, grading, query rewriting) and ``judge``
scores answers during evaluation.  Using the small model for the mechanical
steps keeps latency and cost down without hurting quality - a routing decision
does not need a 70B model.

**Tolerant JSON parsing instead of provider-specific structured output.**
``invoke_json`` asks for JSON, parses it defensively, validates it with pydantic
and retries once with the validation error attached.  That works identically on
Groq, on the offline test double, and on any future provider, and it degrades to
a safe default instead of raising into the graph.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..logging_setup import get_logger
from ..utils.text import extract_json_object

logger = get_logger(__name__)

Purpose = Literal["primary", "fast", "judge"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def build_chat_model(settings: Settings, *, purpose: Purpose = "primary") -> BaseChatModel:
    """Return the configured chat model for a purpose.

    Falls back to the deterministic offline model when no Groq key is set, so
    the whole application - graph, tools, guardrails, evaluation - runs without
    external API calls.
    """
    if settings.is_offline:
        from .offline import OfflineChatModel  # local import avoids a cycle

        logger.warning(
            "GROQ_API_KEY is not set - using the offline deterministic model. "
            "Add your key to .env for real LLM answers."
        )
        return OfflineChatModel()

    from .groq_client import build_groq_model

    model_name = {
        "primary": settings.llm_model,
        "fast": settings.llm_fast_model,
        "judge": settings.llm_judge_model,
    }[purpose]
    return build_groq_model(settings, model_name=model_name)


def message_text(message: BaseMessage | str) -> str:
    """Extract plain text from a message.

    Some providers return ``content`` as a list of typed blocks
    (``[{"type": "text", "text": "..."}]``) instead of a string, so normalise
    both shapes to a single string.
    """
    if isinstance(message, str):
        return message
    content: Any = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content or "")


def tool_call_name(call: dict[str, Any]) -> str:
    """Read a tool name from either normalised or raw OpenAI tool-call shapes."""
    if "name" in call and call["name"]:
        return str(call["name"])
    function = call.get("function") or {}
    return str(function.get("name", ""))


def tool_call_args(call: dict[str, Any]) -> dict[str, Any] | str:
    """Read tool arguments from a tool call, tolerating a JSON string form."""
    if "args" in call and call["args"] is not None:
        return call["args"]
    function = call.get("function") or {}
    return function.get("arguments", {})


def bind_tools(model: BaseChatModel, tools: list[dict[str, Any]]) -> BaseChatModel:
    """Bind function schemas, tolerating providers without native support."""
    try:
        return model.bind_tools(tools)
    except (NotImplementedError, AttributeError):  # pragma: no cover - provider specific
        logger.warning("Model does not implement bind_tools; falling back to bind()")
        return model.bind(tools=tools)


def invoke_json(
    model: BaseChatModel,
    *,
    system: str,
    human: str,
    schema: type[SchemaT],
    attempts: int = 2,
) -> SchemaT | None:
    """Ask the model for a JSON object and validate it into ``schema``.

    Parameters
    ----------
    attempts
        Total model calls.  The second attempt replays the validation error,
        which fixes the overwhelming majority of malformed responses ("2" vs
        2, a missing field, a nested object where a string was expected).

    Returns
    -------
    The validated model instance, or ``None`` when the model could not produce
    valid JSON.  Callers must handle ``None`` - usually by taking a safe default
    path - rather than assuming success.
    """
    messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=human)]

    for attempt in range(1, attempts + 1):
        try:
            response = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - network/quota errors are expected
            logger.error("LLM call failed (%s/%s): %s", attempt, attempts, exc)
            return None

        raw = message_text(response)
        payload = extract_json_object(raw)

        if payload is None:
            logger.warning("Attempt %d: response contained no JSON object", attempt)
            if attempt < attempts:
                messages.append(AIMessage(content=raw[:500]))
                messages.append(
                    HumanMessage(
                        content=(
                            "That reply did not contain a JSON object. Reply again with "
                            "ONLY a single JSON object, no prose and no code fences."
                        )
                    )
                )
                continue
            return None

        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            logger.warning("Attempt %d: JSON failed validation: %s", attempt, exc.errors())
            if attempt < attempts:
                messages.append(AIMessage(content=raw[:500]))
                messages.append(
                    HumanMessage(
                        content=(
                            "That JSON did not match the required schema. "
                            f"Problems: {exc.errors(include_url=False)}. "
                            "Reply again with ONLY the corrected JSON object."
                        )
                    )
                )
                continue
            return None

    return None  # pragma: no cover - loop always returns


def json_schema_hint(schema: type[BaseModel]) -> str:
    """Compact description of a pydantic schema for prompt injection.

    Sending the full JSON Schema costs tokens and confuses small models; a
    field-by-field description with allowed values is enough.
    """
    lines = []
    for name, field in schema.model_fields.items():
        annotation = getattr(field.annotation, "__name__", str(field.annotation))
        description = (field.description or "").strip()
        lines.append(f'  - "{name}" ({annotation}): {description}')
    return "\n".join(lines)


__all__ = [
    "Purpose",
    "bind_tools",
    "build_chat_model",
    "invoke_json",
    "json_schema_hint",
    "message_text",
    "tool_call_args",
    "tool_call_name",
]
