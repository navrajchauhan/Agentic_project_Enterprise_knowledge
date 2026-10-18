"""LLM access layer.

``base``         model factory, message helpers, tolerant JSON invocation
``groq_client``  ChatGroq construction (live mode)
``offline``      deterministic extractive test double (no API key needed)
"""

from __future__ import annotations

from .base import (
    bind_tools,
    build_chat_model,
    invoke_json,
    json_schema_hint,
    message_text,
    tool_call_args,
    tool_call_name,
)

__all__ = [
    "bind_tools",
    "build_chat_model",
    "invoke_json",
    "json_schema_hint",
    "message_text",
    "tool_call_args",
    "tool_call_name",
]
