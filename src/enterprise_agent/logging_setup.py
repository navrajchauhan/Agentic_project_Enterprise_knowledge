"""Logging setup.

A single place that configures the root logger so every module can simply call
``get_logger(__name__)``.  We log to stderr because the MCP server speaks a
line-oriented JSON-RPC protocol on **stdout** - writing log lines there would
corrupt the protocol stream.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

#: Pretty, greppable format: timestamp, level, module, message.
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-38s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger exactly once.

    Repeat calls are no-ops except for level changes, which keeps tests from
    stacking duplicate handlers (a classic source of duplicated log lines).
    """
    global _CONFIGURED
    numeric = getattr(logging, str(level).upper(), logging.INFO)

    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        _CONFIGURED = True

    root.setLevel(numeric)

    # Third-party libraries are noisy at INFO and would drown our own trace.
    for noisy in ("httpx", "httpcore", "urllib3", "mcp", "asyncio", "groq"):
        logging.getLogger(noisy).setLevel(os.environ.get("NOISY_LOG_LEVEL", "WARNING"))


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root logger on first use."""
    if not _CONFIGURED:
        configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    return logging.getLogger(name)
