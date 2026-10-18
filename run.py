#!/usr/bin/env python
"""Convenience entry point: ``python run.py <command>``.

Adds ``src/`` to ``sys.path`` so the project runs from a fresh clone without an
editable install - the friendliest possible onboarding for a reviewer who just
wants to try the agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enterprise_agent.cli import main  # noqa: E402 - path setup must run first

if __name__ == "__main__":
    sys.exit(main())
