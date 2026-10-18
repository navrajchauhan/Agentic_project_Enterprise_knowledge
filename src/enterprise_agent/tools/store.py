"""SQLite persistence for tool side effects and the audit trail.

An agent that can *act* needs a record of what it did.  Two tables matter:

``audit_log``
    One row per tool invocation: who (thread), what (tool + arguments), outcome
    and message.  This is the enterprise requirement that turns "the agent did
    something" into "the agent did exactly this, at this time, for this user".

``tickets`` / ``expenses`` / ``notifications``
    The business objects the actions create.  They are real rows, not console
    prints, which is what makes follow-up questions ("what is the status of
    TCK-1003?") answerable on the next turn.

SQLite is used deliberately: zero setup, one file, and the schema below is
small enough to migrate by hand.  A production system would use Postgres, but
the access pattern (single process, low write volume) is identical.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    row_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       TEXT UNIQUE NOT NULL,
    subject         TEXT NOT NULL,
    description     TEXT NOT NULL,
    priority        TEXT NOT NULL,
    requester_email TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',
    note            TEXT DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS expenses (
    expense_id        TEXT PRIMARY KEY,
    employee_email    TEXT NOT NULL,
    amount_usd        REAL NOT NULL,
    category          TEXT NOT NULL,
    description       TEXT NOT NULL,
    receipts_attached INTEGER NOT NULL,
    decision          TEXT NOT NULL,
    approval_level    TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id TEXT PRIMARY KEY,
    channel         TEXT NOT NULL,
    recipient       TEXT NOT NULL,
    message         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    thread_id      TEXT NOT NULL,
    tool           TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    ok             INTEGER NOT NULL,
    message        TEXT NOT NULL
);
"""


def _now() -> str:
    """ISO-8601 UTC timestamp (sortable, timezone-explicit)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class ActionStore:
    """Thin data-access layer over the agent's SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # A fresh connection per call keeps this class thread-safe (the MCP
        # server may serve requests from several threads) and avoids the
        # "same thread" restriction of a shared connection.
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # -- tickets -----------------------------------------------------------
    def create_ticket(
        self,
        *,
        subject: str,
        description: str,
        priority: str,
        requester_email: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Insert a ticket, or return the existing open one for the same subject.

        Idempotency matters for agents: a retry after a timeout must not create
        a second ticket.  The natural key here is (requester, subject, open).
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM tickets WHERE requester_email = ? AND subject = ? "
                "AND status = 'open' ORDER BY row_id DESC LIMIT 1",
                (requester_email.lower(), subject),
            ).fetchone()
            if existing is not None:
                logger.info("Reusing open ticket %s (idempotent create)", existing["ticket_id"])
                return {**dict(existing), "deduplicated": True}

            cursor = conn.execute(
                "INSERT INTO tickets (ticket_id, subject, description, priority, "
                "requester_email, status, note, created_at) VALUES (?, ?, ?, ?, ?, 'open', ?, ?)",
                ("", subject, description, priority, requester_email.lower(), note, _now()),
            )
            # Derive the human id from the autoincrement row: monotonic and unique.
            ticket_id = f"TCK-{1000 + int(cursor.lastrowid or 0)}"
            conn.execute(
                "UPDATE tickets SET ticket_id = ? WHERE row_id = ?",
                (ticket_id, cursor.lastrowid),
            )
            row = conn.execute(
                "SELECT * FROM tickets WHERE row_id = ?", (cursor.lastrowid,)
            ).fetchone()
            return {**dict(row), "deduplicated": False}

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id.upper(),)
            ).fetchone()
            return dict(row) if row else None

    # -- expenses ----------------------------------------------------------
    def record_expense(
        self,
        *,
        employee_email: str,
        amount_usd: float,
        category: str,
        description: str,
        receipts_attached: bool,
        decision: str,
        approval_level: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO expenses (expense_id, employee_email, amount_usd, category, "
                "description, receipts_attached, decision, approval_level, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "",  # placeholder, updated below with the generated id
                    employee_email.lower(),
                    amount_usd,
                    category,
                    description,
                    int(receipts_attached),
                    decision,
                    approval_level,
                    _now(),
                ),
            )
            expense_id = f"EXP-{2000 + int(cursor.lastrowid or 0)}"
            conn.execute(
                "UPDATE expenses SET expense_id = ? WHERE rowid = ?",
                (expense_id, cursor.lastrowid),
            )
            row = conn.execute(
                "SELECT * FROM expenses WHERE rowid = ?", (cursor.lastrowid,)
            ).fetchone()
            return dict(row)

    # -- notifications -----------------------------------------------------
    def record_notification(
        self, *, channel: str, recipient: str, message: str
    ) -> dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO notifications (notification_id, channel, recipient, message, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                ("", channel, recipient, message, _now()),
            )
            notification_id = f"NTF-{3000 + int(cursor.lastrowid or 0)}"
            conn.execute(
                "UPDATE notifications SET notification_id = ? WHERE rowid = ?",
                (notification_id, cursor.lastrowid),
            )
            row = conn.execute(
                "SELECT * FROM notifications WHERE rowid = ?", (cursor.lastrowid,)
            ).fetchone()
            return dict(row)

    # -- audit -------------------------------------------------------------
    def log_tool_call(
        self,
        *,
        thread_id: str,
        tool: str,
        arguments: dict[str, Any],
        ok: bool,
        message: str,
    ) -> None:
        """Append one audit row.  Never raises: audit must not break an action."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO audit_log (created_at, thread_id, tool, arguments_json, "
                    "ok, message) VALUES (?, ?, ?, ?, ?, ?)",
                    (_now(), thread_id or "-", tool, json.dumps(arguments, default=str), int(ok), message),
                )
        except sqlite3.Error as exc:  # pragma: no cover - disk/permission problems
            logger.error("Failed to write audit row for %s: %s", tool, exc)

    def recent_audit(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
