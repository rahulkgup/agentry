"""Remember per-sender triage outcomes in SQLite (Backlog #8).

After the same sender is classified into the same category on 3 consecutive runs,
we skip the triager LLM and reuse the last classification.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .state import Classification

_lock = threading.Lock()
_PATH = Path.home() / ".inbox_agent" / "sender_memory.sqlite"


def _conn() -> sqlite3.Connection:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(_PATH)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS sender_cache (
            sender_email TEXT PRIMARY KEY,
            last_category TEXT NOT NULL,
            streak INTEGER NOT NULL DEFAULT 1,
            last_seen TEXT
        )
        """
    )
    c.commit()
    return c


def _norm(email: str) -> str:
    return email.strip().lower()


def remember(sender_email: str, category: str) -> None:
    """Update streak: same category increments; any change resets to 1."""
    with _lock:
        with _conn() as c:
            row = c.execute(
                "SELECT last_category, streak FROM sender_cache WHERE sender_email = ?",
                (_norm(sender_email),),
            ).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO sender_cache (sender_email, last_category, streak, last_seen) "
                    "VALUES (?, ?, 1, datetime('now'))",
                    (_norm(sender_email), category),
                )
            else:
                last_cat, streak = row
                if last_cat == category:
                    new_streak = streak + 1
                else:
                    new_streak = 1
                c.execute(
                    "UPDATE sender_cache SET last_category = ?, streak = ?, last_seen = datetime('now') "
                    "WHERE sender_email = ?",
                    (category, new_streak, _norm(sender_email)),
                )
            c.commit()


def cached_classification_if_ready(sender_email: str) -> Classification | None:
    """If streak >= 3, return a copy of the cached triage; else None."""
    with _lock:
        with _conn() as c:
            row = c.execute(
                "SELECT last_category, streak FROM sender_cache WHERE sender_email = ?",
                (_norm(sender_email),),
            ).fetchone()
            if row is None or row[1] < 3:
                return None
            cat = row[0]
            if cat not in (
                "needs_reply",
                "fyi",
                "newsletter",
                "promo_or_spam",
                "calendar",
            ):
                return None
            return Classification(
                category=cat,  # type: ignore[arg-type]
                urgency="normal",
                summary="(cached — same sender/category 3+ runs in a row)",
                intent=None,
                suggested_action="archive" if cat != "needs_reply" else None,
            )
