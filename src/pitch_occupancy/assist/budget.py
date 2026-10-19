"""How much of each provider's free quota has been used, kept in the database.

In the database rather than in memory so a restart - or two processes - cannot start the
count again from zero. Days are UTC days. Providers reset at different times (Groq at
midnight UTC, Gemini at midnight Pacific), and `HEADROOM` is low enough that a day counted on
the wrong clock still stays under the free limit.

**A request is reserved before it is sent** (`reserve`), in one write transaction, and its
tokens are settled afterwards (`settle`). Checking first and counting after the reply would
let eight playground threads all pass the check during one slow request. Every attempt
counts, answered or not: some providers count failed requests against the quota too.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta

from pitch_occupancy.assist.providers import Provider

__all__ = ["HEADROOM", "reserve", "settle", "rest", "usage"]

#: The share of a free limit the budget lets through. Half: a day split across two reset
#: clocks can count at most double, and double half is still the limit.
HEADROOM = 0.5

_minute: dict[str, deque[float]] = {}
_minute_lock = threading.Lock()


def _day(now: datetime) -> str:
    return now.astimezone(UTC).date().isoformat()


def _resting_until(conn: sqlite3.Connection, provider: str, now: datetime) -> datetime | None:
    row = conn.execute("SELECT MAX(cooldown_until) FROM assist_usage WHERE provider = ?",
                       (provider,)).fetchone()
    if row is None or not row[0]:
        return None
    until = datetime.fromisoformat(row[0])
    return until if until > now else None


def reserve(conn: sqlite3.Connection, provider: Provider, now: datetime) -> tuple[bool, str]:
    """Take one request from today's budget if it fits; ``(False, why)`` when it does not."""
    until = _resting_until(conn, provider.name, now)
    if until is not None:
        return False, (f"{provider.name} is resting after a quota refusal until "
                       f"{until.astimezone():%H:%M}")
    with _minute_lock:
        recent = _minute.setdefault(provider.name, deque())
        cutoff = time.monotonic() - 60.0
        while recent and recent[0] < cutoff:
            recent.popleft()
        if len(recent) + 1 > max(1, int(provider.requests_per_minute * HEADROOM)):
            return False, f"{provider.name}: too many requests this minute"
        recent.append(time.monotonic())
    day = _day(now)
    conn.commit()  # end any implicit transaction, so BEGIN starts a fresh one
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT requests, tokens FROM assist_usage WHERE provider = ? "
                           "AND day = ?", (provider.name, day)).fetchone()
        requests, tokens = (row[0], row[1]) if row is not None else (0, 0)
        why = ""
        if requests + 1 > int(provider.requests_per_day * HEADROOM):
            why = f"{provider.name}: today's free requests are used up"
        elif (provider.tokens_per_day is not None and tokens + provider.tokens_per_request
              > int(provider.tokens_per_day * HEADROOM)):
            why = f"{provider.name}: today's free tokens are used up"
        if why:
            conn.execute("ROLLBACK")
            return False, why
        conn.execute(
            """INSERT INTO assist_usage (provider, day, requests, tokens, failures)
               VALUES (?, ?, 1, ?, 0)
               ON CONFLICT(provider, day) DO UPDATE SET requests = requests + 1,
                   tokens = tokens + excluded.tokens""",
            (provider.name, day, provider.tokens_per_request))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return True, ""


def settle(conn: sqlite3.Connection, provider: Provider, now: datetime, *, tokens: int | None,
           failed: bool = False) -> None:
    """Replace a reserved request's estimated tokens with what it really used."""
    actual = provider.tokens_per_request if tokens is None else int(tokens)
    conn.execute(
        """UPDATE assist_usage SET tokens = MAX(0, tokens - ? + ?), failures = failures + ?
           WHERE provider = ? AND day = ?""",
        (provider.tokens_per_request, actual, int(failed), provider.name, _day(now)))
    conn.commit()


def rest(conn: sqlite3.Connection, provider: Provider, now: datetime,
         seconds: float | None) -> None:
    """Stop asking ``provider`` after a quota refusal: for ``seconds``, or until tomorrow."""
    if seconds is None or seconds <= 0:
        until = (now.astimezone(UTC) + timedelta(days=1)).replace(
            hour=0, minute=5, second=0, microsecond=0)
    else:
        until = now.astimezone(UTC) + timedelta(seconds=min(seconds, 24 * 3600))
    conn.execute(
        """INSERT INTO assist_usage (provider, day, requests, tokens, failures, cooldown_until)
           VALUES (?, ?, 0, 0, 0, ?)
           ON CONFLICT(provider, day) DO UPDATE SET cooldown_until = excluded.cooldown_until""",
        (provider.name, _day(now), until.isoformat(timespec="seconds")))
    conn.commit()


def usage(conn: sqlite3.Connection, now: datetime) -> list[dict]:
    """Today's use per provider, for the status page."""
    return [dict(r) for r in conn.execute(
        "SELECT * FROM assist_usage WHERE day = ? ORDER BY provider", (_day(now),))]
