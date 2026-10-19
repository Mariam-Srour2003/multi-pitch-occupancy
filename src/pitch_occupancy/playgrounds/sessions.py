"""Looking back: the stored checks grouped for the Records and Results pages.

* :func:`records` - one row per playground per period of a day (the Records page).
* :func:`sessions` - one verdict per playground per 1.5-hour session, on a grid through
  :data:`SESSION_ANCHOR` (the Results page).

"Minutes playing" is the number of playing checks times the playground's check interval, so
it is an estimate whose step is that interval.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from datetime import time as dtime

from pitch_occupancy.playgrounds import store

__all__ = ["latest", "records", "session_bounds", "sessions", "every_text", "PERIOD_CHOICES",
           "SESSION_MINUTES", "SESSION_ANCHOR", "SESSION_PLAYED", "SESSION_SOME"]

#: Sessions: 1.5 hours each, on a grid that passes through 09:30 (..., 08:00, 09:30, 11:00, ...
#: 21:30, 23:00, 00:30, ...).
SESSION_MINUTES = 90
SESSION_ANCHOR = "09:30"
#: Share of a session's checks that must be playing for the session to count as played.
SESSION_PLAYED = 0.5
#: ... and below which, if any were playing, it was played only a little.
SESSION_SOME = 0.15
#: Periods the Records page groups checks into, in minutes; 90 is the default.
PERIOD_CHOICES: tuple[int, ...] = (15, 30, 60, 90, 120, 180)


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}" if minutes < 24 * 60 else "24:00"


def latest(conn) -> dict[str, dict]:
    """Each playground's most recent check."""
    rows = conn.execute(
        """SELECT p.* FROM playground_checks p JOIN (
               SELECT MAX(check_id) AS id FROM playground_checks GROUP BY playground_id
           ) m ON p.check_id = m.id""").fetchall()
    return {r["playground_id"]: dict(r) for r in rows}


def records(conn, day: date, *, period_minutes: int = 90,
            playground_id: str | None = None) -> list[dict]:
    """One row per playground per period of ``day``: people, most at once, time playing."""
    if period_minutes not in PERIOD_CHOICES:
        raise ValueError(f"period_minutes must be one of {list(PERIOD_CHOICES)}")
    every = {p["id"]: p["every_seconds"] for p in store.load()}
    sql = "SELECT * FROM playground_checks WHERE substr(checked_at, 1, 10) = ?"
    args: list = [day.isoformat()]
    if playground_id:
        sql += " AND playground_id = ?"
        args.append(playground_id)
    groups: dict[tuple, list] = {}
    for r in conn.execute(sql + " ORDER BY checked_at", args).fetchall():
        when = datetime.fromisoformat(r["checked_at"])
        bucket = (when.hour * 60 + when.minute) // period_minutes
        groups.setdefault((bucket, r["playground_id"]), []).append(r)
    out = []
    for (bucket, pid), rows in sorted(groups.items()):
        counted = [r["people"] for r in rows if r["people"] is not None]
        playing = sum(1 for r in rows if r["playing"])
        start = bucket * period_minutes
        out.append({
            "playground_id": pid, "day": day.isoformat(),
            "start": _fmt(start), "end": _fmt(min(start + period_minutes, 24 * 60)),
            "checks": len(rows), "checks_without_picture": len(rows) - len(counted),
            "people_seen": any(n > 0 for n in counted) if counted else None,
            "max_people": max(counted) if counted else None,
            "avg_people": round(sum(counted) / len(counted), 1) if counted else None,
            "playing_checks": playing,
            "minutes_playing": round(playing * every.get(pid, 0) / 60),
        })
    return out


def session_bounds(when: datetime) -> tuple[datetime, datetime]:
    """The 1.5-hour session ``when`` falls in, on the grid through :data:`SESSION_ANCHOR`."""
    anchor = dtime.fromisoformat(SESSION_ANCHOR)
    base = when.replace(hour=anchor.hour, minute=anchor.minute, second=0, microsecond=0)
    length = timedelta(minutes=SESSION_MINUTES)
    start = base + ((when - base) // length) * length
    return start, start + length


def _session_verdict(share: float | None, max_people: int | None) -> str:
    if share is None:
        return "No pictures"
    if share >= SESSION_PLAYED:
        return "Playing"
    if share >= SESSION_SOME:
        return "Played a little"
    if share > 0:
        return "Mostly empty"
    return "Some people, no game" if max_people else "Empty"


def sessions(conn, start: datetime, end: datetime, *, playground_id: str | None = None,
             ) -> list[dict]:
    """One row per playground per 1.5-hour session between ``start`` and ``end``.

    Each session gets one verdict from its checks: **Playing** when at least half were
    playing, **Played a little** from 15 % up, **Mostly empty** below that, **Some people, no
    game** when people were seen but never enough, **Empty** when nobody was seen. Also the
    average and most people, the minutes played and how many pictures were saved.
    """
    saved_pgs = store.load()
    every = {p["id"]: p["every_seconds"] for p in saved_pgs}
    names = {p["id"]: p["name"] for p in saved_pgs}
    sql = "SELECT * FROM playground_checks WHERE checked_at >= ? AND checked_at < ?"
    args: list = [start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")]
    if playground_id:
        sql += " AND playground_id = ?"
        args.append(playground_id)
    saved = {r[0]: r[1] for r in conn.execute(
        """SELECT playground_check_id, COUNT(*) FROM camera_checks
           WHERE picture_path IS NOT NULL AND checked_at >= ? AND checked_at < ?
           GROUP BY playground_check_id""", args[:2]).fetchall()}
    groups: dict[tuple, list] = {}
    for r in conn.execute(sql + " ORDER BY checked_at", args).fetchall():
        s, _ = session_bounds(datetime.fromisoformat(r["checked_at"]))
        groups.setdefault((r["playground_id"], s), []).append(r)
    out = []
    for (pid, s), rows in sorted(groups.items()):
        counted = [r["people"] for r in rows if r["people"] is not None]
        playing = sum(1 for r in rows if r["playing"])
        share = playing / len(counted) if counted else None
        mx = max(counted) if counted else None
        out.append({
            "playground_id": pid, "playground": names.get(pid, pid),
            "start": s.isoformat(timespec="minutes"),
            "end": (s + timedelta(minutes=SESSION_MINUTES)).isoformat(timespec="minutes"),
            "checks": len(rows), "checks_without_picture": len(rows) - len(counted),
            "avg_people": round(sum(counted) / len(counted), 1) if counted else None,
            "max_people": mx, "playing_checks": playing,
            "playing_share": round(share, 2) if share is not None else None,
            "minutes_playing": round(playing * every.get(pid, 0) / 60),
            "saved_pictures": sum(saved.get(r["check_id"], 0) for r in rows),
            "assist_upgrades": sum(1 for r in rows if r["playing_source"] == "assist"),
            "verdict": _session_verdict(share, mx),
        })
    return out


def every_text(seconds: int) -> str:
    """"30 seconds", "minute", "5 minutes" - how often, in words."""
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    return "minute" if minutes == 1 else f"{minutes} minutes"
