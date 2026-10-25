"""Results: how the playgrounds went, session by session - the page for showing it to people.

* ``/results``  A grid of playgrounds by 1.5-hour sessions (on the grid through 09:30, see
  `playgrounds.sessions`), each cell one verdict - Playing, Played a little, Mostly empty,
  Some people no game, Empty. Click a cell for that session: the people at every check drawn
  over time against the playing line, how the verdict was reached, and the pictures saved
  during it (with each camera's count drawn on), which open full screen with arrow keys.

The pictures are the ones the checker saved (one check in ``save_every``) under
`playgrounds.snapshots.SNAPSHOT_DIR`, kept under a size cap; a session whose pictures were
deleted to make room still has its numbers.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from pitch_occupancy import playgrounds
from pitch_occupancy.api.simple_pages import _html
from pitch_occupancy.api.templating import render, template
from pitch_occupancy.db.schema import open_db

__all__ = ["router"]

router = APIRouter()

CSS_EXTRA = template("results_page_css_extra.html")

BODY = render("results_page_body.html", CSS_EXTRA=CSS_EXTRA)

JS = template("results_page_js.js")


@router.get("/results", response_class=HTMLResponse, include_in_schema=False)
def results_page() -> HTMLResponse:
    return _html("Results", BODY, JS, "results")


def _when(value: str | None, default: datetime) -> datetime:
    if not value:
        return default
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, "times must look like 2026-10-18T21:30") from exc
    return parsed if parsed.tzinfo else parsed.astimezone()


def _range(start: str | None, end: str | None) -> tuple[datetime, datetime]:
    now = datetime.now().astimezone()
    lo = _when(start, now - timedelta(days=1))
    hi = _when(end, now + timedelta(minutes=playgrounds.SESSION_MINUTES))
    # Whole sessions: widen both ends to the session they fall in.
    return playgrounds.session_bounds(lo)[0], playgrounds.session_bounds(hi)[1]


def _session_rows(start: str | None, end: str | None) -> tuple[datetime, datetime, list[dict]]:
    lo, hi = _range(start, end)
    conn = open_db()
    try:
        rows = playgrounds.sessions.sessions(conn, lo, hi)
    finally:
        conn.close()
    return lo, hi, rows


@router.get("/api/v1/sessions")
def sessions_json(start: str | None = None, end: str | None = None) -> dict:
    lo, hi, rows = _session_rows(start, end)
    columns = sorted({r["start"] for r in rows})
    return {
        "start": lo.isoformat(timespec="minutes"), "end": hi.isoformat(timespec="minutes"),
        "rows": rows, "sessions": columns, "columns": columns,
        "ends": {r["start"]: r["end"] for r in rows},
        "playgrounds": playgrounds.load(), "play_min": playgrounds.check.play_min(),
        "counting": playgrounds.COUNTING, "counting_short": playgrounds.COUNTING_SHORT,
        "rule": {"played": playgrounds.SESSION_PLAYED, "some": playgrounds.SESSION_SOME},
        "pictures": playgrounds.snapshot_usage(),
    }


@router.get("/api/v1/sessions/detail")
def session_detail(playground: str, start: str) -> dict:
    found = next((p for p in playgrounds.load() if p["id"] == playground), None)
    if found is None:
        raise HTTPException(404, f"no playground {playground!r}")
    lo, hi = playgrounds.session_bounds(_when(start, datetime.now().astimezone()))
    args = (playground, lo.isoformat(timespec="seconds"), hi.isoformat(timespec="seconds"))
    conn = open_db()
    try:
        checks = [dict(r) for r in conn.execute(
            """SELECT * FROM playground_checks WHERE playground_id = ? AND checked_at >= ?
               AND checked_at < ? ORDER BY checked_at""", args)]
        pics = [dict(r) for r in conn.execute(
            """SELECT c.camera_id, c.checked_at, c.people, c.picture_path, p.people AS total,
                      p.playing
               FROM camera_checks c JOIN playground_checks p
                 ON p.check_id = c.playground_check_id
               WHERE p.playground_id = ? AND p.checked_at >= ? AND p.checked_at < ?
                 AND c.picture_path IS NOT NULL ORDER BY c.checked_at, c.camera_id""", args)]
    finally:
        conn.close()
    saved_at = {p["checked_at"] for p in pics}
    for c in checks:
        c["saved"] = c["checked_at"] in saved_at
    folder = playgrounds.snapshots.SNAPSHOT_DIR
    pictures = [{"camera": p["camera_id"], "checked_at": p["checked_at"], "people": p["people"],
                 "total": p["total"], "playing": p["playing"],
                 "url": "/snapshots/" + p["picture_path"]}
                for p in pics if (folder / p["picture_path"]).is_file()]
    return {"playground": found, "start": lo.isoformat(timespec="minutes"),
            "end": hi.isoformat(timespec="minutes"), "checks": checks, "pictures": pictures}


@router.get("/snapshots/{path:path}", include_in_schema=False)
def snapshot(path: str) -> FileResponse:
    root = playgrounds.snapshots.SNAPSHOT_DIR.resolve()
    target = (root / Path(path)).resolve()
    if root not in target.parents or target.suffix != ".jpg" or not target.is_file():
        raise HTTPException(404, "no such picture")
    return FileResponse(target, media_type="image/jpeg")


@router.get("/api/v1/sessions.csv")
def sessions_csv(start: str | None = None, end: str | None = None) -> Response:
    lo, _, rows = _session_rows(start, end)
    buf = io.StringIO()
    fields = ["playground", "start", "end", "verdict", "avg_people", "max_people",
              "playing_share", "minutes_playing", "checks", "checks_without_picture",
              "saved_pictures", "assist_upgrades"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="sessions-{lo.date().isoformat()}.csv"'})
