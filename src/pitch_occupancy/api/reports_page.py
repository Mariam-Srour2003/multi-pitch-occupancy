"""The Reports section: what the live system recorded, added up for the client.

Three reports, each a JSON endpoint with a CSV twin (``format=csv``), and one page that
draws them:

* ``GET /api/v1/reports/summary`` - per pitch per day: slots used / not used / to review,
  hours played, utilisation, no-shows and unbooked play (`reports.daily_summary`).
* ``GET /api/v1/reports/busy-hours`` - weekday x hour share of watched time that was
  played (`reports.busy_hours`). Unwatched hours are "not observed", never 0%.
* ``GET /api/v1/reports/day`` - one recorded day cut into the client's own slot length,
  in the same table the video booking-day test gives (`reports.recorded_day`).
* ``GET /api/v1/reports/playgrounds`` - the playground checks, per playground per day or week:
  sessions by verdict, minutes playing, checks without a picture, second-opinion upgrades
  (`reports.playground_summary`).

Everything here only reads the database; there is no POST. The connection comes from
`routes.get_conn`, so a test overrides one dependency for the whole API.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date, time, timedelta
from html import escape
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from pitch_occupancy import playgrounds, reports
from pitch_occupancy.api.layout import page
from pitch_occupancy.api.routes import get_conn
from pitch_occupancy.api.templating import template

__all__ = ["router"]

router = APIRouter(tags=["reports"])

Conn = Annotated[sqlite3.Connection, Depends(get_conn)]

_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_MAX_RANGE_DAYS = 366


def _range(start: date | None, end: date | None) -> tuple[date, date]:
    end = end or date.today()
    start = start or end - timedelta(days=6)
    if start > end:
        raise HTTPException(422, "The start date must be on or before the end date.")
    if (end - start).days >= _MAX_RANGE_DAYS:
        raise HTTPException(422, f"Choose a range of at most {_MAX_RANGE_DAYS} days.")
    return start, end


def _csv(body: str, filename: str) -> PlainTextResponse:
    return PlainTextResponse(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{filename}"'})


def _field(field_id: str | None) -> str | None:
    value = (field_id or "").strip()
    return value or None


def _span(start: date, end: date, field_id: str | None) -> str:
    suffix = f"-{re.sub(r'[^A-Za-z0-9_-]', '_', field_id)}" if field_id else ""
    return f"{start.isoformat()}_{end.isoformat()}{suffix}"


# --- API --------------------------------------------------------------------------------------


@router.get("/api/v1/reports/summary", response_model=None)
def summary_api(
    conn: Conn,
    start: date | None = None,
    end: date | None = None,
    field_id: str | None = Query(None, max_length=120),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> dict | PlainTextResponse:
    """Per pitch per day between ``start`` and ``end`` (default: the last seven days)."""
    s, e = _range(start, end)
    fid = _field(field_id)
    data = reports.daily_summary(conn, start=s, end=e, field_id=fid)
    if format == "csv":
        return _csv(reports.summary_to_csv(data), f"pitch-summary_{_span(s, e, fid)}.csv")
    return data


@router.get("/api/v1/reports/busy-hours", response_model=None)
def busy_hours_api(
    conn: Conn,
    start: date | None = None,
    end: date | None = None,
    field_id: str | None = Query(None, max_length=120),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> dict | PlainTextResponse:
    """Weekday x hour playing share; ``share`` is null where nothing was observed."""
    s, e = _range(start, end)
    fid = _field(field_id)
    data = reports.busy_hours(conn, start=s, end=e, field_id=fid)
    if format == "csv":
        return _csv(reports.busy_hours_to_csv(data), f"busy-hours_{_span(s, e, fid)}.csv")
    return data


@router.get("/api/v1/reports/day", response_model=None)
def day_api(
    conn: Conn,
    day: date,
    field_id: Annotated[str, Query(min_length=1, max_length=120)],
    first_slot: str = Query("08:00", max_length=8),
    slot_minutes: int = Query(90, ge=1, le=600),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> dict | PlainTextResponse:
    """One pitch's recorded day cut into ``slot_minutes`` slots from ``first_slot``."""
    match = _HHMM.match(first_slot.strip())
    if not match:
        raise HTTPException(422, f"first_slot must be a clock time like 08:00, "
                                 f"not {first_slot!r}")
    first = time(int(match.group(1)), int(match.group(2)))
    fid = field_id.strip()
    rec = reports.recorded_day(conn, field_id=fid, day=day, first_slot=first,
                               slot_minutes=slot_minutes)
    if format == "csv":
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", fid)
        return _csv(reports.recorded_day_to_csv(rec),
                    f"recorded-day_{safe}_{day.isoformat()}_{first:%H%M}_{slot_minutes}min.csv")
    return rec.to_dict()


@router.get("/api/v1/reports/playgrounds", response_model=None)
def playgrounds_api(
    conn: Conn,
    start: date | None = None,
    end: date | None = None,
    playground_id: str | None = Query(None, max_length=120),
    group: str = Query("day", pattern="^(day|week)$"),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> dict | PlainTextResponse:
    """Playground sessions per playground per ``group`` (day or week, named by its Monday)."""
    s, e = _range(start, end)
    pid = _field(playground_id)
    data = reports.playground_summary(conn, start=s, end=e, playground_id=pid, group=group)
    if format == "csv":
        return _csv(reports.playground_summary_to_csv(data),
                    f"playground-sessions-by-{group}_{_span(s, e, pid)}.csv")
    return data


# --- the page ---------------------------------------------------------------------------------


_CSS = template("reports_page_css.css")

_BODY = template("reports_page_body.html")

_EMPTY = """<div class="card empty" id="nodata"><h3>No recordings yet</h3>
<p>Reports are built from what the cameras record. Start recording from the
<a href="/cameras">Cameras page</a>, or switch a playground on from the
<a href="/playgrounds">Playgrounds page</a>, and the reports will fill in as it is checked.</p>
</div>"""


def _exists(conn: sqlite3.Connection, sql: str) -> bool:
    """``sql`` (an EXISTS query) on a database that may predate the table it reads."""
    try:
        return bool(conn.execute(sql).fetchone()[0])
    except sqlite3.OperationalError:
        return False

_SCRIPT = template("reports_page_script.js")


@router.get("/reports", response_class=HTMLResponse, include_in_schema=False)
def reports_page(conn: Conn) -> HTMLResponse:
    fields = conn.execute(
        "SELECT field_id, name FROM fields ORDER BY name, field_id").fetchall()
    recorded = _exists(conn, "SELECT EXISTS(SELECT 1 FROM frame_samples) "
                             "OR EXISTS(SELECT 1 FROM slot_evaluations)")
    checked = _exists(conn, "SELECT EXISTS(SELECT 1 FROM playground_checks)")
    end = date.today()
    start = end - timedelta(days=6)
    options = "".join(
        f'<option value="{escape(f["field_id"])}">{escape(f["name"] or f["field_id"])}'
        "</option>" for f in fields)
    pg_options = "".join(f'<option value="{escape(p["id"])}">{escape(p["name"])}</option>'
                         for p in playgrounds.load())
    hide = ' class="hide"'
    body = (_BODY.replace("__EMPTY__", "" if recorded or checked else _EMPTY)
            .replace("__HIDE_ANY__", "" if recorded or checked else hide)
            .replace("__HIDE__", "" if recorded else hide)
            .replace("__HIDE_PG__", "" if checked else hide)
            .replace("__START__", start.isoformat()).replace("__END__", end.isoformat())
            .replace("__OPTIONS__", options).replace("__PG_OPTIONS__", pg_options))
    script = _SCRIPT if recorded or checked else ""
    return HTMLResponse(page("Pitch Occupancy - Reports", "Reports", body, active="reports",
                             css=_CSS, script=script))

