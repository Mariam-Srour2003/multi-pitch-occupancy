"""The bookings page: connect a booking source, check its data, compare it with the cameras.

Served at ``/bookings``, with its JSON under ``/api/v1/bookings/...``. Everything here
**reads**. The two POST routes take a request body (an uploaded CSV, or nothing) and store
nothing: ``/test`` reads the configured source and ``/check`` runs `booking_check` over rows,
the same shape as ``/schedule/validate`` - a POST because a CSV does not fit in a URL, not
because anything changes.

**There is deliberately no route that saves the source config.** It was considered (the
schedule editor shows how a config write can be made safe) and left out for two reasons. The
config holds a database password, and a web form that accepts one would also have to keep
history copies of it somewhere the repository ignores - which is a second secret file to
get wrong. And a page that can repoint the system at a different database is a page that
can make reconciliation compare against the wrong records. Editing
``configs/bookings_source.json`` by hand on the machine itself is the safer path, and the page
shows exactly what to write.

Credentials never leave the server: the configured source is shown through
`booking_sources.describe`, which masks passwords, and database errors are masked too.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import date, time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from pitch_occupancy.api.layout import CALL_JS, page
from pitch_occupancy.api.limits import read_body
from pitch_occupancy.api.routes import get_conn
from pitch_occupancy.api.templating import template
from pitch_occupancy.booking_check import check
from pitch_occupancy.booking_sources import (
    DEFAULT_SOURCE_CONFIG,
    describe,
    load_source,
    records_from_rows,
    rows_from_csv_text,
)
from pitch_occupancy.bookings import coverage, reconcile_slot
from pitch_occupancy.data.taxonomy import SlotStatus

__all__ = ["router", "bookings_html"]

router = APIRouter(tags=["bookings"])

#: The config file read on every request - module-level so a test can point it elsewhere.
SOURCE_CONFIG = DEFAULT_SOURCE_CONFIG
#: Largest CSV accepted by ``/check``. A year of hourly bookings on ten pitches is ~5 MB.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: What a failing source can raise. Each becomes a 422 with the message, never a 500 - the
#: request was understood, and the person needs to read why it could not be served.
_SOURCE_ERRORS = (ValueError, FileNotFoundError, RuntimeError, OSError, sqlite3.Error)


def _schedule():
    """The camera schedule, or None - the check runs without it rather than failing."""
    try:
        from pitch_occupancy.scheduler import load_schedule
        return load_schedule()
    except Exception:  # noqa: BLE001 - an unreadable schedule only drops one optional rule
        return None


def _configured():
    """``(source, None)``, ``(None, None)`` if none configured, or ``(None, problem)``."""
    try:
        return load_source(SOURCE_CONFIG), None
    except ValueError as exc:
        return None, str(exc)


def _require_source():
    source, problem = _configured()
    if problem:
        raise HTTPException(422, f"The booking source config has a problem: {problem}")
    if source is None:
        raise HTTPException(422, f"No booking source is configured. Create "
                                 f"configs/{SOURCE_CONFIG.name} (see the examples on the "
                                 f"Bookings page), or upload a CSV instead.")
    return source


def _record_dict(r) -> dict:
    return {"field_id": r.field_id, "date": r.date.isoformat(), "start": f"{r.start:%H:%M}",
            "end": f"{r.end:%H:%M}", "customer_ref": r.customer_ref, "status": r.status,
            "entered_by": r.entered_by}


# --- JSON ---------------------------------------------------------------------------------


@router.get("/api/v1/bookings/source")
def booking_source() -> dict:
    """The configured source with credentials masked, or ``{"configured": false}``."""
    source, problem = _configured()
    if problem:
        return {"configured": True, "valid": False, "problem": problem,
                "file": f"configs/{SOURCE_CONFIG.name}"}
    if source is None:
        return {"configured": False, "file": f"configs/{SOURCE_CONFIG.name}"}
    return {"configured": True, "valid": True, "file": f"configs/{SOURCE_CONFIG.name}",
            "source": describe(source.config)}


@router.post("/api/v1/bookings/test")
def try_source() -> dict:
    """Read the configured source once: row count, dates covered, and the first ten rows.

    Reads with the strict reader, so "connected" means "readable by reconciliation", not
    merely "the database answered".
    """
    source = _require_source()
    try:
        rows = source.rows()
    except _SOURCE_ERRORS as exc:
        raise HTTPException(422, f"Could not read the booking source: {exc}") from None
    try:
        records = records_from_rows(rows, source=source.source_tag)
    except ValueError as exc:
        raise HTTPException(
            422, f"Connected and read {len(rows)} row(s), but the data cannot be used as it "
                 f"is: {exc}. Use \"Check booking data\" to see every problem at once."
        ) from None
    span = coverage(records)
    return {
        "ok": True,
        "rows": len(records),
        "pitches": len({r.field_id for r in records}),
        "date_from": span[0].isoformat() if span else None,
        "date_to": span[1].isoformat() if span else None,
        "preview": [_record_dict(r) for r in records[:10]],
    }


@router.post("/api/v1/bookings/check")
async def check_bookings(request: Request) -> dict:
    """Check booking data. Body: CSV text to check that, or empty to check the configured source.

    Stores nothing. Returns the `booking_check` report.
    """
    # Streamed against the cap, never buffered whole first: a multi-gigabyte body sent
    # without a length would otherwise sit in memory before any check ran.
    body = await read_body(request, MAX_UPLOAD_BYTES, what="booking file")
    text = body.decode("utf-8-sig", errors="replace")
    if text.strip():
        try:
            rows = rows_from_csv_text(text)
        except (ValueError, csv.Error) as exc:
            raise HTTPException(422, f"Could not read the CSV: {exc}") from None
        what = "the uploaded CSV"
    else:
        source = _require_source()
        try:
            rows = source.rows()
        except _SOURCE_ERRORS as exc:
            raise HTTPException(422, f"Could not read the booking source: {exc}") from None
        what = f"the configured {source.kind} source"
    report = check(rows, schedule=_schedule())
    return {"checked": what} | report.to_dict()


@router.get("/api/v1/bookings/compare")
def compare(day: Annotated[date, Query(description="YYYY-MM-DD")],
            conn: Annotated[sqlite3.Connection, Depends(get_conn)]) -> dict:
    """What the cameras decided for each slot on ``day``, beside what the bookings say.

    Uses `bookings.reconcile_slot`, so a day the booking data does not reach is reported as
    needing review rather than as a page of unbooked usage.
    """
    slots = [dict(r) for r in conn.execute(
        """SELECT rs.slot_id, rs.field_id, rs.start_time, rs.end_time,
                  se.status, se.mean_confidence, se.is_overridden, se.override_status
           FROM rental_slots rs
           LEFT JOIN slot_evaluations se ON se.slot_id = rs.slot_id
           WHERE rs.slot_date = ?
           ORDER BY rs.field_id, rs.start_time""",
        (day.isoformat(),),
    ).fetchall()]

    source, problem = _configured()
    out: dict = {"day": day.isoformat(), "configured": source is not None, "slots": [],
                 "unmatched_bookings": [], "message": ""}
    if problem:
        raise HTTPException(422, f"The booking source config has a problem: {problem}")
    records = []
    if source is not None:
        try:
            records = source.read()
        except _SOURCE_ERRORS as exc:
            raise HTTPException(422, f"Could not read the booking source: {exc}") from None
    span = coverage(records)
    out["covered"] = bool(span) and span[0] <= day <= span[1]

    todays = [r for r in records if r.date == day]
    matched: set[tuple[str, time]] = set()
    for s in slots:
        start = time.fromisoformat(s["start_time"])
        booking = next((r for r in todays
                        if r.field_id == s["field_id"] and r.start == start), None)
        if booking:
            matched.add((booking.field_id, booking.start))
        verdict = (s["override_status"] if s["is_overridden"] and s["override_status"]
                   else s["status"])
        row = {
            "slot_id": s["slot_id"], "field_id": s["field_id"], "start": s["start_time"],
            "end": s["end_time"], "camera": verdict,
            "corrected": bool(s["is_overridden"]),
            "booking": booking.status if booking else None,
            "customer_ref": booking.customer_ref if booking else None,
            "anomaly": None, "severity": None, "explanation": "",
        }
        if verdict is None:
            row["explanation"] = "The cameras have not produced a verdict for this slot yet."
        elif source is not None:
            rec = reconcile_slot(records, field_id=s["field_id"], day=day, start=start,
                                 vision_status=SlotStatus(verdict))
            row.update(anomaly=str(rec.anomaly), severity=str(rec.severity),
                       explanation=rec.explanation)
        out["slots"].append(row)

    out["unmatched_bookings"] = [
        {k: v for k, v in _record_dict(r).items() if k != "entered_by"}
        for r in todays if (r.field_id, r.start) not in matched
    ]
    if not slots:
        out["message"] = ("The database has no camera slots for this day - the cameras were "
                          "not scheduled, or the worker has not run for it.")
    elif source is None:
        out["message"] = "No booking source is configured, so only the camera verdicts are shown."
    elif not out["covered"]:
        out["message"] = ("The booking data does not reach this day, so no slot is flagged as "
                          "unbooked - an absent booking cannot be told from an absent record.")
    return out


# --- the page -----------------------------------------------------------------------------

_EXAMPLES = {
    "CSV file": """{
  "type": "csv",
  "path": "exports/bookings.csv"
}""",
    "SQLite": """{
  "type": "sqlite",
  "path": "C:/BookingApp/data/bookings.db",
  "query": "SELECT pitch, starts_at, ends_at, ref, state FROM bookings",
  "columns": {"field_id": "pitch", "start": "starts_at", "end": "ends_at",
              "customer_ref": "ref", "status": "state"},
  "status_map": {"Booked": "confirmed", "Cancelled": "cancelled", "DNA": "no_show"}
}""",
    "PostgreSQL": """{
  "type": "sql",
  "url": "postgresql://pitch_reader:PASSWORD@db-host:5432/bookings",
  "query": "SELECT * FROM public.bookings WHERE booking_date >= CURRENT_DATE - 90",
  "columns": {"field_id": "pitch_code", "date": "booking_date", "start": "start_time",
              "end": "end_time", "customer_ref": "reference"}
}""",
    "MySQL / MariaDB": """{
  "type": "sql",
  "url": "mysql+pymysql://pitch_reader:PASSWORD@db-host:3306/bookings",
  "query": "SELECT * FROM bookings",
  "columns": {"field_id": "court", "start": "start_at", "end": "end_at",
              "customer_ref": "booking_no", "status": "state"}
}""",
    "SQL Server": """{
  "type": "sql",
  "url": "mssql+pyodbc://reader:PASSWORD@db-host/Bookings?driver=ODBC+Driver+18+for+SQL+Server",
  "query": "SELECT PitchId, StartsAt, EndsAt, BookingRef, Status FROM dbo.Bookings",
  "columns": {"field_id": "PitchId", "start": "StartsAt", "end": "EndsAt",
              "customer_ref": "BookingRef", "status": "Status"}
}""",
}


def _examples_html() -> str:
    from html import escape
    parts = []
    for i, (label, text) in enumerate(_EXAMPLES.items()):
        parts.append(
            f'<details{" open" if i == 0 else ""}><summary>{escape(label)}</summary>'
            f'<pre id="ex{i}">{escape(text)}</pre>'
            f'<button class="ghost" data-copy="ex{i}">Copy</button></details>')
    return "".join(parts)


_BODY = template("bookings_page_body.html")

_CSS = template("bookings_page_css.css")

_SCRIPT = template("bookings_page_script.js")

def bookings_html() -> str:
    """The whole page. Built per request: it is small, and `layout` is shared and may change."""
    return page("Pitch Occupancy - Bookings", "Booking data",
                _BODY.replace("EXAMPLES", _examples_html()),
                active="bookings", css=_CSS, script=CALL_JS + _SCRIPT)


@router.get("/bookings", response_class=HTMLResponse, include_in_schema=False)
def bookings_page() -> HTMLResponse:
    return HTMLResponse(bookings_html())
