"""The simple front end: playgrounds first, few words, nothing technical on screen.

* ``/``                         Home - every playground as a card: how many people at the last
                                check, playing or not. Click one to see its cameras.
* ``/playground/<id>``          One playground: the total, and each camera's count and picture.
* ``/playgrounds``              Make and change playgrounds: which cameras, how often, which
                                hours, how the cameras are combined - with what this computer
                                can manage shown as you choose (`playgrounds.capacity`).
* ``/cameras/add``              Cameras - add (name, address, password), test, remove.
* ``/live/<place>/<camera>``    One camera on its own, refreshing itself.
* ``/results``                  For showing how it went: each 1.5-hour session's verdict, the
                                people over time, and the pictures saved (`results_page.py`).
* ``/records``                  The database: for each playground and each hour and a half,
                                were there people, the most at once, how long it was played on.
* ``/more``                     The original pages, all still there.

Everything here calls `playgrounds` or the APIs the original pages already use
(`cameras_page`); nothing about how a camera is read or counted differs between them.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from html import escape

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from pitch_occupancy import playgrounds
from pitch_occupancy.api.layout import page
from pitch_occupancy.api.templating import template
from pitch_occupancy.db.schema import open_db

__all__ = ["router"]

router = APIRouter()

#: Records are grouped into periods of this many minutes - the hour and a half the site asked
#: for. The API still accepts the other lengths in `playgrounds.PERIOD_CHOICES`.
RECORD_PERIOD = 90


# --- shared look and helpers ---------------------------------------------------------------

CSS = template("simple_pages_css.css")

#: JavaScript these pages share (after `layout.JS_COMMON`): a JSON fetch that surfaces the
#: server's reason, and the plain words for a number of people.
_JS = template("simple_pages_js.js")


def _html(title: str, body: str, script: str, active: str) -> HTMLResponse:
    return HTMLResponse(page(title, "Camera monitor", body, active=active, css=CSS,
                             script=_JS + script))


def _find(playground_id: str) -> dict:
    found = next((p for p in playgrounds.load() if p["id"] == playground_id), None)
    if found is None:
        raise HTTPException(404, f"no playground {playground_id!r}")
    return found


def _json_block(element_id: str, data: dict) -> str:
    """Data for a page's script, in a non-executable block with "<" escaped so a name can
    never close the tag. The page only ever sets text from it."""
    payload = json.dumps(data).replace("<", "\\u003c")
    return f'<script type="application/json" id="{element_id}">{payload}</script>\n'


# --- Home: every playground, one card each -------------------------------------------------

HOME_BODY = """
<div class="top"><h1>Playgrounds</h1>
  <a class="small" href="/playgrounds">Change playgrounds</a></div>
<p class="msg bad" id="runner" hidden></p>
<div class="cards" id="cards"><div class="empty">Loading…</div></div>
"""

HOME_JS = template("simple_pages_home_js.js")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    return _html("Playgrounds", HOME_BODY, HOME_JS, "home")


# --- One playground ---------------------------------------------------------------------------

PG_BODY = template("simple_pages_pg_body.html")

PG_JS = template("simple_pages_pg_js.js")


@router.get("/playground/{playground_id}", response_class=HTMLResponse,
            include_in_schema=False)
def playground_page(playground_id: str) -> HTMLResponse:
    p = _find(playground_id)
    data = {k: p[k] for k in ("id", "name", "counting", "every_seconds")}
    return _html(p["name"], _json_block("pg", data) + PG_BODY, PG_JS, "home")


# --- Playgrounds: make and change -------------------------------------------------------------

PGS_BODY = template("simple_pages_pgs_body.html")

PGS_JS = template("simple_pages_pgs_js.js")


@router.get("/playgrounds", response_class=HTMLResponse, include_in_schema=False)
def playgrounds_page() -> HTMLResponse:
    return _html("Playgrounds", PGS_BODY, PGS_JS, "playgrounds")


# --- Cameras: add, test, remove ---------------------------------------------------------------

CAMERAS_BODY = template("simple_pages_cameras_body.html")

CAMERAS_JS = template("simple_pages_cameras_js.js")


@router.get("/cameras/add", response_class=HTMLResponse, include_in_schema=False)
def cameras_simple() -> HTMLResponse:
    return _html("Cameras", CAMERAS_BODY, CAMERAS_JS, "cameras-simple")


# --- One camera, live -------------------------------------------------------------------------

LIVE_BODY = template("simple_pages_live_body.html")

LIVE_JS = template("simple_pages_live_js.js")


@router.get("/live/{venue}/{camera}", response_class=HTMLResponse, include_in_schema=False)
def live_camera(venue: str, camera: str) -> HTMLResponse:
    from pitch_occupancy import cameras

    try:
        known = {(c["venue"], c["camera_id"]) for c in cameras.list_cameras()}
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc
    if (venue, camera) not in known:
        raise HTTPException(404, f"no camera {camera!r} at {venue!r}")
    body = _json_block("cam", {"venue": venue, "camera": camera}) + LIVE_BODY
    return _html(camera, body, LIVE_JS, "cameras-simple")


@router.get("/cameras/test", include_in_schema=False)
@router.get("/live", include_in_schema=False)
def old_addresses() -> RedirectResponse:
    """The first version's separate Test and Live pages, folded into Cameras and Home."""
    return RedirectResponse("/", status_code=307)


# --- Records ------------------------------------------------------------------------------------

RECORDS_BODY = template("simple_pages_records_body.html")

RECORDS_JS = template("simple_pages_records_js.js").replace("__PERIOD__", str(RECORD_PERIOD))


@router.get("/records", response_class=HTMLResponse, include_in_schema=False)
def records_page() -> HTMLResponse:
    return _html("Records", RECORDS_BODY, RECORDS_JS, "records")


# --- More ---------------------------------------------------------------------------------------

MORE = (
    ("/client", "Pitch verdicts", "Used / not used / review for each booked slot."),
    ("/reports", "Reports", "Usage over days and weeks, busy hours."),
    ("/testing", "Testing with videos", "Upload a video or pictures and see the result."),
    ("/cameras", "Camera setup (advanced)", "Recording status and the slot schedule."),
    ("/bookings", "Bookings", "Check booking records against the cameras."),
    ("/roi", "Pitch boundaries", "Draw each camera's part of the playground, so people are "
                                 "counted once and only on the pitch."),
    ("/alerts", "Alerts", "Messages when a camera or the recorder stops."),
)


@router.get("/more", response_class=HTMLResponse, include_in_schema=False)
def more_page() -> HTMLResponse:
    links = "".join(f'<a href="{href}"><b>{escape(title)}</b><span>{escape(text)}</span></a>'
                    for href, title, text in MORE)
    body = f'<h1>More tools</h1><div class="card morelist">{links}</div>'
    return _html("More tools", body, "", "more")


# --- JSON: playgrounds --------------------------------------------------------------------------

class PlaygroundIn(BaseModel):
    name: str = Field(max_length=60)
    cameras: list[str] = Field(max_length=32)
    every_seconds: int
    counting: str = "add"
    hours: dict[str, str] | None = None
    save_every: int = 5


class EnabledIn(BaseModel):
    enabled: bool


def _saved_cameras() -> list[dict]:
    from pitch_occupancy import cameras

    try:
        return cameras.list_cameras()
    except ValueError:
        return []


@router.get("/api/v1/playgrounds")
def list_playgrounds() -> dict:
    from pitch_occupancy.alerts import playground_overdue

    items = playgrounds.load()
    runner = playgrounds.status()
    state = runner["playgrounds"]
    conn = open_db()
    try:
        latest = playgrounds.latest(conn)
    finally:
        conn.close()
    owner = {c: p for p in items for c in p["cameras"]}
    cams = []
    for c in _saved_cameras():
        ref = f"{c['venue']}/{c['camera_id']}"
        home = owner.get(ref, {})
        cams.append({"ref": ref, "venue": c["venue"], "camera_id": c["camera_id"],
                     "playground": home.get("id"), "playground_name": home.get("name")})
    return {
        "playgrounds": [{**p, "latest": latest.get(p["id"]), "status": state.get(p["id"], {}),
                         "overdue": playground_overdue(
                             p, (latest.get(p["id"]) or {}).get("checked_at")) is not None}
                        for p in items],
        "running": runner["running"],
        "cameras": cams,
        "capacity": playgrounds.capacity.capacity(items),
        "every_choices": list(playgrounds.EVERY_CHOICES),
        "counting_short": playgrounds.COUNTING_SHORT,
        "play_min": playgrounds.check.play_min(),
    }


def _saved(call):
    try:
        return call()
    except KeyError as exc:
        raise HTTPException(404, f"no playground {exc.args[0]!r}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/api/v1/playgrounds")
def create_playground(body: PlaygroundIn) -> dict:
    return _saved(lambda: playgrounds.save_playground(body.model_dump()))


@router.put("/api/v1/playgrounds/{playground_id}")
def update_playground(playground_id: str, body: PlaygroundIn) -> dict:
    current = _find(playground_id)
    values = {**body.model_dump(), "enabled": current["enabled"]}
    return _saved(lambda: playgrounds.save_playground(values, playground_id=playground_id))


@router.delete("/api/v1/playgrounds/{playground_id}")
def delete_playground(playground_id: str) -> dict:
    _saved(lambda: playgrounds.delete_playground(playground_id))
    return {"ok": True}


@router.post("/api/v1/playgrounds/{playground_id}/enabled")
def switch_playground(playground_id: str, body: EnabledIn) -> dict:
    return _saved(lambda: playgrounds.set_enabled(playground_id, body.enabled))


@router.post("/api/v1/playgrounds/{playground_id}/look")
def look_now(playground_id: str) -> dict:
    """Check the playground now, with each camera's picture. The result is stored like any
    other check; the pictures are returned and not kept."""
    record = playgrounds.check_playground(_find(playground_id), with_pictures=True)
    keep = ("camera", "readable", "people", "problem", "picture")
    return {**{k: v for k, v in record.items() if k != "cameras"},
            "cameras": [{k: c[k] for k in keep} for c in record["cameras"]]}


@router.get("/api/v1/playgrounds/{playground_id}/latest")
def playground_latest(playground_id: str) -> dict:
    p = _find(playground_id)
    conn = open_db()
    try:
        last = playgrounds.latest(conn).get(playground_id)
        rows: dict[str, dict] = {}
        if last is not None:
            rows = {f"{r['venue_id']}/{r['camera_id']}": dict(r) for r in conn.execute(
                "SELECT * FROM camera_checks WHERE playground_check_id = ?",
                (last["check_id"],))}
    finally:
        conn.close()
    return {
        "latest": last, "enabled": p["enabled"], "play_min": playgrounds.check.play_min(),
        "cameras": [{"camera": ref, "checked": ref in rows,
                     "readable": bool(rows.get(ref, {}).get("readable")),
                     "people": rows.get(ref, {}).get("people"),
                     "problem": rows.get(ref, {}).get("problem")} for ref in p["cameras"]],
    }


@router.post("/api/v1/computer-speed")
def computer_speed() -> dict:
    """Measure how fast this computer handles one picture, and save it for the capacity."""
    try:
        result = playgrounds.measure_speed()
    except Exception as exc:  # noqa: BLE001 - reported, not a 500
        raise HTTPException(503, f"The speed could not be measured "
                                 f"({type(exc).__name__}).") from exc
    return {"speed": result, "capacity": playgrounds.capacity.capacity()}


# --- JSON: records ------------------------------------------------------------------------

def _day(value: str | None) -> date:
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, "day must look like 2026-10-17") from exc


def _records(day: str | None, period: int, playground: str | None) -> tuple[date, list[dict]]:
    d = _day(day)
    conn = open_db()
    try:
        rows = playgrounds.records(conn, d, period_minutes=period, playground_id=playground)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        conn.close()
    return d, rows


@router.get("/api/v1/records")
def records_json(day: str | None = None, period: int = Query(RECORD_PERIOD),
                 playground: str | None = None) -> dict:
    d, rows = _records(day, period, playground)
    return {"day": d.isoformat(), "period_minutes": period, "rows": rows}


@router.get("/api/v1/records.csv")
def records_csv(day: str | None = None, period: int = Query(RECORD_PERIOD),
                playground: str | None = None) -> Response:
    d, rows = _records(day, period, playground)
    names = {p["id"]: p["name"] for p in playgrounds.load()}
    buf = io.StringIO()
    fields = ["day", "start", "end", "playground", "max_people", "avg_people", "playing",
              "minutes_playing", "checks", "checks_without_picture"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({**r, "playground": names.get(r["playground_id"], r["playground_id"]),
                         "playing": "yes" if r["playing_checks"] else "no"})
    return Response(buf.getvalue(), media_type="text/csv", headers={
        "Content-Disposition":
            f'attachment; filename="playground-records-{d.isoformat()}.csv"'})
