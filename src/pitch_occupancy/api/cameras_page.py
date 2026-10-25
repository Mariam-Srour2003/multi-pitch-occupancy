"""Connecting cameras: the `/cameras` page and its API.

Four jobs on one page, in the order an installer meets them:

a) **Your cameras** - add, edit, delete and test the URLs in ``configs/cameras.json``
   (`pitch_occupancy.cameras`). Every URL this module returns is masked; an edit can leave the
   password out (or as ``***``) to keep the stored one, so the page never needs to show it.
b) **What is happening now?** - a spot check: one short burst per camera, through the
   detector-first pipeline the other pages share (`pipeline.shared`), each camera
   inside its stored boundary, the counts added across cameras and the rule applied once.
c) **Start recording a day** - turns "start 08:00, 90-minute slots, six of them, weekdays"
   into schedule entries and merges them into ``configs/slots_schedule.json`` through
   `api.schedule_editor.write_schedule` (validated by the real loader, atomic, history kept).
   The *recording* is the live worker's job, not this page's.
d) **How to connect a camera** - the guide, also in ``docs/cameras.md``.

**Inference runs inside the request, on purpose.** The spot check is on demand and small - one
burst (the rule's 3 frames, or one frame if the detector is unavailable) per camera - so a
background job would add a queue and a poll for no gain. The recorded history never comes from
here: it comes from the live schedule, and the page says so next to the button.

**The daily check** sits over the camera list: *Test all cameras* tries each saved camera in
turn with a short timeout (no pictures, so it stays quick), and the *Last picture* column says
when the live run last recorded one from each camera, read from the database. The two answer
different questions - "does it answer now?" and "has it been recording?" - and a camera can
fail either one alone.

Read routes: ``GET /cameras`` (the page), ``GET /api/v1/cameras/registry`` (masked list, with
each camera's last recorded picture; not ``/api/v1/cameras``, which the simulator owns) and
``GET /api/v1/cameras/health`` (the recording part alone) and ``GET /api/v1/recording/status``
(the live worker's heartbeat, `status.read_status`, behind the *Recording status* card at the
top of the page). Mutating routes:
``PUT /api/v1/cameras``, ``DELETE /api/v1/cameras/{venue}/{camera_id}``,
``POST /api/v1/cameras/test``, ``POST /api/v1/cameras/test-all`` (opens connections, writes
nothing), ``POST /api/v1/cameras/predict``, ``POST /api/v1/cameras/schedule/preview`` (writes
nothing) and ``PUT /api/v1/cameras/schedule``.
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pitch_occupancy import cameras
from pitch_occupancy.api import schedule_editor
from pitch_occupancy.api.layout import page
from pitch_occupancy.api.templating import template

__all__ = ["router", "plan_slots", "merge_schedule"]

router = APIRouter(tags=["cameras"])

#: The schedule file this page writes. ``None`` means `schedule_editor.DEFAULT_SCHEDULE`,
#: resolved at call time so a test can point either at a temporary file.
SCHEDULE_PATH: Path | None = None
#: Per-camera connection timeout for the page's reads, in milliseconds.
TIMEOUT_MS = 5000
#: Per-camera timeout for "Test all cameras": short, because it tries every camera in turn.
TEST_ALL_TIMEOUT_MS = 3000
#: Width the page's snapshots are scaled to.
SNAPSHOT_WIDTH = 640

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

#: The states in the words the page uses.
LABELS = {
    "C1_EMPTY": "Empty",
    "C2_ACTIVE_PLAY": "Playing",
    "C3_MAINTENANCE_NON_SPORTING": "In use, not a game",
    "UNCERTAIN": "Not sure",
    "UNAVAILABLE": "Cannot judge right now",
}


def _schedule_path() -> Path:
    return SCHEDULE_PATH or schedule_editor.DEFAULT_SCHEDULE


def _db_path() -> Path:
    """The database the live run writes, as `api.routes.get_conn` opens it. Read at call time
    so a test can point it away; never created from here."""
    from pitch_occupancy.config import settings

    return Path(settings.db_path)


def _live_model() -> str:
    """The detector `configs/rules.json` names - what the spot check and the live run share."""
    from pitch_occupancy.config import settings

    try:
        return str(json.loads(settings.rules_path.read_text(encoding="utf-8"))["detector"])
    except (OSError, ValueError, KeyError):
        return "yolov8n"


def _commands() -> dict[str, str]:
    model = _live_model()
    base = f"uv run python -m pitch_occupancy.worker --source live --model {model}"
    return {
        "check": f"{base} --dry-run",
        "start": base,
        "with_images": f"{base} --evidence-dir data/evidence",
    }


def _results_info() -> list[str]:
    return [
        "Each finished slot is written to the database (data/db/pitch_monitor.db) and shows "
        "on the dashboard at /.",
        "Evidence pictures are OFF unless you start with --evidence-dir: they show "
        "identifiable people.",
        "Retention: sampled frames are kept 7 days and evidence pictures 365 days "
        "(`uv run pitch retention` applies it).",
        "The worker reads the schedule when it starts. If it is already running, stop it "
        "(Ctrl+C) and start it again to pick up a new schedule.",
    ]


# --- a) the registry ----------------------------------------------------------------------

class CameraIn(BaseModel):
    venue: str = Field(max_length=64)
    camera_id: str = Field(max_length=64)
    #: Leave the password blank or as *** to keep the stored one.
    url: str = Field(max_length=2000)


class TestIn(BaseModel):
    venue: str | None = Field(default=None, max_length=64)
    camera_id: str | None = Field(default=None, max_length=64)
    url: str | None = Field(default=None, max_length=2000)


def _registry_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@router.get("/api/v1/cameras/registry")
def list_cameras() -> dict:
    """Every configured camera, **URLs masked**, plus each camera's boundary status.

    Not ``GET /api/v1/cameras``: the simulator already serves that path (token-protected, its
    own recorded cameras), and whichever router is mounted first would shadow the other.
    """
    from pitch_occupancy.vision import roi

    try:
        rows = cameras.list_cameras()
    except ValueError as exc:
        return {"cameras": [], "problem": str(exc)}
    for row in rows:
        try:
            _, key = roi.resolve_with_key(row["camera_id"], venue=row["venue"])
        except Exception:  # noqa: BLE001 - a broken boundary file must not hide the cameras
            key = None
        row["boundary"] = key
    health = _health(rows)
    for row in rows:
        row["recording"] = health[(row["venue"], row["camera_id"])]
    return {"cameras": rows, "problem": None}


def _health(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Each camera's last recorded picture, plus whether a saved schedule uses it.

    ``scheduled`` is what makes an old picture a problem: a camera no slot uses is expected
    to have recorded nothing. A database or schedule that cannot be read is reported as
    "never recorded" / "not scheduled", never as an error that would hide the camera list.
    """
    keys = [(r["venue"], r["camera_id"]) for r in rows]
    try:
        found = cameras.recording_health(_db_path(), keys)
    except Exception:  # noqa: BLE001 - a broken database must not hide the cameras
        found = {k: {"last_recorded": None, "minutes_since": None, "captured_24h": 0,
                     "expected_24h": None, "note": "the database could not be read"}
                 for k in keys}
    scheduled = {(e.get("venue_id"), c) for e in _read_schedule_entries(quiet=True)
                 for c in (e.get("cameras") or [])}
    for key in keys:
        found[key]["scheduled"] = key in scheduled
        minutes = found[key]["minutes_since"]
        found[key]["stale"] = found[key]["scheduled"] and (minutes is None or minutes > 24 * 60)
    return found


@router.get("/api/v1/cameras/health")
def camera_health() -> dict:
    """When the live run last recorded a picture from each saved camera. Opens no camera."""
    try:
        rows = cameras.list_cameras()
    except ValueError as exc:
        return {"cameras": [], "problem": str(exc), "database_exists": _db_path().is_file()}
    health = _health(rows)
    return {"cameras": [{"venue": r["venue"], "camera_id": r["camera_id"],
                         **health[(r["venue"], r["camera_id"])]} for r in rows],
            "problem": None, "database_exists": _db_path().is_file()}


@router.put("/api/v1/cameras")
def save_camera(body: CameraIn) -> dict:
    """Add or replace one camera. Returns it masked; a blank password keeps the stored one."""
    try:
        saved = cameras.save_camera(body.venue, body.camera_id, body.url)
    except ValueError as exc:
        raise _registry_error(exc) from exc
    return {"ok": True, "camera": saved}


@router.delete("/api/v1/cameras/{venue}/{camera_id}")
def delete_camera(venue: str, camera_id: str) -> dict:
    """Remove one camera, and say which scheduled slots still name it."""
    try:
        cameras.delete_camera(venue, camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no camera {camera_id} at {venue}") from exc
    except ValueError as exc:
        raise _registry_error(exc) from exc
    still = [e for e in _read_schedule_entries(quiet=True)
             if e.get("venue_id") == venue and camera_id in (e.get("cameras") or [])]
    warning = None
    if still:
        warning = (f"{len(still)} scheduled slot(s) at {venue} still use {camera_id}; the live "
                   f"worker will refuse to start until the camera is added back or the slots "
                   f"are changed")
    return {"ok": True, "warning": warning}


def _resolve_test_url(body: TestIn) -> str:
    if body.url and body.url.strip():
        stored = None
        if body.venue and body.camera_id:
            try:
                stored = cameras.get_url(body.venue, body.camera_id)
            except (KeyError, ValueError):
                stored = None
        try:
            return cameras.merge_password(body.url, stored)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if body.venue and body.camera_id:
        try:
            return cameras.get_url(body.venue, body.camera_id)
        except KeyError as exc:
            raise HTTPException(404, f"no camera {body.camera_id} at {body.venue}") from exc
        except ValueError as exc:
            raise _registry_error(exc) from exc
    raise HTTPException(status_code=422, detail="give a saved camera (venue and camera) or a URL")


@router.post("/api/v1/cameras/test")
def test_camera(body: TestIn) -> dict:
    """Try one camera and say where it stopped. **Opens a real connection.**

    The response carries the masked URL only, the plain-English reason, the fix, and a
    snapshot scaled to ~640 px when a picture arrived. Nothing is saved.
    """
    url = _resolve_test_url(body)
    result = cameras.probe(url, timeout_ms=TIMEOUT_MS)
    out = result.to_dict()
    out["snapshot"] = cameras.jpeg_base64(result.frame, max_width=SNAPSHOT_WIDTH)
    out["tested_at"] = datetime.now().isoformat(timespec="seconds")
    return out


@router.post("/api/v1/cameras/test-all")
def test_all_cameras() -> dict:
    """Try every saved camera, one after another, with a short timeout. **Opens real
    connections**; returns masked URLs and plain reasons, no pictures, and saves nothing."""
    try:
        results = cameras.probe_all(timeout_ms=TEST_ALL_TIMEOUT_MS)
    except ValueError as exc:
        raise _registry_error(exc) from exc
    answered = sum(1 for r in results if r["ok"])
    n = len(results)
    summary = (f"{answered} of {n} camera{'s' if n != 1 else ''} answered" if n
               else "No cameras saved yet")
    return {"cameras": results, "answered": answered, "total": n, "summary": summary,
            "tested_at": datetime.now().isoformat(timespec="seconds")}


# --- b) the spot check --------------------------------------------------------------------

class PredictIn(BaseModel):
    venue: str = Field(max_length=64)
    #: Which of the venue's cameras; empty or missing means all of them.
    cameras: list[str] | None = None


def _pipeline():
    """The shared detector-first pipeline (`pipeline.shared`). A function for the tests."""
    from pitch_occupancy.pipeline import shared

    return shared()


def _open(url: str):
    # Looked up on the module at call time, so a test that replaces `cameras.open_capture`
    # replaces it here too.
    return cameras.open_capture(url, TIMEOUT_MS)


def _grab(urls: dict[str, str], n: int, spacing_s: float) -> dict[str, list]:
    """One burst per camera through `RTSPSource` - the live worker's own reader - so a camera
    that reads here reads the same way in the schedule."""
    from pitch_occupancy.frame_source import RTSPSource

    # max_late_s=inf: a spot check reads every camera as minute 0, one after another, and a
    # slow first camera must not make the next one count as "too late" and come back empty.
    source = RTSPSource(urls, minutes=1, timeout_ms=TIMEOUT_MS, retries=1, open_capture=_open,
                        max_late_s=float("inf"))
    return {cam: [f.image_bgr for f in source.read_burst(cam, 0, n=n, spacing_s=spacing_s)]
            for cam in sorted(urls)}


def _picture(frame, verdict, instances, polygon) -> str | None:
    """The frame with what was counted drawn on it; the plain frame if drawing fails."""
    try:
        from pitch_occupancy.vision import overlay

        drawn = overlay.draw_verdict(frame, verdict, instances=instances, polygon=polygon,
                                     trace=False)
    except Exception:  # noqa: BLE001 - a picture without the drawing beats no picture
        drawn = frame
    return cameras.jpeg_base64(drawn, max_width=SNAPSHOT_WIDTH)


def _state(value) -> str:
    return str(getattr(value, "value", value))


@router.post("/api/v1/cameras/predict")
def predict_now(body: PredictIn) -> dict:
    """A spot check of one pitch: one burst per camera, scored now, **nothing recorded**.

    Runs the detector in the request (an on-demand check, one burst - the rule's frames, or a
    single frame when the detector cannot load - per camera). Each camera is counted inside
    its stored boundary (`roi.resolve`), or over the whole frame when none is drawn, and the
    response says which. People are added across cameras and a ball seen by either counts.
    If the detector cannot load, the verdict is ``UNAVAILABLE`` with the reason, not a 500,
    and the snapshots are still returned.
    """
    from pitch_occupancy.vision import roi

    try:
        urls = cameras.venue_urls(body.venue)
    except ValueError as exc:
        raise _registry_error(exc) from exc
    if not urls:
        raise HTTPException(404, f"no cameras saved for venue {body.venue}")
    wanted = [c for c in (body.cameras or []) if c] or sorted(urls)
    unknown = [c for c in wanted if c not in urls]
    if unknown:
        raise HTTPException(422, f"venue {body.venue} has no camera(s) {', '.join(unknown)}")

    pipeline, unavailable = None, None
    try:
        pipeline = _pipeline()
    except Exception as exc:  # noqa: BLE001 - reported as a verdict, not a 500
        unavailable = f"the detector could not be loaded ({type(exc).__name__}: {exc})"[:300]

    n, spacing = pipeline.burst if pipeline is not None else (1, 0.0)
    frames = _grab({c: urls[c] for c in wanted}, n, spacing)

    per_camera: list[dict] = []
    observations = {}
    for cam in wanted:
        burst = frames.get(cam) or []
        row: dict = {"camera_id": cam, "readable": bool(burst), "frames": len(burst),
                     "snapshot": None, "people": None, "ball": None, "state": None,
                     "label": None, "confidence": None, "boundary": None, "trace": [],
                     "problem": None, "hint": None}
        try:
            polygon, key = roi.resolve_with_key(cam, venue=body.venue)
        except Exception:  # noqa: BLE001
            polygon, key = None, None
        row["boundary"] = key
        if not burst:
            why = cameras.probe(urls[cam], timeout_ms=TIMEOUT_MS)
            row["problem"], row["hint"] = why.message, why.hint
            per_camera.append(row)
            continue
        if pipeline is None:
            row["snapshot"] = cameras.jpeg_base64(burst[-1], max_width=SNAPSHOT_WIDTH)
            per_camera.append(row)
            continue
        try:
            obs = pipeline.observe_minute(cam, burst, None, polygon=polygon, venue=body.venue)
        except Exception as exc:  # noqa: BLE001 - one camera failing must not hide the other
            row["problem"] = f"this camera's picture could not be scored ({type(exc).__name__})"
            row["snapshot"] = cameras.jpeg_base64(burst[-1], max_width=SNAPSHOT_WIDTH)
            per_camera.append(row)
            continue
        observations[cam] = obs
        v = obs.verdict
        index = min(max(getattr(obs, "evidence_index", 0), 0), len(burst) - 1)
        row.update(
            snapshot=_picture(burst[index], v, getattr(obs, "instances", ()), polygon),
            people=v.people, ball=v.ball, state=_state(v.state),
            label=LABELS.get(_state(v.state), _state(v.state)),
            confidence=round(float(v.confidence), 2), trace=list(v.trace),
        )
        per_camera.append(row)

    read = [r["camera_id"] for r in per_camera if r["readable"]]
    missing = [r["camera_id"] for r in per_camera if not r["readable"]]
    notes: list[str] = []
    if missing and read:
        notes.append(f"Only {', '.join(read)} could be read - this verdict covers part of "
                     f"the pitch.")
    for r in per_camera:
        if r["readable"] and r["boundary"] is None:
            notes.append(f"{r['camera_id']} has no pitch boundary drawn yet, so the whole "
                         f"picture was counted (neighbouring pitches included). Draw one at "
                         f"/roi.")

    if unavailable is not None:
        pitch = {"state": "UNAVAILABLE", "label": LABELS["UNAVAILABLE"], "confidence": None,
                 "people": None, "ball": None, "trace": [unavailable]}
    elif not observations:
        pitch = {"state": "UNCERTAIN", "label": LABELS["UNCERTAIN"], "confidence": 0.0,
                 "people": None, "ball": None,
                 "trace": ["no camera could be read or scored"]}
    else:
        fused = pipeline.fuse(observations)
        pitch = {"state": _state(fused.state), "label": LABELS.get(_state(fused.state)),
                 "confidence": round(float(fused.confidence), 2),
                 "people": fused.people_inside, "ball": fused.ball_seen,
                 "trace": list(fused.trace)}

    return {
        "venue": body.venue,
        "taken_at": datetime.now().isoformat(timespec="seconds"),
        "model": getattr(pipeline, "model_key", None),
        "frames_per_camera": n,
        "pitch": pitch,
        "cameras": per_camera,
        "notes": notes,
        "recorded": False,
    }


# --- c) the schedule ----------------------------------------------------------------------

class ScheduleIn(BaseModel):
    venue_id: str = Field(max_length=64)
    field_id: str = Field(default="", max_length=64)
    cameras: list[str] = Field(default_factory=list)
    start: str = Field(max_length=5)
    slot_minutes: int
    #: Either how many slots, or the time the last one must end by.
    count: int | None = None
    end: str | None = Field(default=None, max_length=5)
    #: mon..sun; empty means every day.
    days: list[str] = Field(default_factory=list)


def _minutes(hhmm: str, what: str) -> int:
    try:
        hours, mins = hhmm.strip().split(":")
        h, m = int(hours), int(mins)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{what} {hhmm!r} is not a time like 08:00") from exc
    if h == 24 and m == 0:
        return 24 * 60
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"{what} {hhmm!r} is not a time like 08:00")
    return h * 60 + m


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def plan_slots(start: str, slot_minutes: int, *, count: int | None = None,
               end: str | None = None) -> tuple[list[tuple[int, int]], int]:
    """Back-to-back slots as ``(start, end)`` minutes since midnight, and the leftover minutes.

    Given ``count``, that many; given ``end``, as many whole slots as fit before it (the
    remainder is returned, not silently covered by a short slot). Refuses a day that runs
    past midnight: a slot's date is its start date, and one that crossed midnight would be
    filed under the wrong day.
    """
    begin = _minutes(start, "start time")
    if begin >= 24 * 60:
        raise ValueError("the start time must be before midnight")
    if not 5 <= slot_minutes <= 720:
        raise ValueError("a slot must be between 5 and 720 minutes long")
    leftover = 0
    if count is not None:
        if not 1 <= count <= 96:
            raise ValueError("the number of slots must be between 1 and 96")
        n = count
    elif end:
        stop = _minutes(end, "end time")
        if stop == 0:
            stop = 24 * 60  # "00:00" as an end time means midnight
        if stop <= begin:
            raise ValueError("the end time must be after the start time on the same day; "
                             "slots that cross midnight are not supported")
        n, leftover = divmod(stop - begin, slot_minutes)
        if n == 0:
            raise ValueError(f"no {slot_minutes}-minute slot fits between {start} and {end}")
    else:
        raise ValueError("give either the number of slots or an end time")
    last_end = begin + n * slot_minutes
    if last_end > 24 * 60:
        raise ValueError(
            f"{n} slots of {slot_minutes} minutes from {_hhmm(begin)} would end at "
            f"{_hhmm(last_end - 24 * 60)} the next day; slots that cross midnight are not "
            f"supported. Use fewer slots or an earlier start")
    return [(begin + i * slot_minutes, begin + (i + 1) * slot_minutes) for i in range(n)], leftover


def _field(entry: dict) -> str:
    return str(entry.get("field_id") or f"{entry.get('venue_id')}_field")


def _days(entry: dict) -> set[str]:
    return {str(d).lower()[:3] for d in entry.get("days") or []} or set(WEEKDAYS)


def _read_schedule_entries(*, quiet: bool = False) -> list[dict]:
    path = _schedule_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if quiet:
            return []
        raise ValueError(f"the current schedule file is not valid JSON (line {exc.lineno}); "
                         f"fix it before saving a new day") from exc
    if not isinstance(raw, list):
        if quiet:
            return []
        raise ValueError("the current schedule file does not hold a list of slots")
    return [e for e in raw if isinstance(e, dict)]


def merge_schedule(existing: list[dict], new: list[dict], venue: str,
                   field: str) -> tuple[list[dict], int, list[str]]:
    """Replace ``venue``+``field``'s entries with ``new``, keep the rest; report clashes.

    A clash is a kept slot at the same venue, on a shared day, whose time overlaps a new one
    and which either shares a camera (one camera cannot record two slots at once) or starts at
    the same minute (the slot's id is venue+date+start, so one would overwrite the other).
    """
    kept = [e for e in existing
            if not (e.get("venue_id") == venue and _field(e) == field)]
    replaced = len(existing) - len(kept)
    problems: list[str] = []
    for old in kept:
        if old.get("venue_id") != venue:
            continue
        try:
            o_start = _minutes(str(old.get("start")), "start")
            o_end = o_start + int(old.get("duration_minutes", 0))
        except (ValueError, TypeError):
            continue
        for slot in new:
            s_start = _minutes(slot["start"], "start")
            s_end = s_start + slot["duration_minutes"]
            if not _days(old) & _days(slot):
                continue
            same_start = s_start == o_start
            overlap = s_start < o_end and o_start < s_end
            shared = set(old.get("cameras") or []) & set(slot["cameras"])
            if same_start or (overlap and shared):
                why = ("starts at the same time" if same_start
                       else f"uses the same camera ({', '.join(sorted(shared))})")
                problems.append(
                    f"{slot['start']}-{_hhmm(s_end)} overlaps the existing slot "
                    f"{_hhmm(o_start)}-{_hhmm(o_end)} on {_field(old)}, which {why}")
    return kept + new, replaced, problems


def _plan(body: ScheduleIn) -> dict:
    """Everything preview and save share. Writes nothing."""
    from pitch_occupancy.vision import roi

    problems: list[str] = []
    try:
        venue = cameras.validate_id(body.venue_id, "venue id")
        field = cameras.validate_id(body.field_id, "field id") if body.field_id.strip() \
            else f"{venue}_field"
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    days = []
    for d in body.days:
        key = d.strip().lower()[:3]
        if key not in WEEKDAYS:
            raise HTTPException(422, f"unknown day {d!r}; use mon, tue, wed, thu, fri, sat, sun")
        if key not in days:
            days.append(key)
    days = [d for d in WEEKDAYS if d in days]
    try:
        windows, leftover = plan_slots(body.start, body.slot_minutes, count=body.count,
                                       end=body.end)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    try:
        configured = cameras.venue_urls(venue)
    except ValueError as exc:
        configured = {}
        problems.append(str(exc))
    chosen = [c for c in dict.fromkeys(c.strip() for c in body.cameras) if c] or \
        sorted(configured)
    if not chosen:
        problems.append(f"venue {venue} has no cameras yet - add them under 'Your cameras' "
                        f"first")
    missing = [c for c in chosen if c not in configured]
    if missing:
        problems.append(f"camera(s) {', '.join(missing)} are not saved for {venue}; the live "
                        f"worker would refuse to start. Add them under 'Your cameras' first")

    new = [{"venue_id": venue, "field_id": field, "start": _hhmm(a),
            "duration_minutes": b - a, "cameras": chosen, "days": days} for a, b in windows]
    try:
        existing = _read_schedule_entries()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    merged, replaced, clashes = merge_schedule(existing, new, venue, field)
    problems += clashes
    if not problems:
        problem = schedule_editor.validate(merged, _schedule_path())
        if problem:
            problems.append(problem)

    boundaries = {}
    for cam in chosen:
        try:
            boundaries[cam] = roi.resolve_with_key(cam, venue=venue)[1]
        except Exception:  # noqa: BLE001
            boundaries[cam] = None
    warnings = [f"{cam} has no pitch boundary yet: the live run scores it as 'not sure' "
                f"every minute until one is drawn at /roi (or start the worker with "
                f"--derive-roi)" for cam, key in boundaries.items() if key is None]
    if leftover:
        warnings.append(f"the last {leftover} minute(s) before the end time are not covered "
                        f"by a whole slot")

    return {
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "venue_id": venue,
        "field_id": field,
        "cameras": chosen,
        "days": days or list(WEEKDAYS),
        "slots": [{"start": _hhmm(a), "end": _hhmm(b), "minutes": b - a} for a, b in windows],
        "entries": new,
        "replaced": replaced,
        "kept": len(merged) - len(new),
        "total_after": len(merged),
        "boundaries": boundaries,
        "commands": _commands(),
        "results": _results_info(),
        "_merged": merged,
    }


@router.post("/api/v1/cameras/schedule/preview")
def preview_schedule(body: ScheduleIn) -> dict:
    """The slots a start time and slot length produce, and what saving would change.
    **Writes nothing.**"""
    out = _plan(body)
    out.pop("_merged")
    return out


@router.put("/api/v1/cameras/schedule")
def save_schedule(body: ScheduleIn) -> dict:
    """Merge the generated slots into the schedule: this venue+field's entries are replaced,
    every other entry is kept. Validated by the scheduler's own loader, written atomically,
    previous version kept (`schedule_editor.write_schedule`)."""
    out = _plan(body)
    merged = out.pop("_merged")
    if not out["ok"]:
        raise HTTPException(422, " ; ".join(out["problems"]))
    try:
        backup = schedule_editor.write_schedule(merged, _schedule_path())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    out["saved"] = True
    out["previous_version"] = backup.name
    return out


# --- recording status ---------------------------------------------------------------------

@router.get("/api/v1/recording/status")
def recording_status() -> dict:
    """Is the live worker running, and what is it recording? Read from its heartbeat file.

    ``status`` is `status.read_status()` - the worker's own document plus ``alive`` and
    ``age_s`` - or ``None`` when the worker has never been started on this computer, in which
    case the card shows ``start_command``. The file holds camera ids only, never URLs, so it
    is safe to pass through as it is.
    """
    from pitch_occupancy.status import read_status

    return {"status": read_status(), "start_command": _commands()["start"],
            "service_doc": "docs/service.md"}


# --- the page -----------------------------------------------------------------------------

def _guide() -> str:
    rows = "".join(
        f"<tr><td><b>{escape(msg)}</b></td><td>{escape(hint)}</td></tr>"
        for stage, (msg, hint) in cameras.STAGES.items() if stage != "ok")
    return GUIDE_HTML.replace("{TROUBLE_ROWS}", rows)


@router.get("/cameras", response_class=HTMLResponse)
def cameras_page() -> HTMLResponse:
    body = BODY_HTML.replace("{GUIDE}", _guide()).replace(
        "{CMD_START}", escape(_commands()["start"]))
    return HTMLResponse(page("Pitch Occupancy - Cameras", "Connect and check cameras", body,
                             active="cameras", css=CSS, script=SCRIPT))


CSS = template("cameras_page_css.css")

GUIDE_HTML = template("cameras_page_guide.html")

BODY_HTML = template("cameras_page_body.html")

SCRIPT = template("cameras_page_script.js")
