"""Connecting cameras: the `/cameras` page and its API.

Four jobs on one page, in the order an installer meets them:

a) **Your cameras** - add, edit, delete and test the URLs in ``configs/cameras.json``
   (`pitch_occupancy.cameras`). Every URL this module returns is masked; an edit can leave the
   password out (or as ``***``) to keep the stored one, so the page never needs to show it.
b) **What is happening now?** - a spot check: one short burst per camera, through the
   detector-first pipeline the two-camera reviewer uses (`clip_pair._pipeline`), each camera
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
``GET /api/v1/cameras/health`` (the recording part alone). Mutating routes:
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
    """The detector-first pipeline the two-camera reviewer uses. A function for the tests."""
    from pitch_occupancy.api.clip_pair import _pipeline as shared

    return shared()


def _open(url: str):
    # Looked up on the module at call time, so a test that replaces `cameras.open_capture`
    # replaces it here too.
    return cameras.open_capture(url, TIMEOUT_MS)


def _grab(urls: dict[str, str], n: int, spacing_s: float) -> dict[str, list]:
    """One burst per camera through `RTSPSource` - the live worker's own reader - so a camera
    that reads here reads the same way in the schedule."""
    from pitch_occupancy.frame_source import RTSPSource

    source = RTSPSource(urls, minutes=1, timeout_ms=TIMEOUT_MS, retries=1, open_capture=_open)
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


CSS = """
details.g{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  margin:0 0 8px;padding:0 16px}
details.g summary{cursor:pointer;font-weight:600;padding:12px 0;list-style-position:inside}
details.g[open] summary{border-bottom:1px solid var(--line);margin-bottom:10px}
details.g > div{padding:0 0 14px;font-size:14px;color:var(--ink-2)}
details.g table{min-width:0}
.snaps{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  margin-top:12px}
.snaps figure{margin:0;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;overflow:hidden}
.snaps img{display:block;width:100%;background:var(--surface-2)}
.snaps figcaption{padding:10px 13px;font-size:13px;color:var(--ink-2)}
.snaps figcaption b{color:var(--ink)}
.pill.UNCERTAIN,.pill.UNAVAILABLE{background:var(--surface-2);color:var(--ink-3)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.controls input.wide{width:min(520px,100%)}
.controls input[type=checkbox]{width:auto}
.days label{display:inline-flex;gap:4px;align-items:center;margin-right:10px;
  text-transform:none;letter-spacing:0;font:13px Archivo,sans-serif;color:var(--ink-2)}
.days input{width:auto}
.result{margin-top:12px}
td .ghost{margin-right:4px}
td.status{min-width:150px;font-size:13px}
td.status .hint{display:block;color:var(--ink-2);font-size:12px;max-width:340px}
td.last{white-space:nowrap;font-size:13px}
.big{font:700 22px Archivo,sans-serif}
"""

GUIDE_HTML = """
<details class="g"><summary>What you need before you start</summary><div>
<ul>
<li>The camera on the <b>same network</b> as the Mini-PC that runs this system (same switch,
router or VLAN), or reachable over a VPN.</li>
<li>The camera's <b>IP address</b> (from its app, the router's device list, or the NVR).</li>
<li>A <b>username and password</b> that is allowed to view live video. A separate
"viewer" user is better than the admin account.</li>
<li><b>RTSP turned on</b> in the camera's settings (often under Network &gt; Advanced, or
"Integration protocol" / "ONVIF").</li>
</ul></div></details>

<details class="g"><summary>Which address to use for your camera</summary><div>
<p>Replace <code>user</code>, <code>pass</code> and <code>IP</code> with your own. If the
password contains <code>@</code>, <code>/</code>, <code>?</code> or <code>#</code>, write them
as <code>%40</code>, <code>%2F</code>, <code>%3F</code>, <code>%23</code>.</p>
<div class="scroll"><table>
<thead><tr><th>Camera</th><th>Address pattern</th><th>Notes</th></tr></thead><tbody>
<tr><td>Hikvision (and HiLook)</td>
<td><code>rtsp://user:pass@IP:554/Streaming/Channels/101</code></td>
<td>101 = main stream, <b>102 = sub-stream</b> (enough for this system)</td></tr>
<tr><td>Dahua, Amcrest</td>
<td><code>rtsp://user:pass@IP:554/cam/realmonitor?channel=1&amp;subtype=0</code></td>
<td>subtype=0 main, subtype=1 sub-stream</td></tr>
<tr><td>Axis</td><td><code>rtsp://user:pass@IP/axis-media/media.amp</code></td>
<td>add <code>?resolution=1280x720</code> to ask for a smaller picture</td></tr>
<tr><td>Reolink</td><td><code>rtsp://user:pass@IP:554/h264Preview_01_main</code></td>
<td><code>h264Preview_01_sub</code> for the sub-stream</td></tr>
<tr><td>Uniview</td><td><code>rtsp://user:pass@IP:554/unicast/c1/s0/live</code></td>
<td>s1 = sub-stream</td></tr>
<tr><td>TP-Link Tapo</td><td><code>rtsp://user:pass@IP:554/stream1</code></td>
<td>create a "camera account" in the Tapo app first; stream2 = sub-stream</td></tr>
<tr><td>Other ONVIF cameras</td><td>varies</td>
<td>use the free <b>ONVIF Device Manager</b> on a PC on the same network: it finds the camera
and shows its stream address</td></tr>
<tr><td>Through an NVR / DVR</td>
<td><code>rtsp://user:pass@NVR-IP:554/Streaming/Channels/301</code></td>
<td>one address per camera; on a Hikvision NVR channel 3 is <code>301</code> (main) /
<code>302</code> (sub). Dahua NVR: <code>channel=3</code></td></tr>
<tr><td>HTTP snapshot or MJPEG</td><td><code>http://user:pass@IP/path/to/snapshot.jpg</code></td>
<td>works when the camera offers no RTSP; use the Test button to make sure a picture
arrives</td></tr>
<tr><td>USB webcam (bench tests)</td><td><code>usb:0</code></td>
<td><code>usb:1</code> for a second webcam on the same PC</td></tr>
<tr><td>Cloud-only cameras (Ring, Nest, Arlo, most Wi-Fi "app" cameras)</td><td>-</td>
<td><b>Not supported directly.</b> They need RTSP/ONVIF switched on, or to be recorded
by a local NVR whose stream this system can read</td></tr>
</tbody></table></div></div></details>

<details class="g"><summary>Recommended camera settings</summary><div>
<ul>
<li><b>Sub-stream is enough.</b> One picture a minute is all this system takes;
720p or even 640x360 works and is lighter on the network.</li>
<li>Use <b>H.264</b> rather than H.265 (H.265+) if pictures fail to arrive.</li>
<li>Give each camera a <b>fixed IP</b> (or a DHCP reservation in the router) so the address
does not change after a power cut.</li>
<li>Keep the camera's <b>clock synced</b> (NTP) so its on-screen time matches the records.</li>
</ul></div></details>

<details class="g"><summary>Where to point the cameras</summary><div>
<ul>
<li>Two cameras per pitch, <b>each seeing about half</b>. A small overlap in the middle is
fine; a big one counts the same people twice.</li>
<li>Include the touchlines and goal areas; avoid looking straight into floodlights or the
low sun.</li>
<li>Once a camera works, <b>draw the pitch boundary</b> for it on the
<a href="/roi">Boundaries</a> page, so people on the next pitch are not counted.</li>
</ul></div></details>

<details class="g"><summary>Checking the network</summary><div>
<ul>
<li>Check the address in <b>VLC</b>: Media &gt; Open Network Stream, paste the address.
If VLC shows video, this system can read it.</li>
<li>Firewalls: the Mini-PC must reach the camera on its RTSP port (usually
<b>554</b>).</li>
<li>Remote sites: connect them over a <b>VPN</b>. <b>Never</b> forward the RTSP port to the
internet - anyone could watch the pitch and try the password.</li>
</ul></div></details>

<details class="g"><summary>When the test fails</summary><div>
<div class="scroll"><table><thead><tr><th>What the test says</th><th>What to do</th></tr>
</thead><tbody>{TROUBLE_ROWS}</tbody></table></div>
<p>If a camera stops working during a recorded day, the slot is recorded as partly or not
observed rather than guessed - see the runbook, <code>docs/runbook.md</code>
("Camera offline").</p>
</div></details>

<details class="g"><summary>Privacy</summary><div>
<ul>
<li>The cameras film <b>identifiable people</b>. Put up signs and follow your local rules
for CCTV.</li>
<li>Camera addresses and passwords stay in <code>configs/cameras.json</code> on this
computer only; this page never shows a saved password.</li>
<li>Pictures are not stored unless the live worker is started with
<code>--evidence-dir</code>; this page's snapshots are not saved at all.</li>
</ul></div></details>
"""

BODY_HTML = """
<h2>Your cameras</h2>
<p class="sub2">Each pitch has two cameras, each seeing about half of it. Add each camera's
address here and press <b>Test</b> to check it sends a picture.</p>
<div id="camproblem" class="note err" hidden></div>
<div class="row" style="margin:0 0 10px">
  <button class="act" id="testall">Test all cameras</button>
  <span id="testallsum" class="muted">Checks each camera answers now (a few seconds each).
  "Last picture" is when the recording last saved one.</span>
</div>
<div class="scroll"><table>
<thead><tr><th>Venue</th><th>Camera</th><th>Address (password hidden)</th>
<th>Boundary</th><th>Status now</th><th>Last picture</th><th></th></tr></thead>
<tbody id="camrows"><tr><td colspan="7" class="muted">Loading...</td></tr></tbody>
</table></div>

<div class="card" style="margin-top:14px">
<h3 id="formtitle">Add a camera</h3>
<div class="controls">
  <div><label for="f_venue">Venue</label><input id="f_venue" placeholder="venue_01"></div>
  <div><label for="f_cam">Camera</label><input id="f_cam" placeholder="camera_A"></div>
  <div style="flex:1"><label for="f_url">Address</label>
    <input id="f_url" class="wide"
      placeholder="rtsp://user:password@192.168.1.64:554/Streaming/Channels/102"></div>
</div>
<p class="sub2" id="f_hint" style="margin-top:8px">When editing, the password shows as
<code>***</code>: leave it as it is to keep the saved password.</p>
<div class="row" style="margin-top:6px">
  <button class="act" id="f_save">Save camera</button>
  <button class="ghost" id="f_test">Test before saving</button>
  <button class="ghost" id="f_clear">Clear</button>
</div>
<div id="f_msg" class="result"></div>
</div>

<div class="card" style="margin-top:14px">
<h3>Test an address without saving it</h3>
<div class="controls">
  <div style="flex:1"><label for="t_url">Address</label>
    <input id="t_url" class="wide" placeholder="rtsp://user:password@IP:554/..."></div>
  <div><button class="ghost" id="t_go">Test</button></div>
</div>
</div>
<div id="testout" class="result"></div>

<h2>What is happening now?</h2>
<p class="sub2">Takes a picture from each camera of one pitch right now and says whether the
pitch is empty, being played on, or used for something else. People seen by both cameras are
added together; a ball seen by either camera counts.</p>
<div class="card">
<div class="controls">
  <div><label for="p_venue">Venue</label><select id="p_venue" style="width:180px"></select></div>
  <div><label>Cameras</label><div id="p_cams" class="days"></div></div>
  <div><button class="act" id="p_go">Check now</button></div>
  <div><label class="days"><input type="checkbox" id="p_auto">
    Repeat every minute</label></div>
</div>
<p class="note">This is a <b>spot check</b>: nothing here is recorded. The history on the
dashboard comes only from the live schedule below. A check takes a few seconds per camera.</p>
</div>
<div id="busy"><span class="spin"></span><span>Reading the cameras...</span></div>
<div id="predout" class="result"></div>

<h2>Start recording a day</h2>
<p class="sub2">Tell the system when the pitch's slots run. Every slot is watched one picture
per camera per minute, and the result (empty / playing / other use) is saved at the end of
the slot.</p>
<div class="card">
<div class="controls">
  <div><label for="s_venue">Venue</label><select id="s_venue" style="width:180px"></select></div>
  <div><label for="s_field">Pitch id</label><input id="s_field" placeholder="field_01"></div>
  <div><label>Cameras</label><div id="s_cams" class="days"></div></div>
</div>
<div class="controls">
  <div><label for="s_start">First slot starts</label><input id="s_start" type="time"
    value="08:00"></div>
  <div><label for="s_len">Slot length (min)</label><input id="s_len" type="number"
    min="5" max="720" value="90"></div>
  <div><label for="s_mode">Until</label><select id="s_mode" style="width:160px">
    <option value="count">a number of slots</option><option value="end">an end time</option>
  </select></div>
  <div id="s_countbox"><label for="s_count">Slots</label><input id="s_count" type="number"
    min="1" max="96" value="6"></div>
  <div id="s_endbox" hidden><label for="s_end">Last slot ends by</label><input id="s_end"
    type="time" value="23:00"></div>
</div>
<div class="controls"><div><label>Days</label><div class="days" id="s_days"></div></div></div>
<div class="row" style="margin-top:12px">
  <button class="ghost" id="s_preview">Preview slots</button>
  <button class="act" id="s_save" disabled>Save schedule</button>
</div>
<p class="sub2" style="margin-top:8px">Saving replaces this pitch's existing slots and keeps
every other pitch's. The previous schedule is kept as a backup.</p>
</div>
<div id="schedout" class="result"></div>

<h2>How to connect a camera</h2>
<p class="sub2">A step-by-step guide for whoever installs the cameras. The same guide is in
<code>docs/cameras.md</code>.</p>
{GUIDE}
"""

SCRIPT = r"""
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const DAYS = ['mon','tue','wed','thu','fri','sat','sun'];
let CAMS = [];
let STATUS = {};
const key = (c) => c.venue + '/' + c.camera_id;

function ago(minutes) {
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const h = Math.floor(minutes / 60);
  if (h < 24) return `${h} hour${h === 1 ? '' : 's'} ago`;
  const d = Math.floor(h / 24);
  return `${d} day${d === 1 ? '' : 's'} ago`;
}
function lastCell(rec) {
  if (!rec) return '<span class="muted">-</span>';
  const when = rec.last_recorded ? ago(rec.minutes_since) : 'never';
  const cls = rec.stale ? 'bad' : (rec.last_recorded ? '' : 'muted');
  let tip = rec.last_recorded ? rec.last_recorded.replace('T', ' ') : 'no picture recorded';
  if (rec.expected_24h) tip += ` - last 24 h: ${rec.captured_24h} of ${rec.expected_24h} ` +
    'minutes recorded';
  if (rec.note) tip += ` - ${rec.note}`;
  const extra = rec.expected_24h ? `<br><span class="muted">${rec.captured_24h} of ` +
    `${rec.expected_24h} min in 24 h</span>` : '';
  return `<span class="${cls}" title="${esc(tip)}">${esc(when)}</span>${extra}`;
}
function statusCell(c) {
  const r = STATUS[key(c)];
  if (!r) return '<span class="muted">not tested</span>';
  if (r.ok) return `<span class="ok">answered in ${r.elapsed_ms} ms</span>`;
  return `<span class="bad">${esc(r.message)}</span>` +
    (r.hint ? `<span class="hint">${esc(r.hint)}</span>` : '');
}

async function api(method, url, body) {
  const r = await fetch(url, {method, headers: {'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body)});
  let data = null;
  try { data = await r.json(); } catch (e) { data = null; }
  if (!r.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === 'string' ? d : (d ? JSON.stringify(d) : 'request failed'));
  }
  return data;
}

function venues() { return [...new Set(CAMS.map((c) => c.venue))].sort(); }

function camBoxes(target, venue) {
  const cams = CAMS.filter((c) => c.venue === venue);
  $(target).innerHTML = cams.length ? cams.map((c) =>
    `<label><input type="checkbox" value="${esc(c.camera_id)}" checked>` +
    `${esc(c.camera_id)}</label>`).join('') : '<span class="muted">no cameras</span>';
}
function checked(target) {
  return [...$(target).querySelectorAll('input:checked')].map((i) => i.value);
}

function fillVenues() {
  for (const [sel, boxes] of [['p_venue', 'p_cams'], ['s_venue', 's_cams']]) {
    const keep = $(sel).value;
    $(sel).innerHTML = venues().map((v) => `<option>${esc(v)}</option>`).join('');
    if (venues().includes(keep)) $(sel).value = keep;
    camBoxes(boxes, $(sel).value);
  }
}

async function loadCams() {
  const data = await api('GET', '/api/v1/cameras/registry');
  CAMS = data.cameras;
  $('camproblem').hidden = !data.problem;
  $('camproblem').textContent = data.problem || '';
  $('camrows').innerHTML = CAMS.length ? CAMS.map((c, i) => `<tr>
    <td>${esc(c.venue)}</td><td><b>${esc(c.camera_id)}</b></td>
    <td class="mono">${esc(c.url)}</td>
    <td>${c.boundary ? '<span class="ok">drawn</span>'
         : '<a href="/roi" class="bad">not drawn yet</a>'}</td>
    <td class="status">${statusCell(c)}</td><td class="last">${lastCell(c.recording)}</td>
    <td style="white-space:nowrap"><button class="ghost" data-test="${i}">Test</button>
    <button class="ghost" data-edit="${i}">Edit</button>
    <button class="ghost" data-del="${i}">Delete</button></td></tr>`).join('')
    : '<tr><td colspan="7" class="muted">No cameras yet - add the first one below.</td></tr>';
  fillVenues();
}

function showTest(r, where) {
  const img = r.snapshot ? `<img src="data:image/jpeg;base64,${r.snapshot}" alt="snapshot">` : '';
  const size = r.width ? ` - picture ${r.width}x${r.height}` : '';
  $(where).innerHTML = `<div class="snaps"><figure>${img}<figcaption>
    <div class="${r.ok ? 'ok' : 'bad'}"><b>${esc(r.message)}</b></div>
    <div class="mono">${esc(r.url)}</div>
    ${r.hint ? `<p class="note warn">${esc(r.hint)}</p>` : ''}
    <div class="muted">${esc((r.steps || []).join(' -> '))} (${r.elapsed_ms} ms${size})</div>
    </figcaption></figure></div>`;
}

async function runTest(body, where) {
  $(where).innerHTML = '<p class="muted">Connecting (up to ~10 seconds)...</p>';
  try { showTest(await api('POST', '/api/v1/cameras/test', body), where); }
  catch (e) { $(where).innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
}

$('camrows').addEventListener('click', async (ev) => {
  const b = ev.target.closest('button'); if (!b) return;
  const c = CAMS[+(b.dataset.test ?? b.dataset.edit ?? b.dataset.del)];
  if (b.dataset.test !== undefined) {
    runTest({venue: c.venue, camera_id: c.camera_id}, 'testout');
    $('testout').scrollIntoView({behavior: 'smooth', block: 'nearest'});
  } else if (b.dataset.edit !== undefined) {
    $('f_venue').value = c.venue; $('f_cam').value = c.camera_id; $('f_url').value = c.url;
    $('formtitle').textContent = `Edit ${c.venue} / ${c.camera_id}`;
    $('f_url').focus();
  } else if (b.dataset.del !== undefined) {
    if (!confirm(`Delete camera ${c.camera_id} at ${c.venue}?`)) return;
    try {
      const r = await api('DELETE',
        `/api/v1/cameras/${encodeURIComponent(c.venue)}/${encodeURIComponent(c.camera_id)}`);
      $('f_msg').innerHTML = r.warning ? `<p class="note warn">${esc(r.warning)}</p>` : '';
      await loadCams();
    } catch (e) { $('f_msg').innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
  }
});

function formBody() {
  return {venue: $('f_venue').value.trim(), camera_id: $('f_cam').value.trim(),
          url: $('f_url').value.trim()};
}
$('f_save').onclick = async () => {
  try {
    const r = await api('PUT', '/api/v1/cameras', formBody());
    $('f_msg').innerHTML = `<p class="note">Saved <b>${esc(r.camera.venue)} / ` +
      `${esc(r.camera.camera_id)}</b> as <span class="mono">${esc(r.camera.url)}</span>. ` +
      `Press Test to check it.</p>`;
    $('f_url').value = r.camera.url;
    await loadCams();
  } catch (e) { $('f_msg').innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
};
$('f_test').onclick = () => runTest(formBody(), 'f_msg');
$('f_clear').onclick = () => {
  for (const id of ['f_venue', 'f_cam', 'f_url']) $(id).value = '';
  $('formtitle').textContent = 'Add a camera'; $('f_msg').innerHTML = '';
};
$('t_go').onclick = () => runTest({url: $('t_url').value.trim()}, 'testout');

$('testall').onclick = async () => {
  $('testall').disabled = true;
  $('testallsum').className = 'muted';
  $('testallsum').textContent = `Testing ${CAMS.length} camera(s), one after another...`;
  try {
    const r = await api('POST', '/api/v1/cameras/test-all');
    STATUS = {};
    for (const c of r.cameras) STATUS[key(c)] = c;
    $('testallsum').className = r.answered === r.total ? 'ok' : 'bad';
    $('testallsum').textContent = `${r.summary} (${r.tested_at.replace('T', ' ')}).`;
    await loadCams();
  } catch (e) {
    $('testallsum').className = 'bad'; $('testallsum').textContent = e.message;
  } finally { $('testall').disabled = false; }
};

// --- live check ---
let busy = false, timer = null;
const yesno = (v) => v === null || v === undefined ? '-' : (v ? 'yes' : 'no');
async function predict() {
  if (busy || !$('p_venue').value) return;
  busy = true; $('p_go').disabled = true; $('busy').classList.add('on');
  try {
    const r = await api('POST', '/api/v1/cameras/predict',
      {venue: $('p_venue').value, cameras: checked('p_cams')});
    const p = r.pitch;
    const cams = r.cameras.map((c) => `<figure>
      ${c.snapshot ? `<img src="data:image/jpeg;base64,${c.snapshot}" alt="${esc(c.camera_id)}">`
        : ''}
      <figcaption><b>${esc(c.camera_id)}</b>
      ${c.state ? ` <span class="pill ${esc(c.state)}">${esc(c.label)}</span>` : ''}<br>
      ${c.readable ? `People on its half: <b>${c.people ?? '-'}</b> &middot; ball seen: ` +
        `<b>${yesno(c.ball)}</b><br><span class="muted">${c.boundary ? 'counted inside the ' +
        'drawn boundary' : 'no boundary drawn - whole picture counted'}</span>`
        : `<span class="bad">${esc(c.problem || 'no picture')}</span>` +
          (c.hint ? `<p class="note warn">${esc(c.hint)}</p>` : '')}
      ${c.readable && c.problem ? `<div class="bad">${esc(c.problem)}</div>` : ''}
      </figcaption></figure>`).join('');
    $('predout').innerHTML = `<div class="verdictbar">
      <span class="pill ${esc(p.state)}">${esc(p.label)}</span>
      <span>People on the pitch: <b>${p.people ?? '-'}</b></span>
      <span>Ball seen: <b>${yesno(p.ball)}</b></span>
      <span class="spacer muted">${esc(r.taken_at.replace('T', ' '))}</span></div>
      ${(r.notes || []).map((n) => `<p class="note warn">${esc(n)}</p>`).join('')}
      <div class="snaps">${cams}</div>
      <details class="g" style="margin-top:12px"><summary>Why this answer</summary><div>
      <pre>${esc((p.trace || []).join('\n'))}</pre></div></details>`;
  } catch (e) {
    $('predout').innerHTML = `<p class="note err">${esc(e.message)}</p>`;
  } finally {
    busy = false; $('p_go').disabled = false; $('busy').classList.remove('on');
  }
}
$('p_go').onclick = predict;
$('p_venue').onchange = () => camBoxes('p_cams', $('p_venue').value);
$('p_auto').onchange = () => {
  clearInterval(timer); timer = null;
  if ($('p_auto').checked) { predict(); timer = setInterval(predict, 60000); }
};

// --- schedule ---
$('s_days').innerHTML = DAYS.map((d) =>
  `<label><input type="checkbox" value="${d}" checked>${d}</label>`).join('');
$('s_venue').onchange = () => { camBoxes('s_cams', $('s_venue').value); stale(); };
$('s_mode').onchange = () => {
  $('s_countbox').hidden = $('s_mode').value !== 'count';
  $('s_endbox').hidden = $('s_mode').value !== 'end'; stale();
};
function stale() { $('s_save').disabled = true; }
for (const id of ['s_field', 's_start', 's_len', 's_count', 's_end', 's_days', 's_cams'])
  $(id).addEventListener('input', stale);
$('s_days').addEventListener('change', stale); $('s_cams').addEventListener('change', stale);

function schedBody() {
  const b = {venue_id: $('s_venue').value, field_id: $('s_field').value.trim(),
    cameras: checked('s_cams'), start: $('s_start').value, slot_minutes: +$('s_len').value,
    days: checked('s_days')};
  if ($('s_mode').value === 'count') b.count = +$('s_count').value; else b.end = $('s_end').value;
  return b;
}
function showPlan(r, saved) {
  const rows = r.slots.map((s, i) => `<tr><td class="num">${i + 1}</td>` +
    `<td class="mono">${esc(s.start)} - ${esc(s.end)}</td><td class="num">${s.minutes}</td></tr>`)
    .join('');
  const cmds = r.commands;
  $('schedout').innerHTML = `
    ${r.problems.map((p) => `<p class="note err">${esc(p)}</p>`).join('')}
    ${r.warnings.map((p) => `<p class="note warn">${esc(p)}</p>`).join('')}
    <p class="sub2" style="margin-top:12px"><b>${esc(r.venue_id)} / ${esc(r.field_id)}</b>,
      cameras ${esc(r.cameras.join(', '))}, on ${esc(r.days.join(' '))}.
      ${saved ? 'Saved' : 'Saving will'} replace ${r.replaced} existing slot(s) for this pitch
      and keep ${r.kept} other slot(s).</p>
    <div class="scroll"><table style="min-width:0"><thead><tr><th>#</th><th>Slot</th>
      <th>Minutes</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${saved ? `<div class="card" style="margin-top:14px"><h3 class="ok">Schedule saved</h3>
      <p>Now start the live recorder on the Mini-PC, in the project folder:</p>
      <p class="muted">1. Check everything is in place (opens no camera):</p>
      <pre>${esc(cmds.check)}</pre>
      <p class="muted">2. Start recording (it asks you to type <b>yes</b>):</p>
      <pre>${esc(cmds.start)}</pre>
      <p class="muted">Only if you need pictures kept as evidence:</p>
      <pre>${esc(cmds.with_images)}</pre>
      <ul>${r.results.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
      <p class="muted">Previous schedule kept as ${esc(r.previous_version)}.</p></div>` : ''}`;
}
$('s_preview').onclick = async () => {
  try {
    const r = await api('POST', '/api/v1/cameras/schedule/preview', schedBody());
    showPlan(r, false); $('s_save').disabled = !r.ok;
  } catch (e) { $('schedout').innerHTML = `<p class="note err">${esc(e.message)}</p>`; stale(); }
};
$('s_save').onclick = async () => {
  try { showPlan(await api('PUT', '/api/v1/cameras/schedule', schedBody()), true); stale(); }
  catch (e) { $('schedout').innerHTML = `<p class="note err">${esc(e.message)}</p>`; }
};

loadCams().catch((e) => {
  $('camrows').innerHTML = `<tr><td colspan="7" class="bad">${esc(e.message)}</td></tr>`;
});
"""
