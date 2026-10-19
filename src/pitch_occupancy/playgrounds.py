"""Playgrounds: groups of cameras that watch one pitch, checked together on their own timer.

A playground is what a person means by "the pitch": one camera on a small one, two or three on
a big one. Each playground has

* **its cameras** - a camera belongs to one playground at most;
* **how often** it is checked - every 5 seconds up to every hour;
* **its hours** - all day, or from-to (a playground used only in the evening is not checked
  in the morning);
* **how its cameras are combined** - and this is the part that decides whether one person is
  counted once:

  - ``add``: the cameras watch *different parts* of the playground. Each camera counts only
    the people inside its own drawn area (`vision/roi.py`, the Pitch boundaries page), and the
    counts are added: 3 on camera 1 and 5 on camera 2 are 8 people. A person stands in one
    area, so nobody is counted twice - **provided the drawn areas do not overlap**. Where they
    overlap, a person standing in both is counted by both; drawing the areas edge to edge is
    the fix, not anything the counting can repair afterwards.
  - ``largest``: the cameras watch *the same area* from different angles. Adding would count
    everyone twice, so the largest single camera's count is used instead.
  - ``average``: the cameras watch the same area and one of them sometimes misses people
    (glare, a post in the way); the counts are averaged, so one bad camera pulls the result
    only part of the way.
  - ``vote``: each camera decides on its own whether there is a game, and the playground is
    playing when at least half of its cameras say so. The count shown is the largest.

  Telling the same person apart in two cameras' pictures without areas (re-identification)
  would need the cameras calibrated to each other; with two cheap fixed cameras it is wrong
  often enough to be worse than either rule above, so it is not attempted.

A playground with ``play_min`` (5, from `configs/rules.json`) or more people is **playing**.

**Running them.** One background thread in the server keeps a clock per playground. When a
playground is due, all its cameras are opened at the same moment, one whole picture is taken
from each, people are counted, the counts combined and the result stored
(``playground_checks``, plus one ``camera_checks`` row per camera). Playgrounds run at the
same time as each other; counting is done one picture at a time, because the detector is the
part that uses the processor. A check still running when its next one falls due skips that
turn rather than queueing - a queue that never empties is how a busy computer falls hours
behind.

**How much this computer can do.** :func:`measure_speed` times the whole job for one picture -
opening a camera, decoding a whole frame, counting - using the processor time it costs, and
:func:`capacity` turns that into pictures per minute and compares it with what the saved
playgrounds ask for (each camera, once per check). Settings that would ask for more than the
computer can do are refused when saved.

Settings live in ``configs/playgrounds.json`` (gitignored, like the camera file).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

from pitch_occupancy.config import CONFIGS_DIR, settings

__all__ = [
    "CONFIG_PATH", "SPEED_PATH", "EVERY_CHOICES", "COUNTING", "load", "save_playground",
    "delete_playground", "set_enabled", "capacity", "measure_speed", "load_speed",
    "check_playground", "records", "latest", "start", "stop", "run_now", "status", "play_min",
    "every_text", "in_hours", "PERIOD_CHOICES", "COUNTING_SHORT", "combine", "sessions",
    "session_bounds", "snapshot_usage", "SNAPSHOT_DIR",
]

log = logging.getLogger(__name__)

CONFIG_PATH: Path = CONFIGS_DIR / "playgrounds.json"
SPEED_PATH: Path = CONFIGS_DIR / "computer_speed.json"

#: How often a playground may be checked, in seconds.
EVERY_CHOICES: tuple[int, ...] = (5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600)
#: How a playground's cameras are combined. See the module docstring.
COUNTING: dict[str, str] = {
    "add": "Cameras see different parts - add their counts",
    "largest": "Cameras see the same area - use the biggest count",
    "average": "Cameras see the same area - use the average count",
    "vote": "Each camera decides - playing when at least half say playing",
}
#: Short names of the counting modes, for tables.
COUNTING_SHORT: dict[str, str] = {
    "add": "counts added", "largest": "biggest count", "average": "average count",
    "vote": "cameras vote",
}
#: Saved pictures, per playground and day, for showing afterwards how the checks went.
SNAPSHOT_DIR: Path = settings.db_path.parent.parent / "snapshots"
#: Saved pictures never take more than this; the oldest are deleted to make room.
SNAPSHOT_CAP_BYTES: int = int(os.environ.get("PITCH_SNAPSHOT_CAP_MB", "1024")) * 1024 * 1024
#: Sessions: 1.5 hours each, on a grid that passes through 09:30 (..., 08:00, 09:30, 11:00, ...
#: 21:30, 23:00, 00:30, ...). Each session gets one verdict - see :func:`sessions`.
SESSION_MINUTES = 90
SESSION_ANCHOR = "09:30"
#: Share of a session's checks that must be playing for the session to count as played.
SESSION_PLAYED = 0.5
#: ... and below which, if any were playing, it was played only a little.
SESSION_SOME = 0.15
#: Periods the Records page groups checks into, in minutes; 90 is the default.
PERIOD_CHOICES: tuple[int, ...] = (15, 30, 60, 90, 120, 180)
#: Fraction of the measured speed the playgrounds may use. The rest is for everything else
#: the computer does - the web pages, the recorder, Windows - so the checks keep their time.
HEADROOM = 0.7
#: Used until the speed has been measured: a cautious guess for one picture on a laptop CPU.
DEFAULT_SECONDS_PER_PICTURE = 1.0
#: Longest wait for one camera's picture in a check, whatever the interval.
GRAB_TIMEOUT_S = 20.0
#: Camera streams open at the same time, across all playgrounds. Over a slow link (2026-10-18:
#: ~85 ms to the cameras) opening a dozen 1080p streams at once timed every one of them out,
#: while two at a time each delivered a picture in 5-25 s.
PARALLEL_GRABS = int(os.environ.get("PITCH_PARALLEL_GRABS", "2"))
_grab_slots = threading.BoundedSemaphore(max(1, PARALLEL_GRABS))

_lock = threading.RLock()
_detect_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now().astimezone()


def play_min() -> int:
    """People at or above which a playground counts as playing - the rule's own number."""
    try:
        from pitch_occupancy.vision.rules import RuleConfig

        return int(RuleConfig.load(settings.rules_path).play_min)
    except Exception:  # noqa: BLE001 - a broken rule file must not stop the counting
        return 5


# --- settings --------------------------------------------------------------------------------

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "playground"


def _read(path: Path | None = None) -> list[dict]:
    source = path or CONFIG_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = raw.get("playgrounds") if isinstance(raw, dict) else None
    return [p for p in items if isinstance(p, dict)] if isinstance(items, list) else []


def _write(items: list[dict], path: Path | None = None) -> None:
    target = path or CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".json.new")
    staging.write_text(json.dumps({"playgrounds": items}, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, target)


def load(path: Path | None = None) -> list[dict]:
    """Every saved playground, in the order they were made."""
    return [_normalise(p) for p in _read(path)]


def _normalise(p: dict) -> dict:
    hours = p.get("hours") if isinstance(p.get("hours"), dict) else None
    return {
        "id": str(p.get("id") or _slug(str(p.get("name", "")))),
        "name": str(p.get("name") or p.get("id") or "Playground"),
        "cameras": [str(c) for c in p.get("cameras") or []],
        "every_seconds": int(p.get("every_seconds") or 30),
        "counting": p.get("counting") if p.get("counting") in COUNTING else "add",
        "hours": hours,
        "enabled": bool(p.get("enabled", True)),
        "save_every": max(0, int(p.get("save_every", 5) or 0)),
    }


def _hhmm(value: str, what: str) -> dtime:
    try:
        return dtime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must look like 08:00") from exc


def _validate(p: dict, others: list[dict], saved_cameras: set[str]) -> dict:
    name = str(p.get("name") or "").strip()
    if not name:
        raise ValueError("Give the playground a name.")
    if len(name) > 60:
        raise ValueError("The name is too long (60 characters at most).")
    cams = [str(c) for c in p.get("cameras") or []]
    if not cams:
        raise ValueError("Choose at least one camera.")
    if len(set(cams)) != len(cams):
        raise ValueError("A camera is listed twice.")
    unknown = [c for c in cams if c not in saved_cameras]
    if unknown:
        raise ValueError(f"Unknown camera(s): {', '.join(unknown)}. Add them on the Cameras "
                         f"page first.")
    taken = {c: o["name"] for o in others for c in o["cameras"]}
    clash = [f"{c} (already in {taken[c]})" for c in cams if c in taken]
    if clash:
        raise ValueError("A camera can belong to one playground only: " + ", ".join(clash))
    every = p.get("every_seconds")
    if isinstance(every, bool) or every not in EVERY_CHOICES:
        raise ValueError("Choose how often from the list.")
    counting = p.get("counting", "add")
    if counting not in COUNTING:
        raise ValueError("Choose how the cameras are combined.")
    hours = p.get("hours")
    if hours:
        start, end = _hhmm(hours.get("from"), "From"), _hhmm(hours.get("to"), "To")
        if start == end:
            raise ValueError("From and To are the same time; choose All day instead.")
        hours = {"from": start.strftime("%H:%M"), "to": end.strftime("%H:%M")}
    else:
        hours = None
    save_every = p.get("save_every", 5)
    if (isinstance(save_every, bool) or not isinstance(save_every, int)
            or not 0 <= save_every <= 1000):
        raise ValueError("Save pictures must be a whole number: 0 (off), 1 (every check), "
                         "5 (one check in five)...")
    return {"name": name, "cameras": cams, "every_seconds": int(every), "counting": counting,
            "hours": hours, "enabled": bool(p.get("enabled", True)), "save_every": save_every}


def _saved_cameras() -> set[str]:
    from pitch_occupancy import cameras

    return {f"{c['venue']}/{c['camera_id']}" for c in cameras.list_cameras()}


def save_playground(values: dict, *, playground_id: str | None = None,
                    path: Path | None = None) -> dict:
    """Add a playground, or replace the one with ``playground_id``. Raises ``ValueError``
    with a sentence for the page - including when the computer could not keep up."""
    with _lock:
        items = load(path)
        others = [p for p in items if p["id"] != playground_id]
        if playground_id is not None and len(others) == len(items):
            raise KeyError(playground_id)
        clean = _validate(values, others, _saved_cameras())
        if playground_id is None:
            base = _slug(clean["name"])
            pid, n = base, 2
            while any(p["id"] == pid for p in items):
                pid, n = f"{base}-{n}", n + 1
        else:
            pid = playground_id
        new = {"id": pid, **clean}
        proposed = [new if p["id"] == pid else p for p in items]
        if playground_id is None:
            proposed.append(new)
        _check_capacity(proposed)
        _write(proposed, path)
    _wake.set()
    return new


def _check_capacity(proposed: list[dict]) -> None:
    cap = capacity(proposed)
    if cap["need_per_min"] > cap["can_per_min"]:
        raise ValueError(
            f"This computer can analyse about {cap['can_per_min']:.0f} pictures a minute, and "
            f"these settings ask for {cap['need_per_min']:.0f}. Check less often, use fewer "
            f"cameras, or switch a playground off.")


def set_enabled(playground_id: str, enabled: bool, path: Path | None = None) -> dict:
    with _lock:
        items = load(path)
        found = next((p for p in items if p["id"] == playground_id), None)
        if found is None:
            raise KeyError(playground_id)
        found["enabled"] = bool(enabled)
        if enabled:
            _check_capacity(items)
        _write(items, path)
    _wake.set()
    return found


def delete_playground(playground_id: str, path: Path | None = None) -> None:
    with _lock:
        items = load(path)
        rest = [p for p in items if p["id"] != playground_id]
        if len(rest) == len(items):
            raise KeyError(playground_id)
        _write(rest, path)
    _wake.set()


def in_hours(p: dict, now: datetime) -> bool:
    hours = p.get("hours")
    if not hours:
        return True
    start, end = dtime.fromisoformat(hours["from"]), dtime.fromisoformat(hours["to"])
    t = now.time().replace(tzinfo=None)
    return start <= t < end if start < end else (t >= start or t < end)  # past midnight


# --- how much the computer can do -----------------------------------------------------------

def load_speed(path: Path | None = None) -> dict | None:
    try:
        raw = json.loads((path or SPEED_PATH).read_text(encoding="utf-8"))
        return raw if float(raw.get("seconds_per_picture", 0)) > 0 else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def capacity(items: list[dict] | None = None, speed: dict | None = None) -> dict:
    """Pictures per minute the computer can analyse, and what the playgrounds ask for."""
    items = load() if items is None else items
    speed = speed if speed is not None else load_speed()
    per_picture = float(speed["seconds_per_picture"]) if speed else DEFAULT_SECONDS_PER_PICTURE
    can = 60.0 / per_picture * HEADROOM
    need = sum(len(p["cameras"]) * 60.0 / p["every_seconds"] for p in items if p["enabled"])
    examples = {every: int(can * every // 60) for every in (5, 10, 30, 60, 300)}
    return {
        "seconds_per_picture": round(per_picture, 3), "measured": speed is not None,
        "measured_at": speed.get("measured_at") if speed else None,
        "can_per_min": round(can, 1), "need_per_min": round(need, 1),
        "percent": round(100 * need / can) if can else 0,
        "cameras_at": examples,
    }


def _grab(url: str, timeout_s: float):
    """One whole picture from a camera: ``(frame | None, problem | None, wall_ms)``."""
    import cv2

    from pitch_occupancy import cameras
    from pitch_occupancy.frame_source import _read_whole

    started = time.monotonic()
    capture = cameras.open_capture(url, int(timeout_s * 1000))
    try:
        if not bool(getattr(capture, "isOpened", lambda: True)()):
            return None, "The camera did not answer.", int((time.monotonic() - started) * 1000)
        get = getattr(capture, "get", None)
        fps = float(get(cv2.CAP_PROP_FPS) or 25.0) if get is not None else 25.0
        ok, frame = _read_whole(capture, fps)
    finally:
        capture.release()
    wall = int((time.monotonic() - started) * 1000)
    if not ok or frame is None:
        return None, "No whole picture arrived in time.", wall
    return frame, None, wall


def _pipeline():
    from pitch_occupancy.api.clip_pair import _pipeline as shared

    return shared()


def _count(pipeline, venue: str, camera: str, frame) -> dict:
    """People inside the camera's drawn area (the whole picture when none is drawn)."""
    with _detect_lock:
        started = time.perf_counter()
        obs = pipeline.observe_minute(camera, [frame], None, venue=venue)
        took = time.perf_counter() - started
    v = obs.verdict
    return {"people": v.people, "state": str(getattr(v.state, "value", v.state)),
            "detect_ms": int(took * 1000),
            "instances": getattr(obs, "instances", ()), "verdict": v}


def measure_speed(path: Path | None = None, *, samples: int = 3) -> dict:
    """Time the whole job for one picture - open, decode, count - and save the result.

    Processor time rather than clock time, because waiting for a camera costs the computer
    nothing while decoding and counting cost it everything. Uses up to ``samples`` saved
    cameras that answer; if none does, counting alone on a stored picture, plus a margin for
    the decoding that could not be timed.
    """
    import numpy as np

    from pitch_occupancy import cameras

    pipeline = _pipeline()
    blank = np.zeros((1080, 1920, 3), np.uint8)
    with _detect_lock:
        pipeline.observe_minute("speed-test", [blank], None)  # warm the model up
    costs: list[float] = []
    for cam in cameras.list_cameras()[: samples * 3]:
        if len(costs) >= samples:
            break
        url = cameras.get_url(cam["venue"], cam["camera_id"])
        cpu = time.process_time()
        frame, _, _ = _grab(url, GRAB_TIMEOUT_S)
        if frame is None:
            continue
        _count(pipeline, cam["venue"], cam["camera_id"], frame)
        costs.append(time.process_time() - cpu)
    if costs:
        per_picture, how = sorted(costs)[len(costs) // 2], f"{len(costs)} camera(s)"
    else:
        cpu = time.process_time()
        with _detect_lock:
            pipeline.observe_minute("speed-test", [blank], None)
        per_picture, how = (time.process_time() - cpu) + 0.5, "counting only, no camera answered"
    result = {"seconds_per_picture": round(max(per_picture, 0.02), 3), "how": how,
              "measured_at": _now().isoformat(timespec="seconds")}
    target = path or SPEED_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


# --- one check of one playground -------------------------------------------------------------

def check_playground(p: dict, *, grab: Callable | None = None, pipeline=None,
                     db_path: Path | None = None, with_pictures: bool = False,
                     save: bool = False) -> dict:
    """Take one picture from every camera of ``p`` at the same moment, count, combine, store.

    Returns the stored playground row, with each camera's result under ``cameras`` - and
    each camera's picture (base64 JPEG, drawn) when ``with_pictures`` is set, for the live
    page. With ``save`` the drawn pictures are also kept under :data:`SNAPSHOT_DIR` (the
    runner sets it on one check in ``save_every``), within :data:`SNAPSHOT_CAP_BYTES`.
    """
    from pitch_occupancy import cameras
    from pitch_occupancy.db.schema import connect, initialise, transaction

    grab = grab or _grab
    started = time.monotonic()
    when = _now()
    timeout = min(GRAB_TIMEOUT_S, max(3.0, p["every_seconds"] * 0.4))
    targets = []
    for ref in p["cameras"]:
        venue, _, camera = ref.partition("/")
        try:
            targets.append((ref, venue, camera, cameras.get_url(venue, camera)))
        except (KeyError, ValueError):
            targets.append((ref, venue, camera, None))

    # All cameras at once: the pictures of one check should show the same moment.
    def take(target):
        ref, venue, camera, url = target
        if url is None:
            return None, "This camera is no longer saved.", 0
        try:
            with _grab_slots:
                return grab(url, timeout)
        except Exception as exc:  # noqa: BLE001 - one camera must not stop the others
            return None, f"Reading failed ({type(exc).__name__}).", 0

    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        grabbed = list(pool.map(take, targets))

    if pipeline is None and any(frame is not None for frame, _, _ in grabbed):
        pipeline = _pipeline()
    per_camera = []
    for (ref, venue, camera, _), (frame, problem, grab_ms) in zip(targets, grabbed, strict=True):
        row = {"camera": ref, "venue": venue, "camera_id": camera, "readable": frame is not None,
               "people": None, "state": None, "problem": problem, "grab_ms": grab_ms,
               "detect_ms": None, "picture": None, "picture_path": None}
        if frame is not None:
            try:
                counted = _count(pipeline, venue, camera, frame)
                row.update(people=counted["people"], state=counted["state"],
                           detect_ms=counted["detect_ms"])
                if with_pictures:
                    row["picture"] = _picture(frame, counted)
                if save:
                    row["picture_path"] = _save_snapshot(p["id"], camera, when, frame, counted)
            except Exception as exc:  # noqa: BLE001
                row.update(readable=False, problem=f"Counting failed ({type(exc).__name__}).")
        per_camera.append(row)

    counts = [r["people"] for r in per_camera if r["readable"] and r["people"] is not None]
    total, playing = combine(p["counting"], counts, play_min())
    duration = int((time.monotonic() - started) * 1000)
    record = {"playground_id": p["id"], "checked_at": when.isoformat(timespec="seconds"),
              "people": total, "playing": playing, "cameras_read": len(counts),
              "cameras_total": len(per_camera), "counting": p["counting"],
              "duration_ms": duration}

    conn = connect(db_path or settings.db_path)
    try:
        initialise(conn)
        with transaction(conn):
            cur = conn.execute(
                """INSERT INTO playground_checks (playground_id, checked_at, people, playing,
                       cameras_read, cameras_total, counting, duration_ms)
                   VALUES (:playground_id, :checked_at, :people, :playing, :cameras_read,
                           :cameras_total, :counting, :duration_ms)""", record)
            record["check_id"] = cur.lastrowid
            for r in per_camera:
                conn.execute(
                    """INSERT INTO camera_checks (venue_id, camera_id, checked_at, readable,
                           people, state, label, confidence, picture_path, problem,
                           playground_check_id, grab_ms, detect_ms)
                       VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)""",
                    (r["venue"], r["camera_id"], record["checked_at"], int(r["readable"]),
                     r["people"], r["state"], r["picture_path"], r["problem"],
                     record["check_id"], r["grab_ms"], r["detect_ms"]))
    finally:
        conn.close()
    record["cameras"] = per_camera
    return record


def combine(counting: str, counts: list[int], threshold: int) -> tuple[int | None, int | None]:
    """The playground's people and playing (1/0) from its cameras' counts; ``None`` without any."""
    if not counts:
        return None, None
    if counting == "largest":
        total = max(counts)
    elif counting == "average":
        total = round(sum(counts) / len(counts))
    elif counting == "vote":
        yes = sum(1 for n in counts if n >= threshold)
        return max(counts), int(2 * yes >= len(counts))
    else:
        total = sum(counts)
    return total, int(total >= threshold)


def _picture(frame, counted) -> str | None:
    from pitch_occupancy import cameras

    try:
        from pitch_occupancy.vision import overlay

        drawn = overlay.draw_verdict(frame, counted["verdict"], instances=counted["instances"],
                                     trace=False)
    except Exception:  # noqa: BLE001 - a picture without the drawing beats no picture
        drawn = frame
    return cameras.jpeg_base64(drawn, max_width=640)


# --- saved pictures ------------------------------------------------------------------------------

_snap_lock = threading.Lock()
_snap_bytes: int | None = None


def _snapshot_files() -> list[Path]:
    return [f for f in SNAPSHOT_DIR.rglob("*.jpg") if f.is_file()] if SNAPSHOT_DIR.exists() else []


def snapshot_usage() -> dict:
    """How much the saved pictures take, and the cap."""
    files = _snapshot_files()
    used = sum(f.stat().st_size for f in files)
    return {"files": len(files), "bytes": used, "cap_bytes": SNAPSHOT_CAP_BYTES,
            "folder": str(SNAPSHOT_DIR)}


def _save_snapshot(pid: str, camera: str, when: datetime, frame, counted) -> str | None:
    """Draw the count on the picture and keep it; returns the path relative to SNAPSHOT_DIR.

    The pictures stay under the cap: when a new one would pass it, the oldest are deleted
    first, so a long run keeps the most recent pictures instead of filling the disk.
    """
    global _snap_bytes
    import cv2

    try:
        from pitch_occupancy.vision import overlay

        drawn = overlay.draw_verdict(frame, counted["verdict"], instances=counted["instances"],
                                     trace=False)
    except Exception:  # noqa: BLE001 - a picture without the drawing beats no picture
        drawn = frame
    h, w = drawn.shape[:2]
    if w > 1280:
        drawn = cv2.resize(drawn, (1280, max(1, int(h * 1280 / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    data = buf.tobytes()
    rel = Path(pid) / when.strftime("%Y-%m-%d") / f"{when.strftime('%H%M%S')}_{_slug(camera)}.jpg"
    with _snap_lock:
        if _snap_bytes is None:
            _snap_bytes = sum(f.stat().st_size for f in _snapshot_files())
        if _snap_bytes + len(data) > SNAPSHOT_CAP_BYTES:
            for f in sorted(_snapshot_files(), key=lambda f: f.stat().st_mtime):
                if _snap_bytes + len(data) <= SNAPSHOT_CAP_BYTES * 0.95:
                    break
                try:
                    size = f.stat().st_size
                    f.unlink()
                    _snap_bytes -= size
                except OSError:
                    pass
        target, n = SNAPSHOT_DIR / rel, 2
        while target.exists():  # two checks in one second: keep both
            target = target.with_name(f"{rel.stem}_{n}.jpg")
            n += 1
        rel = target.relative_to(SNAPSHOT_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        _snap_bytes += len(data)
    return rel.as_posix()


# --- the background runner ---------------------------------------------------------------------

_state: dict[str, dict] = {}
_thread: threading.Thread | None = None
_stop: threading.Event | None = None
_wake = threading.Event()
_running: set[str] = set()
_forced: set[str] = set()
_counter: dict[str, int] = {}


def _due_after(now: datetime, every: int) -> datetime:
    """The next tick on the playground's own grid: every 30 s means :00 and :30."""
    epoch = now.timestamp()
    return datetime.fromtimestamp((epoch // every + 1) * every, tz=now.tzinfo)


def _run_one(p: dict) -> None:
    try:
        n = _counter[p["id"]] = _counter.get(p["id"], 0) + 1
        every = p.get("save_every", 0)
        # The first check of a run saves too, so a fresh start shows a picture at once.
        record = check_playground(p, save=bool(every) and (n - 1) % every == 0)
        with _lock:
            st = _state.setdefault(p["id"], {})
            st.update(last=_summary(record), error=None)
    except Exception as exc:  # noqa: BLE001 - the runner must outlive one bad check
        log.exception("playground %s check failed", p["id"])
        with _lock:
            _state.setdefault(p["id"], {})["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        with _lock:
            _running.discard(p["id"])


def _summary(record: dict) -> dict:
    return {"checked_at": record["checked_at"], "people": record["people"],
            "playing": record["playing"], "cameras_read": record["cameras_read"],
            "cameras_total": record["cameras_total"], "duration_ms": record["duration_ms"]}


def _loop(stop: threading.Event) -> None:
    due: dict[str, datetime] = {}
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="playground") as pool:
        while not stop.is_set():
            now = _now()
            try:
                items = load()
            except Exception:  # noqa: BLE001
                items = []
            for p in items:
                forced = p["id"] in _forced
                if not forced and (not p["enabled"] or not in_hours(p, now)):
                    due.pop(p["id"], None)
                    continue
                nxt = due.get(p["id"])
                if nxt is None or nxt.timestamp() - p["every_seconds"] > now.timestamp():
                    nxt = due[p["id"]] = now if forced else _due_after(now, p["every_seconds"])
                if not forced and now < nxt:
                    continue
                with _lock:
                    _forced.discard(p["id"])
                    if p["id"] in _running:
                        _state.setdefault(p["id"], {})["skipped"] = (
                            _state.get(p["id"], {}).get("skipped", 0) + 1)
                    else:
                        _running.add(p["id"])
                        pool.submit(_run_one, p)
                due[p["id"]] = _due_after(now, p["every_seconds"])
                with _lock:
                    _state.setdefault(p["id"], {})["next_at"] = (
                        due[p["id"]].isoformat(timespec="seconds"))
            _wake.wait(timeout=0.5)
            _wake.clear()


def start() -> threading.Thread:
    """Start the runner (idle while no playground is on). A second call returns the first."""
    global _thread, _stop
    with _lock:
        if _thread is not None and _thread.is_alive():
            return _thread
        _stop = threading.Event()
        _thread = threading.Thread(target=_loop, args=(_stop,), name="playgrounds",
                                   daemon=True)
        _thread.start()
        return _thread


def stop(timeout: float = 5.0) -> None:
    global _thread, _stop
    with _lock:
        stopping, thread = _stop, _thread
        _thread, _stop = None, None
    if stopping is not None:
        stopping.set()
        _wake.set()
    if thread is not None:
        thread.join(timeout)


def run_now(playground_id: str) -> None:
    """Check one playground now, whether or not it is on or within its hours."""
    if not any(p["id"] == playground_id for p in load()):
        raise KeyError(playground_id)
    start()
    with _lock:
        _forced.add(playground_id)
    _wake.set()


def status() -> dict:
    with _lock:
        return {"running": _thread is not None and _thread.is_alive(),
                "playgrounds": {pid: {**st, "checking": pid in _running}
                                for pid, st in _state.items()}}


# --- looking back -------------------------------------------------------------------------------

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
    """One row per playground per period of ``day``: people, most at once, time playing.

    "Time playing" is the number of checks that were playing times the time between checks,
    so it is an estimate whose step is the check interval.
    """
    if period_minutes not in PERIOD_CHOICES:
        raise ValueError(f"period_minutes must be one of {list(PERIOD_CHOICES)}")
    every = {p["id"]: p["every_seconds"] for p in load()}
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
    saved_pgs = load()
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
            "verdict": _session_verdict(share, mx),
        })
    return out


def every_text(seconds: int) -> str:
    """"30 seconds", "minute", "5 minutes" - how often, in words."""
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    return "minute" if minutes == 1 else f"{minutes} minutes"
