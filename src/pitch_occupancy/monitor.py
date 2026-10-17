"""Automatic camera checks: every few minutes, count the people each camera sees, and keep it.

The slot worker answers a booking question - was this pitch used during its rental slot - and
needs a schedule, fields and two cameras per pitch before it records anything. Most people
setting a site up want something plainer first: *is anyone there, and how many*, for every
camera, kept so it can be looked back on. That is this module.

* **What a check is.** The same spot check the Cameras page runs on demand
  (`api.cameras_page.predict_now`): one short burst from the camera, people counted inside
  the camera's boundary (the whole picture when none is drawn), the decision table's verdict.
  One row per camera per check goes into ``camera_checks``.
* **When.** A background thread in the server, switched on and off from the web page and
  configured in ``configs/monitor.json`` (gitignored, like the camera file): how often, which
  cameras, whether to keep a picture. It sleeps while switched off, so it costs nothing
  until someone turns it on, and it starts again by itself when the server restarts.
* **Looking back.** :func:`records` groups the checks into periods - an hour and a half by
  default - per camera: were there people, the most at once, the average.

Pictures are off by default. A check's picture shows identifiable people, and keeping one per
camera every few minutes is a decision for whoever switches it on, not a side effect.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from pitch_occupancy.config import CONFIGS_DIR, settings

__all__ = [
    "CONFIG_PATH", "EVERY_CHOICES", "PERIOD_CHOICES", "load_settings", "save_settings",
    "selected_cameras", "check_once", "records", "start_background", "stop_background",
    "run_now", "status",
]

log = logging.getLogger(__name__)

CONFIG_PATH: Path = CONFIGS_DIR / "monitor.json"

#: How often the checker may run, in minutes. A list rather than any number: each check of
#: one camera takes a few seconds, so "every 0.1 minutes" on fifteen cameras is not a setting
#: but a queue that never empties.
EVERY_CHOICES: tuple[int, ...] = (1, 2, 5, 10, 15, 30, 60, 90)
#: How the Records page may group checks, in minutes. 90 is the default the site asked for.
PERIOD_CHOICES: tuple[int, ...] = (15, 30, 60, 90, 120, 180)

DEFAULTS: dict = {"enabled": False, "every_minutes": 5, "cameras": [], "save_pictures": False}

_lock = threading.RLock()
_state: dict = {"checking_now": None, "last_started_at": None, "last_finished_at": None,
                "next_at": None, "last_summary": None, "last_error": None}
_thread: threading.Thread | None = None
_stop: threading.Event | None = None
_wake = threading.Event()
_force = threading.Event()


def _now() -> datetime:
    return datetime.now().astimezone()


# --- settings ----------------------------------------------------------------------------

def load_settings(path: Path | None = None) -> dict:
    """The checker's settings, defaults filled in. A missing or broken file reads as "off"."""
    source = path or CONFIG_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULTS)
    if not isinstance(raw, dict):
        return dict(DEFAULTS)
    out = dict(DEFAULTS)
    try:
        out.update(_clean({k: raw[k] for k in DEFAULTS if k in raw}))
    except ValueError as exc:
        log.warning("configs/%s ignored: %s", source.name, exc)
        return dict(DEFAULTS)
    return out


def _clean(values: dict) -> dict:
    out: dict = {}
    if "enabled" in values:
        if not isinstance(values["enabled"], bool):
            raise ValueError("enabled must be true or false")
        out["enabled"] = values["enabled"]
    if "every_minutes" in values:
        every = values["every_minutes"]
        if isinstance(every, bool) or every not in EVERY_CHOICES:
            raise ValueError(f"every_minutes must be one of {list(EVERY_CHOICES)}")
        out["every_minutes"] = int(every)
    if "cameras" in values:
        cams = values["cameras"]
        if not isinstance(cams, list) or not all(isinstance(c, str) and "/" in c for c in cams):
            raise ValueError('cameras must be a list of "place/camera" names (empty = all)')
        out["cameras"] = sorted(set(cams))
    if "save_pictures" in values:
        if not isinstance(values["save_pictures"], bool):
            raise ValueError("save_pictures must be true or false")
        out["save_pictures"] = values["save_pictures"]
    return out


def save_settings(changes: dict, path: Path | None = None) -> dict:
    """Merge ``changes`` into the settings and write them atomically. Raises ``ValueError`` on
    a bad value, before anything is written. Wakes the checker so a change applies now."""
    target = path or CONFIG_PATH
    with _lock:
        merged = load_settings(target)
        merged.update(_clean(changes))
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(".json.new")
        staging.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, target)
    _wake.set()
    return merged


def selected_cameras(conf: dict | None = None) -> list[tuple[str, str]]:
    """``(place, camera)`` pairs the checker covers: every saved camera, or the chosen ones."""
    from pitch_occupancy import cameras

    conf = conf or load_settings()
    saved = [(c["venue"], c["camera_id"]) for c in cameras.list_cameras()]
    wanted = set(conf.get("cameras") or [])
    if not wanted:
        return saved
    return [(v, c) for v, c in saved if f"{v}/{c}" in wanted]


# --- one round of checks -----------------------------------------------------------------

def _default_scorer(venue: str, camera: str) -> dict:
    """The Cameras page's spot check for one camera - the same counting the Live page shows."""
    from pitch_occupancy.api.cameras_page import PredictIn, predict_now

    result = predict_now(PredictIn(venue=venue, cameras=[camera]))
    return result["cameras"][0]


def _save_picture(snapshot_b64: str, venue: str, camera: str, when: datetime) -> str | None:
    folder = Path(settings.evidence_dir) / "checks" / when.strftime("%Y-%m-%d")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{venue}__{camera}__{when:%H%M%S}.jpg"
        path.write_bytes(base64.b64decode(snapshot_b64))
        return str(path)
    except (OSError, ValueError) as exc:
        log.warning("check picture for %s/%s not saved: %s", venue, camera, exc)
        return None


def check_once(*, scorer=None, db_path: Path | None = None, conf: dict | None = None,
               stop: threading.Event | None = None) -> list[dict]:
    """Check every selected camera once and store one row each. Returns the rows stored.

    One camera failing - unreachable, or its picture could not be scored - is a row saying
    so, never a reason to skip the others.
    """
    from pitch_occupancy.db.schema import connect, initialise, transaction

    conf = conf or load_settings()
    score = scorer or _default_scorer
    stored: list[dict] = []
    conn = connect(db_path or settings.db_path)
    try:
        initialise(conn)
        for venue, camera in selected_cameras(conf):
            if stop is not None and stop.is_set():
                break
            with _lock:
                _state["checking_now"] = f"{venue}/{camera}"
            when = _now()
            try:
                row = score(venue, camera) or {}
            except Exception as exc:  # noqa: BLE001 - one camera must not stop the round
                row = {"readable": False, "problem": f"check failed ({type(exc).__name__})"}
            picture = None
            if conf.get("save_pictures") and row.get("snapshot"):
                picture = _save_picture(row["snapshot"], venue, camera, when)
            record = {
                "venue_id": venue, "camera_id": camera,
                "checked_at": when.isoformat(timespec="seconds"),
                "readable": bool(row.get("readable")),
                "people": row.get("people"), "state": row.get("state"),
                "label": row.get("label"), "confidence": row.get("confidence"),
                "picture_path": picture, "problem": row.get("problem"),
            }
            with transaction(conn):
                cur = conn.execute(
                    """INSERT INTO camera_checks (venue_id, camera_id, checked_at, readable,
                           people, state, label, confidence, picture_path, problem)
                       VALUES (:venue_id, :camera_id, :checked_at, :readable, :people, :state,
                               :label, :confidence, :picture_path, :problem)""",
                    record,
                )
            record["check_id"] = cur.lastrowid
            stored.append(record)
    finally:
        conn.close()
        with _lock:
            _state["checking_now"] = None
    return stored


# --- looking back --------------------------------------------------------------------------

def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def records(conn, day: date, *, period_minutes: int = 90, venue: str | None = None,
            camera: str | None = None) -> list[dict]:
    """One row per camera per period of ``day``: were there people, how many, how often.

    Periods start at midnight: 00:00-01:30, 01:30-03:00, ... for the default hour and a half.
    Only periods with at least one check appear - a period nobody checked has no answer,
    which is different from "nobody was there".
    """
    if period_minutes not in PERIOD_CHOICES:
        raise ValueError(f"period_minutes must be one of {list(PERIOD_CHOICES)}")
    sql = "SELECT * FROM camera_checks WHERE substr(checked_at, 1, 10) = ?"
    args: list = [day.isoformat()]
    if venue:
        sql += " AND venue_id = ?"
        args.append(venue)
    if camera:
        sql += " AND camera_id = ?"
        args.append(camera)
    groups: dict[tuple, list] = {}
    for r in conn.execute(sql + " ORDER BY checked_at", args).fetchall():
        when = datetime.fromisoformat(r["checked_at"])
        bucket = (when.hour * 60 + when.minute) // period_minutes
        groups.setdefault((bucket, r["venue_id"], r["camera_id"]), []).append(r)

    out = []
    for (bucket, v, c), rows in sorted(groups.items()):
        counted = [r["people"] for r in rows if r["people"] is not None]
        readable = [r for r in rows if r["readable"]]
        labels = Counter(r["label"] for r in readable if r["label"])
        busiest = max((r for r in rows if r["people"] is not None),
                      key=lambda r: r["people"], default=None)
        start = bucket * period_minutes
        end = min(start + period_minutes, 24 * 60)
        out.append({
            "venue": v, "camera": c, "day": day.isoformat(),
            "start": _fmt(start), "end": _fmt(end) if end < 24 * 60 else "24:00",
            "checks": len(rows), "readable": len(readable),
            "people_seen": any(p > 0 for p in counted) if counted else None,
            "checks_with_people": sum(1 for p in counted if p > 0),
            "max_people": max(counted) if counted else None,
            "avg_people": round(sum(counted) / len(counted), 1) if counted else None,
            "usual_label": labels.most_common(1)[0][0] if labels else None,
            "first_check": rows[0]["checked_at"][11:16],
            "last_check": rows[-1]["checked_at"][11:16],
            "busiest_check_id": busiest["check_id"] if busiest is not None else None,
            "busiest_has_picture": bool(busiest is not None and busiest["picture_path"]),
        })
    return out


# --- the background thread ---------------------------------------------------------------

def _loop(stop: threading.Event) -> None:
    while not stop.is_set():
        conf = load_settings()
        forced = _force.is_set()
        _force.clear()
        wait = 30.0
        if conf["enabled"] or forced:
            started = time.monotonic()
            with _lock:
                _state["last_started_at"] = _now().isoformat(timespec="seconds")
                _state["last_error"] = None
            try:
                rows = check_once(conf=conf, stop=stop)
                summary = {"checked": len(rows),
                           "readable": sum(1 for r in rows if r["readable"]),
                           "with_people": sum(1 for r in rows if (r["people"] or 0) > 0)}
            except Exception as exc:  # noqa: BLE001 - the thread must outlive one bad round
                log.exception("camera check round failed")
                summary = None
                with _lock:
                    _state["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
            with _lock:
                _state["last_finished_at"] = _now().isoformat(timespec="seconds")
                if summary is not None:
                    _state["last_summary"] = summary
            if conf["enabled"]:
                wait = max(1.0, conf["every_minutes"] * 60 - (time.monotonic() - started))
        with _lock:
            _state["next_at"] = ((_now() + timedelta(seconds=wait)).isoformat(timespec="seconds")
                                 if load_settings()["enabled"] else None)
        _wake.wait(timeout=wait)
        _wake.clear()


def start_background() -> threading.Thread:
    """Start the checker thread (idle while switched off). A second call returns the first."""
    global _thread, _stop
    with _lock:
        if _thread is not None and _thread.is_alive():
            return _thread
        _stop = threading.Event()
        _thread = threading.Thread(target=_loop, args=(_stop,), name="pitch-monitor",
                                   daemon=True)
        _thread.start()
        return _thread


def stop_background(timeout: float = 5.0) -> None:
    global _thread, _stop
    with _lock:
        stop, thread = _stop, _thread
        _thread, _stop = None, None
    if stop is not None:
        stop.set()
        _wake.set()
    if thread is not None:
        thread.join(timeout)


def run_now() -> bool:
    """Ask for one round now, switched on or not. False when a round is already running."""
    with _lock:
        if _state["checking_now"] is not None:
            return False
    start_background()
    _force.set()
    _wake.set()
    return True


def status() -> dict:
    with _lock:
        running = _thread is not None and _thread.is_alive()
        return {**_state, "thread_running": running, "settings": load_settings()}
