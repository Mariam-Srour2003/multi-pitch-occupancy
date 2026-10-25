"""One check of one playground: a picture from every camera at once, counted, combined, stored.

Grabbing, counting and combining are the shared live step (`runtime`); this module adds the
playground's own parts. Picture-taking across all playgrounds is limited to
``settings.parallel_grabs`` streams at once, since a slow link times out when many open together.

After a check is stored, a second opinion from a free vision model may be queued
(`assist/`, off by default). It runs on its own thread with a short queue, so a slow provider
never holds a check open and a backlog is dropped rather than built up.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pitch_occupancy import runtime
from pitch_occupancy.config import settings
from pitch_occupancy.playgrounds import snapshots, store
from pitch_occupancy.runtime import combine_counts as combine
from pitch_occupancy.runtime import detect_lock, play_min

__all__ = ["check_playground", "combine", "play_min", "wait_for_second_opinions",
           "detect_lock", "GRAB_TIMEOUT_S", "MAX_ASSIST_WAITING"]

log = logging.getLogger(__name__)

#: Longest wait for one camera's picture in a check, whatever the interval.
GRAB_TIMEOUT_S = 20.0
_grab_slots = threading.BoundedSemaphore(max(1, settings.parallel_grabs))


def check_playground(p: dict, *, grab=None, pipeline=None,
                     db_path: Path | None = None, with_pictures: bool = False,
                     save: bool = False) -> dict:
    """Take one picture from every camera of ``p`` at the same moment, count, combine, store.

    Returns the stored playground row, with each camera's result under ``cameras`` - and
    each camera's picture (base64 JPEG, drawn) when ``with_pictures`` is set, for the live
    page. With ``save`` the drawn pictures are also kept (`snapshots.save_snapshot`).
    ``playing`` is NULL when the rule could not decide (`runtime.combine_pitch`).
    """
    from pitch_occupancy import cameras
    from pitch_occupancy.db.schema import connect, initialise, transaction

    grab = grab or runtime.grab_whole
    started = time.monotonic()
    when = store.now()
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
        pipeline = runtime.shared_pipeline()
    per_camera = []
    verdicts: dict[str, object] = {}  # each counted camera's verdict
    pictures: list[tuple] = []  # (frame, person boxes) for the second opinion
    for (ref, venue, camera, _), (frame, problem, grab_ms) in zip(targets, grabbed, strict=True):
        row = {"camera": ref, "venue": venue, "camera_id": camera, "readable": frame is not None,
               "people": None, "state": None, "problem": problem, "grab_ms": grab_ms,
               "detect_ms": None, "picture": None, "picture_path": None,
               "label": None, "confidence": None, "ball": None, "ball_confidence": None,
               "people_seen": None, "rule": None, "reason": None}
        if frame is not None:
            try:
                counted = runtime.count_camera(pipeline, venue, camera, frame)
                row.update(people=counted["people"], state=counted["state"],
                           detect_ms=counted["detect_ms"], **_details(counted["verdict"]))
                pictures.append((frame, _person_boxes(counted)))
                if with_pictures:
                    row["picture"] = _picture(frame, counted)
            except Exception as exc:  # noqa: BLE001
                row.update(readable=False, problem=f"Counting failed ({type(exc).__name__}).")
            else:
                verdicts[ref] = counted["verdict"]
                if save:
                    # Kept apart from the count: a full disk loses the picture, not the count.
                    try:
                        row["picture_path"] = snapshots.save_snapshot(p["id"], camera, when,
                                                                      frame, counted)
                    except OSError as exc:
                        row["problem"] = f"Picture not saved ({type(exc).__name__})."
        per_camera.append(row)

    counts = [r["people"] for r in per_camera if r["readable"] and r["people"] is not None]
    rules = getattr(pipeline, "rules", None) or runtime.load_rules()
    total, playing = runtime.combine_pitch(p["counting"], verdicts, rules)
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
                           playground_check_id, grab_ms, detect_ms, ball, ball_confidence,
                           people_seen, rule, reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r["venue"], r["camera_id"], record["checked_at"], int(r["readable"]),
                     r["people"], r["state"], r["label"], r["confidence"], r["picture_path"],
                     r["problem"], record["check_id"], r["grab_ms"], r["detect_ms"],
                     r["ball"], r["ball_confidence"], r["people_seen"], r["rule"],
                     r["reason"]))
    finally:
        conn.close()
    if not with_pictures:  # a spot check from the live page does not spend the free quota
        _second_opinion(record, pictures, db_path or settings.db_path)
    record["cameras"] = per_camera
    return record


#: The words the pages and the database use for each state.
LABELS = {"C1_EMPTY": "Empty", "C2_ACTIVE_PLAY": "Playing",
          "C3_MAINTENANCE_NON_SPORTING": "People, not playing", "UNCERTAIN": "Not sure"}


def _details(verdict) -> dict:
    """What a camera's verdict rests on, for the database: label, confidence, ball,
    everyone detected (inside the area or not), the rule row and its reason."""
    state = str(getattr(verdict.state, "value", verdict.state))
    count = verdict.count
    return {
        "label": LABELS.get(state, state),
        "confidence": round(float(verdict.confidence), 3),
        "ball": None if count is None else int(bool(count.ball_seen)),
        "ball_confidence": None if count is None else round(float(count.ball_confidence), 3),
        "people_seen": None if count is None else len(count.seen_people),
        "rule": verdict.rule or None,
        "reason": verdict.trace[-1] if verdict.trace else None,
    }


def _person_boxes(counted: dict) -> tuple[tuple[int, int, int, int], ...]:
    count = getattr(counted.get("verdict"), "count", None)
    people = getattr(count, "seen_people", ()) or ()
    return tuple(d.box for d in people)


def _picture(frame, counted) -> str | None:
    from pitch_occupancy import cameras

    return cameras.jpeg_base64(snapshots.drawn(frame, counted), max_width=640)


# --- the second opinion ----------------------------------------------------------------------

#: At most this many second opinions wait; beyond that a question is dropped, since a backlog
#: of old pictures is worth less than the next check.
MAX_ASSIST_WAITING = 2
_assist_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="assist")
_assist_waiting = 0
_assist_lock = threading.Lock()


def _second_opinion(record: dict, pictures: list[tuple], db_path: Path) -> None:
    """Queue a question to the free vision model about this check, when
    `configs/assist.json` says to. Never fails the check."""
    global _assist_waiting
    if not pictures:
        return
    try:
        from pitch_occupancy.assist import config as assist_config

        if not assist_config.load().enabled:
            return
    except Exception:  # noqa: BLE001 - a broken settings file means no second opinion
        log.exception("assist settings could not be read")
        return
    with _assist_lock:
        if _assist_waiting >= MAX_ASSIST_WAITING:
            return
        _assist_waiting += 1
    _assist_pool.submit(_ask_second_opinion, dict(record), pictures, db_path)


def _ask_second_opinion(record: dict, pictures: list[tuple], db_path: Path) -> None:
    global _assist_waiting
    try:
        from pitch_occupancy.assist import config as assist_config
        from pitch_occupancy.assist.reviewer import CameraPicture, review
        from pitch_occupancy.db.schema import connect

        conn = connect(db_path)
        try:
            review(assist_config.load(), conn, record,
                   [CameraPicture(f, b) for f, b in pictures], now=store.now(),
                   play_min=play_min())
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - a second opinion must never cost the first
        log.exception("second opinion failed for %s", record.get("playground_id"))
    finally:
        with _assist_lock:
            _assist_waiting -= 1


def wait_for_second_opinions(timeout: float = 60.0) -> None:
    """Block until every queued second opinion has finished (for tests and shutdown)."""
    _assist_pool.submit(lambda: None).result(timeout=timeout)
