"""The live worker's heartbeat: ``data/worker_status.json``.

The live worker is a separate process from the dashboard, on purpose (`worker.py`), and until
2026-10-10 nothing told the dashboard whether it was running at all. A worker that crashed at
03:00 looked exactly like one recording an empty evening: no new verdicts either way, and the
first sign of trouble was a day of missing slots found by the reconciliation.

So the worker writes a small JSON file - what it is recording, how far through, the last frame
per camera, what comes next, what finished, what went wrong - and refreshes ``updated_at`` at
least once a minute. A reader decides liveness from the age of that stamp, not from the
``state`` alone: a process that dies cannot write "stopped", so a crash leaves an old stamp
behind and :func:`read_status` reports ``alive: false``. That is the whole design - the
absence of a fresh heartbeat is the signal, which is the only kind a dead process can send.

**A file rather than a socket or a database row.** The dashboard may be down while the worker
records (and vice versa), the file survives either restarting, and it can be read with
``type data\\worker_status.json`` on a Mini-PC with nothing else working. Writes are atomic
(temporary file, then ``os.replace``) so a reader never sees half a document.

**Camera URLs never go in it.** An RTSP URL usually carries the camera's password, and this
file is served to the browser by ``GET /api/v1/recording/status``. Cameras appear by id, and
error messages are scrubbed of anything that looks like a URL before they are stored.

**A heartbeat write never stops a recording.** Every write swallows its own failure (a reader
holding the file open on Windows, a full disk): losing one status update costs an operator a
stale card for a minute, losing the slot costs an hour of observation.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pitch_occupancy.jsonstore import write_json_atomic

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pitch_occupancy.scheduler import Schedule, ScheduledSlot

__all__ = ["STATUS_FILE", "ALIVE_WITHIN_S", "status_path", "write_status", "read_status",
           "upcoming", "Heartbeat"]

STATUS_FILE = "worker_status.json"
#: How fresh ``updated_at`` must be for the worker to count as alive. The worker writes at
#: least once a minute, so three minutes tolerates two missed writes (a slow disk, a long
#: camera timeout) before the card turns red.
ALIVE_WITHIN_S = 180.0
#: States in which the process is (or claims to be) still working. "stopped" is a clean exit
#: and "error" is written only on the way out of a crash, so neither is alive however recent.
_LIVE_STATES = frozenset({"starting", "running"})

_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")


def status_path() -> Path:
    """``settings.data_dir / worker_status.json``, read at call time so a test can move it."""
    from pitch_occupancy.config import settings

    return Path(settings.data_dir) / STATUS_FILE


def _stamp(at: datetime | None = None) -> str:
    """Local time with its UTC offset, so a reader in another zone still gets it right."""
    return (at or datetime.now()).astimezone().isoformat(timespec="seconds")


def _scrub(message: str) -> str:
    """Remove anything that looks like a URL; it may carry a camera's password."""
    return _URL.sub("<url hidden>", str(message))


def write_status(state: dict, path: Path | None = None) -> bool:
    """Atomically write ``state`` as the worker's status file. Returns whether it was written.

    ``pid`` and ``updated_at`` are filled in when absent, so every write is a heartbeat. The
    document goes to a temporary file beside the target and is moved over it, so a reader sees
    either the old document or the new one. On Windows the move can fail while a reader has the
    file open; it is retried briefly and then given up, never raised (see the module docstring).
    """
    target = path or status_path()
    doc = dict(state)
    doc.setdefault("pid", os.getpid())
    doc.setdefault("updated_at", _stamp())
    try:
        write_json_atomic(target, doc, indent=1, trailing_newline=False, default=str,
                          attempts=5)
    except OSError:
        return False
    return True


def read_status(path: Path | None = None, *, now: datetime | None = None) -> dict | None:
    """The worker's last status, plus ``alive`` and ``age_s``; ``None`` if it never wrote one.

    ``alive`` is true when ``updated_at`` is under three minutes old and the state is
    "starting" or "running". ``age_s`` is the age of that stamp in seconds, so a page can say
    "last seen 2 hours ago" for a worker that died. A file that cannot be parsed is reported as
    state "unreadable", not alive, rather than as no file - something did write it.
    """
    source = path or status_path()
    if not source.exists():
        return None
    try:
        doc = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as exc:
        return {"state": "unreadable", "error": str(exc), "alive": False, "age_s": None}
    current = (now or datetime.now()).astimezone()
    age: float | None = None
    try:
        updated = datetime.fromisoformat(str(doc.get("updated_at")))
        if updated.tzinfo is None:
            updated = updated.astimezone()
        age = round((current - updated).total_seconds(), 1)
    except (TypeError, ValueError):
        pass
    doc["age_s"] = age
    doc["alive"] = bool(age is not None and age < ALIVE_WITHIN_S
                        and doc.get("state") in _LIVE_STATES)
    return doc


def upcoming(schedule: Schedule, now: datetime, n: int = 5) -> list[dict]:
    """The next ``n`` slot starts after ``now``, today or tomorrow."""
    found: list[tuple[datetime, ScheduledSlot, Date]] = []
    for offset in (0, 1):
        day = (now + timedelta(days=offset)).date()
        for slot in schedule.slots:
            if not slot.runs_on(day):
                continue
            begins, _ = slot.window(day)
            if begins > now:
                found.append((begins, slot, day))
    found.sort(key=lambda item: (item[0], item[1].venue_id))
    return [{"slot_id": slot.slot_id(day), "venue_id": slot.venue_id,
             "field_id": slot.field_id, "start": _stamp(begins)}
            for begins, slot, day in found[:n]]


class Heartbeat:
    """The live worker's status, kept in memory and written on every change.

    One instance per worker process, shared by the scheduler loop and every slot thread, so all
    of it sits behind one lock. Each method changes the document and writes it; the writes are
    tiny (a few KB) and at most a handful per minute, so there is no batching to go wrong.

    ``write=False`` keeps everything in memory, for a test that wants to inspect the document
    without touching a disk.
    """

    def __init__(self, *, model: str | None = None, schedule: Schedule | None = None,
                 path: Path | None = None, write: bool = True) -> None:
        self._lock = threading.Lock()
        self._path = path
        self._write = write
        self._schedule = schedule
        self._state = "starting"
        self._started_at = _stamp()
        self._model = model
        self._running: dict[str, dict] = {}
        self._next: list[dict] = []
        self._finished: deque[dict] = deque(maxlen=10)
        self._errors: deque[dict] = deque(maxlen=10)

    # -- the document ---------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return self._document()

    def _document(self) -> dict:
        return {
            "state": self._state,
            "pid": os.getpid(),
            "started_at": self._started_at,
            "updated_at": _stamp(),
            "model": self._model,
            "slots_running": [
                {**info, "cameras": {c: dict(v) for c, v in info["cameras"].items()}}
                for info in self._running.values()
            ],
            "next_slots": list(self._next),
            "last_finished": list(reversed(self._finished)),
            "errors": list(reversed(self._errors)),
        }

    def _flush(self) -> None:
        if self._write:
            write_status(self._document(), self._path)

    # -- events ---------------------------------------------------------------------------

    def start(self) -> None:
        """The process is up and loading; nothing is recording yet."""
        with self._lock:
            self._state = "starting"
            self._flush()

    def tick(self, now: datetime | None = None) -> None:
        """One scheduler pass: the worker is alive, and what comes next is recomputed."""
        with self._lock:
            if self._state == "starting":
                self._state = "running"
            if self._schedule is not None and now is not None:
                self._next = upcoming(self._schedule, now)
            self._flush()

    def slot_started(self, slot: ScheduledSlot, day: Date, *, minutes_expected: int,
                     cameras: list[str] | tuple[str, ...]) -> None:
        begins, ends = slot.window(day)
        with self._lock:
            self._running[slot.slot_id(day)] = {
                "slot_id": slot.slot_id(day), "venue_id": slot.venue_id,
                "field_id": slot.field_id, "start": _stamp(begins), "end": _stamp(ends),
                "minutes_done": 0, "minutes_expected": int(minutes_expected),
                "cameras": {c: {"last_frame_at": None, "frames": 0, "missed": 0}
                            for c in cameras},
            }
            self._flush()

    def frame(self, slot_id: str, camera: str, minute: int, *, got: bool) -> None:
        """One camera read for one minute: a frame (``got``) or a gap."""
        with self._lock:
            info = self._running.get(slot_id)
            if info is None:
                return
            cam = info["cameras"].setdefault(
                camera, {"last_frame_at": None, "frames": 0, "missed": 0})
            if got:
                cam["frames"] += 1
                cam["last_frame_at"] = _stamp()
            else:
                cam["missed"] += 1
            info["minutes_done"] = max(info["minutes_done"], minute + 1)
            self._flush()

    def slot_finished(self, slot_id: str, status: str) -> None:
        """A slot ended: its verdict status, or FAILED / CANCELLED when none was recorded."""
        with self._lock:
            self._running.pop(slot_id, None)
            self._finished.append(
                {"slot_id": slot_id, "status": str(status), "finished_at": _stamp()})
            self._flush()

    def error(self, message: str) -> None:
        with self._lock:
            self._errors.append({"at": _stamp(), "message": _scrub(message)[:500]})
            self._flush()

    def stop(self, state: str = "stopped", message: str | None = None) -> None:
        """The last write of a process: "stopped" on a clean exit, "error" on a crash."""
        with self._lock:
            self._state = state
            if message:
                self._errors.append({"at": _stamp(), "message": _scrub(message)[:500]})
            self._running.clear()
            self._flush()
