"""The background runner: one thread in the server keeping a clock per playground.

Each playground is checked on its own grid (every 30 s means :00 and :30), only while it is
on and within its hours. Playgrounds run at the same time as each other. A check still running
when its next one falls due skips that turn rather than queueing - a queue that never empties
is how a busy computer falls hours behind.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from pitch_occupancy.playgrounds import check, store

__all__ = ["start", "stop", "run_now", "status"]

log = logging.getLogger(__name__)

_lock = threading.RLock()
_state: dict[str, dict] = {}
_thread: threading.Thread | None = None
_stop: threading.Event | None = None
_running: set[str] = set()
_forced: set[str] = set()
#: Checks run per playground since the server started.
_counter: dict[str, int] = {}


def _due_after(now: datetime, every: int) -> datetime:
    """The next tick on the playground's own grid: every 30 s means :00 and :30."""
    epoch = now.timestamp()
    return datetime.fromtimestamp((epoch // every + 1) * every, tz=now.tzinfo)


#: The clock the picture spacing is measured on (replaced in tests).
_clock = time.monotonic
#: When each playground last saved its pictures (``_clock`` seconds), for ``save_minutes``.
_last_saved: dict[str, float] = {}


def _save_due(pid: str, minutes: int, now: float) -> bool:
    """Whether this check should save its pictures; the first check of a run always does,
    so a fresh start shows a picture at once."""
    if not minutes:
        return False
    with _lock:
        last = _last_saved.get(pid)
        # A few seconds of slack: checks land on a grid, and a check that starts a moment
        # early must not push the next picture a whole interval later.
        if last is not None and now - last < minutes * 60 - 5:
            return False
        _last_saved[pid] = now
        return True


def _run_one(p: dict) -> None:
    try:
        _counter[p["id"]] = _counter.get(p["id"], 0) + 1
        save = _save_due(p["id"], p.get("save_minutes", 0), _clock())
        record = check.check_playground(p, save=save)
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


def _stay_awake(on: bool) -> None:
    """Ask Windows not to idle-sleep while playgrounds are being checked (on this thread).

    The same request a video player makes: it changes no setting, and it ends when the
    server stops. A closed lid still sleeps. Does nothing elsewhere.
    """
    import sys

    if sys.platform != "win32":
        return
    try:
        import ctypes

        continuous, system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            continuous | (system_required if on else 0))
    except Exception:  # noqa: BLE001 - staying awake is a courtesy, never a failure
        log.debug("could not change the sleep request", exc_info=True)


def _loop(stop: threading.Event) -> None:
    due: dict[str, datetime] = {}
    awake = False
    try:
        _run_loop(stop, due, awake)
    finally:
        _stay_awake(False)


def _run_loop(stop: threading.Event, due: dict[str, datetime], awake: bool) -> None:
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="playground") as pool:
        while not stop.is_set():
            now = store.now()
            try:
                items = store.load()
            except Exception:  # noqa: BLE001
                items = []
            wanted = any(p["enabled"] for p in items)
            if wanted != awake:
                _stay_awake(wanted)
                awake = wanted
            for p in items:
                forced = p["id"] in _forced
                if not forced and (not p["enabled"] or not store.in_hours(p, now)):
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
            store.changed.wait(timeout=0.5)
            store.changed.clear()


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
        store.changed.set()
    if thread is not None:
        thread.join(timeout)


def run_now(playground_id: str) -> None:
    """Check one playground now, whether or not it is on or within its hours."""
    if not any(p["id"] == playground_id for p in store.load()):
        raise KeyError(playground_id)
    start()
    with _lock:
        _forced.add(playground_id)
    store.changed.set()


def status() -> dict:
    with _lock:
        return {"running": _thread is not None and _thread.is_alive(),
                "playgrounds": {pid: {**st, "checking": pid in _running}
                                for pid, st in _state.items()}}
