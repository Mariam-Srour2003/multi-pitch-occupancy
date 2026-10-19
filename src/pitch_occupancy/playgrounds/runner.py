"""The background runner: one thread in the server keeping a clock per playground.

Each playground is checked on its own grid (every 30 s means :00 and :30), only while it is
on and within its hours. Playgrounds run at the same time as each other. A check still running
when its next one falls due skips that turn rather than queueing - a queue that never empties
is how a busy computer falls hours behind.
"""

from __future__ import annotations

import logging
import threading
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
#: Checks run per playground since the server started, for ``save_every``.
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
        record = check.check_playground(p, save=bool(every) and (n - 1) % every == 0)
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
            now = store.now()
            try:
                items = store.load()
            except Exception:  # noqa: BLE001
                items = []
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
