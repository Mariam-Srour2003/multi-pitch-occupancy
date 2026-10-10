"""A tiny in-process job runner, so a day of footage does not have to fit in one request.

A booking-day test samples hours of video. That is minutes of CPU, longer than a browser or a
proxy will hold a request open, so the route hands the work to this runner and returns a job
id the page polls.

**One worker thread, on purpose.** The work is CPU-bound and the box is a Mini-PC: two
analyses side by side would each take twice as long and finish no sooner. So jobs queue and
run one at a time, in submission order.

**Results only.** A job holds what its function returned - numbers and labels - or the error
message. The footage paths belong to the caller's function, which deletes them itself; nothing
about the video is kept here. Finished jobs are pruned to the last :data:`KEEP` so a long-running
server does not grow a list of every test ever run (saved runs live on disk, elsewhere).
"""

from __future__ import annotations

import itertools
import queue
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Job", "submit", "get", "KEEP"]

#: How many finished jobs are remembered.
KEEP: int = 50

Progress = Callable[[int, int], None]


@dataclass
class Job:
    id: str
    kind: str
    seq: int
    status: str = "queued"  # queued, running, done, failed
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id, "kind": self.kind, "status": self.status,
            "progress": {"done": self.done, "total": self.total},
            "result": self.result if self.status == "done" else None,
            "error": self.error, "created": self.created, "finished": self.finished,
        }


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_fns: dict[str, Callable[[Progress], Any]] = {}
_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None
_seq = itertools.count()


def _prune() -> None:
    """Drop the oldest finished jobs beyond :data:`KEEP`. Caller holds the lock."""
    finished = sorted((j for j in _jobs.values() if j.status in ("done", "failed")),
                      key=lambda j: j.seq)
    for job in finished[:max(0, len(finished) - KEEP)]:
        del _jobs[job.id]


def _run_one(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        fn = _fns.pop(job_id, None)
        if job is None or fn is None:
            return
        job.status = "running"

    def progress(done: int, total: int) -> None:
        with _lock:
            job.done, job.total = int(done), int(total)

    try:
        result = fn(progress)
    except Exception as exc:  # noqa: BLE001 - reported to the page, not swallowed
        with _lock:
            job.status, job.error = "failed", str(exc) or exc.__class__.__name__
            job.finished = time.time()
            _prune()
        return
    with _lock:
        job.status, job.result, job.finished = "done", result, time.time()
        if job.total and job.done < job.total:
            job.done = job.total
        _prune()


def _loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            _run_one(job_id)
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_loop, name="testing-jobs", daemon=True)
        _worker.start()


def submit(kind: str, fn: Callable[[Progress], Any]) -> str:
    """Queue ``fn(progress)`` and return its job id. ``fn`` must return something JSON-able."""
    job_id = secrets.token_hex(8)
    with _lock:
        _jobs[job_id] = Job(id=job_id, kind=kind, seq=next(_seq))
        _fns[job_id] = fn
        _ensure_worker()
    _queue.put(job_id)
    return job_id


def get(job_id: str) -> dict[str, Any] | None:
    """The job's status, progress and result or error; None for an unknown or pruned id."""
    with _lock:
        job = _jobs.get(job_id)
        return None if job is None else job.snapshot()
