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

**Cancelling is cooperative.** A queued job is simply never started. A running one cannot be
killed from outside - Python threads have no safe way to be stopped - so :func:`cancel` sets a
flag and the job's own ``progress`` callback raises :class:`Cancelled` at the next sample. The
analyses call ``progress`` after every sample, so a cancel takes effect within one sample, and
the exception unwinds through the function's own ``finally`` blocks: a test of an upload still
deletes its temporary video, exactly as when it fails. The job ends as ``cancelled``. A job
cancelled before it started never reaches that ``finally``, so :func:`submit` takes an
``on_cancel`` clean-up for that case.
"""

from __future__ import annotations

import contextlib
import itertools
import queue
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Job", "Cancelled", "submit", "get", "cancel", "list_jobs", "KEEP", "FINISHED"]

#: How many finished jobs are remembered.
KEEP: int = 50

Progress = Callable[[int, int], None]

#: Statuses a job does not leave.
FINISHED: frozenset[str] = frozenset({"done", "failed", "cancelled"})


class Cancelled(Exception):  # noqa: N818 - it is an event, not an error, and reads as one
    """Raised from a running job's ``progress`` callback once :func:`cancel` asked it to stop."""


@dataclass
class Job:
    id: str
    kind: str
    seq: int
    status: str = "queued"  # queued, running, done, failed, cancelled
    label: str = ""
    cancel_requested: bool = False
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def snapshot(self, *, with_result: bool = True) -> dict[str, Any]:
        out = {
            "job_id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
            "progress": {"done": self.done, "total": self.total},
            "cancel_requested": self.cancel_requested,
            "error": self.error, "created": self.created, "finished": self.finished,
        }
        if with_result:
            out["result"] = self.result if self.status == "done" else None
        return out


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_fns: dict[str, Callable[[Progress], Any]] = {}
_on_cancel: dict[str, Callable[[], None]] = {}
_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None
_seq = itertools.count()


def _prune() -> None:
    """Drop the oldest finished jobs beyond :data:`KEEP`. Caller holds the lock."""
    finished = sorted((j for j in _jobs.values() if j.status in FINISHED),
                      key=lambda j: j.seq)
    for job in finished[:max(0, len(finished) - KEEP)]:
        del _jobs[job.id]


def _run_one(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        fn = _fns.pop(job_id, None)
        _on_cancel.pop(job_id, None)  # it has started: its own ``finally`` cleans up now
        if job is None or fn is None:
            return
        job.status = "running"

    def progress(done: int, total: int) -> None:
        with _lock:
            job.done, job.total = int(done), int(total)
            if job.cancel_requested:
                raise Cancelled(f"job {job.id} was cancelled")

    try:
        result = fn(progress)
    except Cancelled:
        with _lock:
            job.status, job.finished = "cancelled", time.time()
            _prune()
        return
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


def submit(kind: str, fn: Callable[[Progress], Any], label: str = "",
           on_cancel: Callable[[], None] | None = None) -> str:
    """Queue ``fn(progress)`` and return its job id. ``fn`` must return something JSON-able.

    ``label`` is what a list of jobs shows for it, such as the video's name. ``on_cancel`` is
    called if the job is cancelled *before it starts* - for whatever ``fn`` would have cleaned
    up in its own ``finally``, such as an upload's temporary files.
    """
    job_id = secrets.token_hex(8)
    with _lock:
        _jobs[job_id] = Job(id=job_id, kind=kind, seq=next(_seq), label=label)
        _fns[job_id] = fn
        if on_cancel is not None:
            _on_cancel[job_id] = on_cancel
        _ensure_worker()
    _queue.put(job_id)
    return job_id


def get(job_id: str) -> dict[str, Any] | None:
    """The job's status, progress and result or error; None for an unknown or pruned id."""
    with _lock:
        job = _jobs.get(job_id)
        return None if job is None else job.snapshot()


def cancel(job_id: str) -> bool:
    """Stop a job. True if it was queued or running; False if unknown or already finished.

    A queued job is marked ``cancelled`` at once and never starts. A running job is flagged;
    it becomes ``cancelled`` when its next ``progress`` call raises :class:`Cancelled` - its
    snapshot says ``cancel_requested`` meanwhile.
    """
    cleanup = None
    with _lock:
        job = _jobs.get(job_id)
        if job is None or job.status in FINISHED:
            return False
        if job.status == "queued":
            _fns.pop(job_id, None)
            cleanup = _on_cancel.pop(job_id, None)
            job.status, job.finished = "cancelled", time.time()
            _prune()
        else:
            job.cancel_requested = True
    if cleanup is not None:
        with contextlib.suppress(Exception):  # the job is cancelled either way
            cleanup()
    return True


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """The most recent jobs, newest first, without their results (which can be large)."""
    with _lock:
        recent = sorted(_jobs.values(), key=lambda j: j.seq, reverse=True)[:max(0, limit)]
        return [j.snapshot(with_result=False) for j in recent]
