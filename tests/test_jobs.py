"""The background job runner: listing and cancelling.

The functions here stand in for an analysis: they loop calling ``progress`` the way
`clip_analysis` and `pair_analysis` do after every sample, so cancelling one exercises exactly
the path a real day test takes - the callback raises, the function's ``finally`` runs.
"""

from __future__ import annotations

import threading
import time

from pitch_occupancy.api import jobs


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached in time")


def _status(job_id: str) -> str:
    return jobs.get(job_id)["status"]


def _looping(state: dict, steps: int = 100_000):
    """An analysis that never finishes on its own within a test."""

    def fn(progress):
        try:
            for i in range(steps):
                progress(i + 1, steps)
                time.sleep(0.005)
            return {"finished": True}
        finally:
            state["cleaned"] = True

    return fn


def test_a_running_job_stops_at_its_next_sample(tmp_path) -> None:
    state: dict = {}
    calls = []
    job_id = jobs.submit("day", _looping(state), label="morning.mp4",
                         on_cancel=lambda: calls.append("queued clean-up"))
    _wait_for(lambda: jobs.get(job_id)["progress"]["done"] > 0)
    assert jobs.cancel(job_id) is True
    _wait_for(lambda: _status(job_id) == "cancelled", timeout=5)
    snap = jobs.get(job_id)
    assert snap["result"] is None and snap["finished"] is not None
    assert state["cleaned"], "the function's finally did not run"
    assert calls == [], "the queued-only clean-up ran for a job that had started"
    assert jobs.cancel(job_id) is False, "a finished job cannot be cancelled again"


def test_a_queued_job_never_starts_and_is_cleaned_up() -> None:
    release = threading.Event()
    started: list[str] = []

    def blocker(progress):
        started.append("blocker")
        while not release.is_set():
            progress(0, 1)
            time.sleep(0.01)
        return {}

    def second(progress):
        started.append("second")
        return {}

    calls: list[str] = []
    first = jobs.submit("day", blocker)
    queued = jobs.submit("day", second, label="later.mp4",
                         on_cancel=lambda: calls.append("deleted"))
    try:
        _wait_for(lambda: _status(first) == "running")
        assert _status(queued) == "queued"
        assert jobs.cancel(queued) is True
        assert _status(queued) == "cancelled"
        assert calls == ["deleted"]
    finally:
        release.set()
    _wait_for(lambda: _status(first) == "done")
    time.sleep(0.05)
    assert started == ["blocker"], "a cancelled queued job was run anyway"


def test_unknown_and_finished_jobs_cannot_be_cancelled() -> None:
    assert jobs.cancel("no-such-job") is False
    job_id = jobs.submit("day", lambda progress: {"ok": 1})
    _wait_for(lambda: _status(job_id) == "done")
    assert jobs.cancel(job_id) is False


def test_list_jobs_is_newest_first_without_results() -> None:
    a = jobs.submit("day", lambda progress: {"big": "x" * 1000}, label="a.mp4")
    b = jobs.submit("day", lambda progress: {"big": "y"}, label="b.mp4")
    _wait_for(lambda: _status(b) == "done")
    listed = jobs.list_jobs(5)
    ids = [j["job_id"] for j in listed]
    assert ids.index(b) < ids.index(a)
    row = next(j for j in listed if j["job_id"] == a)
    assert row["label"] == "a.mp4" and row["kind"] == "day" and row["status"] == "done"
    assert "result" not in row
    assert set(row) >= {"job_id", "kind", "label", "status", "progress", "created"}


def test_a_failing_job_still_reports_failed() -> None:
    def fn(progress):
        raise ValueError("unreadable video")

    job_id = jobs.submit("day", fn)
    _wait_for(lambda: _status(job_id) == "failed")
    assert jobs.get(job_id)["error"] == "unreadable video"

