"""The live worker's heartbeat, `data/worker_status.json` (2026-10-10).

A worker that crashed at 03:00 used to look exactly like one recording a quiet evening. The
heartbeat is a file the worker refreshes at least once a minute; the reader decides liveness
from its age, because a dead process cannot write "stopped". Every test here points
`settings.data_dir` at a temporary directory - none may write the real one.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import status
from pitch_occupancy.config import settings
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.scheduler import Schedule, ScheduledSlot, live_sources, run_concurrently

SECRET = "s3cretPW"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def slot(venue: str = "venue_01", start: str = "10:00", minutes: int = 3) -> ScheduledSlot:
    hh, mm = (int(x) for x in start.split(":"))
    return ScheduledSlot(venue, time(hh, mm), minutes, ("camera_A", "camera_B"),
                         field_id="field_01")


# --- the file -------------------------------------------------------------------------------


def test_no_file_means_none(data_dir) -> None:
    assert status.read_status() is None


def test_a_fresh_running_status_is_alive(data_dir) -> None:
    assert status.write_status({"state": "running"})
    doc = status.read_status()
    assert doc["state"] == "running" and doc["alive"] is True
    assert doc["age_s"] < 60
    assert "pid" in doc
    assert doc["updated_at"][19] in "+-", "local time with its UTC offset"
    assert (data_dir / "worker_status.json").exists()
    assert not list(data_dir.glob(".*.tmp")), "the temporary file was moved, not left behind"


def test_an_old_heartbeat_is_not_alive_whatever_it_says(data_dir) -> None:
    """A crash leaves the last document behind with state "running"; its age is the signal."""
    old = (datetime.now() - timedelta(minutes=10)).astimezone().isoformat(timespec="seconds")
    status.write_status({"state": "running", "updated_at": old})
    doc = status.read_status()
    assert doc["alive"] is False
    assert 590 < doc["age_s"] < 700


def test_stopped_and_error_are_not_alive_even_when_fresh(data_dir) -> None:
    for state in ("stopped", "error"):
        status.write_status({"state": state})
        assert status.read_status()["alive"] is False


def test_alive_is_judged_against_the_time_given(data_dir) -> None:
    status.write_status({"state": "running"})
    later = datetime.now() + timedelta(minutes=4)
    assert status.read_status(now=later)["alive"] is False


def test_an_unreadable_file_is_reported_not_treated_as_absent(data_dir) -> None:
    (data_dir / "worker_status.json").write_text("{half a docu", encoding="utf-8")
    doc = status.read_status()
    assert doc["state"] == "unreadable" and doc["alive"] is False


def test_errors_are_scrubbed_of_urls() -> None:
    beat = status.Heartbeat(model="m", write=False)
    beat.error(f"cannot open rtsp://admin:{SECRET}@10.0.0.5:554/Streaming/Channels/102 now")
    text = json.dumps(beat.snapshot())
    assert SECRET not in text and "rtsp://" not in text
    assert "<url hidden>" in text


# --- the lifecycle --------------------------------------------------------------------------


def test_the_heartbeat_goes_starting_running_stopped(data_dir) -> None:
    schedule = Schedule((slot("venue_01", "10:00"), slot("venue_02", "18:00")))
    beat = status.Heartbeat(model="yolov8n", schedule=schedule)
    beat.start()
    assert status.read_status()["state"] == "starting"

    beat.tick(datetime(2026, 7, 11, 9, 0))
    doc = status.read_status()
    assert doc["state"] == "running" and doc["alive"] and doc["model"] == "yolov8n"
    assert [s["slot_id"] for s in doc["next_slots"]][:3] == [
        "venue_01_2026-07-11_1000", "venue_02_2026-07-11_1800", "venue_01_2026-07-12_1000"]

    beat.slot_started(schedule.slots[0], date(2026, 7, 11), minutes_expected=3,
                      cameras=["camera_A", "camera_B"])
    beat.frame("venue_01_2026-07-11_1000", "camera_A", 0, got=True)
    beat.frame("venue_01_2026-07-11_1000", "camera_B", 0, got=False)
    (running,) = status.read_status()["slots_running"]
    assert running["minutes_done"] == 1 and running["minutes_expected"] == 3
    assert running["cameras"]["camera_A"]["frames"] == 1
    assert running["cameras"]["camera_A"]["last_frame_at"]
    assert running["cameras"]["camera_B"] == {"last_frame_at": None, "frames": 0, "missed": 1}

    beat.slot_finished("venue_01_2026-07-11_1000", "USED")
    doc = status.read_status()
    assert doc["slots_running"] == []
    assert doc["last_finished"][0]["slot_id"] == "venue_01_2026-07-11_1000"
    assert doc["last_finished"][0]["status"] == "USED"

    beat.stop()
    doc = status.read_status()
    assert doc["state"] == "stopped" and doc["alive"] is False


def test_only_the_last_ten_finished_and_errors_are_kept() -> None:
    beat = status.Heartbeat(write=False)
    for i in range(15):
        beat.slot_finished(f"s{i}", "USED")
        beat.error(f"e{i}")
    doc = beat.snapshot()
    assert len(doc["last_finished"]) == 10 and doc["last_finished"][0]["slot_id"] == "s14"
    assert len(doc["errors"]) == 10 and doc["errors"][0]["message"] == "e14"


class FakeCapture:
    def read(self):
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        pass


class FakeClock:
    def __init__(self, at: datetime) -> None:
        self.now = at

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def test_the_live_runner_updates_the_heartbeat_and_never_writes_a_url(data_dir,
                                                                      monkeypatch) -> None:
    """Every document the runner writes is captured, and none may carry a camera URL - the file
    is served to the browser. Per-minute updates, the finish and the tick are all there."""
    written: list[dict] = []
    original = status.write_status

    def capture(state, path=None):
        written.append(json.loads(json.dumps(state, default=str)))
        return original(state, path)

    monkeypatch.setattr(status, "write_status", capture)
    urls = {"venue_01": {"camera_A": f"rtsp://admin:{SECRET}@10.0.0.5/a",
                         "camera_B": f"rtsp://admin:{SECRET}@10.0.0.6/b"}}
    clock = FakeClock(datetime(2026, 7, 11, 10, 0))
    schedule = Schedule((slot(minutes=3),))
    beat = status.Heartbeat(model="m", schedule=schedule)
    beat.start()
    ran = run_concurrently(
        schedule, source_for=live_sources(urls, clock=clock, sleep=clock.sleep,
                                          open_capture=lambda url: FakeCapture()),
        classify=lambda image: (Class3.ACTIVE_PLAY, 0.9), clock=clock, iterations=1,
        heartbeat=beat,
    )
    beat.stop()
    assert ran == ["venue_01_2026-07-11_1000"]

    text = json.dumps(written)
    assert SECRET not in text and "rtsp://" not in text and "10.0.0.5" not in text
    progress = [d["slots_running"][0]["minutes_done"] for d in written if d["slots_running"]]
    assert progress[-1] == 3 and sorted(set(progress)) == [0, 1, 2, 3], "updated every minute"
    final = status.read_status()
    assert final["state"] == "stopped"
    assert final["last_finished"][0]["slot_id"] == "venue_01_2026-07-11_1000"
    assert final["last_finished"][0]["status"] == "USED"
    assert SECRET not in (data_dir / "worker_status.json").read_text(encoding="utf-8")


# --- the worker writes "stopped" or "error" on the way out ----------------------------------


def _live_args(tmp_path) -> argparse.Namespace:
    config = tmp_path / "cameras.json"
    config.write_text(json.dumps({"venue_01": {"camera_A": "rtsp://h/1",
                                               "camera_B": "rtsp://h/2"}}), encoding="utf-8")
    return argparse.Namespace(cameras=config, dry_run=False, yes=True, model="m",
                              evidence_dir=None, iterations=1, derive_roi=False)


@pytest.fixture
def live_worker(monkeypatch):
    from pitch_occupancy import scheduler, worker

    monkeypatch.setattr(scheduler, "load_schedule", lambda: Schedule((slot(),)))
    return worker


def test_a_clean_live_exit_writes_stopped(tmp_path, live_worker, monkeypatch) -> None:
    states: list[str] = []

    def record(args, urls, schedule, heartbeat):
        states.append(status.read_status()["state"])
        return []

    monkeypatch.setattr(live_worker, "_record_live", record)
    live_worker._run_live(_live_args(tmp_path))
    assert states == ["starting"], "the heartbeat is written before the pipeline loads"
    assert status.read_status()["state"] == "stopped"


def test_a_crashed_live_run_writes_error_with_the_message(tmp_path, live_worker,
                                                          monkeypatch) -> None:
    def explode(*args):
        raise RuntimeError("the model would not load")

    monkeypatch.setattr(live_worker, "_record_live", explode)
    with pytest.raises(RuntimeError):
        live_worker._run_live(_live_args(tmp_path))
    doc = status.read_status()
    assert doc["state"] == "error" and doc["alive"] is False
    assert "the model would not load" in doc["errors"][0]["message"]


def test_a_dry_run_writes_no_status_file(tmp_path, live_worker) -> None:
    args = _live_args(tmp_path)
    args.dry_run = True
    live_worker._run_live(args)
    assert status.read_status() is None


# --- GET /api/v1/recording/status -----------------------------------------------------------


@pytest.fixture
def client():
    from pitch_occupancy.api import cameras_page

    app = FastAPI()
    app.include_router(cameras_page.router)
    return TestClient(app)


def test_the_route_without_a_file_gives_the_start_command(client) -> None:
    body = client.get("/api/v1/recording/status").json()
    assert body["status"] is None
    assert "pitch_occupancy.worker --source live" in body["start_command"]


def test_the_route_passes_the_status_through_with_alive(client) -> None:
    beat = status.Heartbeat(model="yolov8n")
    beat.start()
    beat.tick()
    body = client.get("/api/v1/recording/status").json()
    assert body["status"]["state"] == "running"
    assert body["status"]["alive"] is True
    assert body["status"]["model"] == "yolov8n"


def test_the_cameras_page_has_the_recording_card(client) -> None:
    html = client.get("/cameras").text
    assert "Recording status" in html and "/api/v1/recording/status" in html
    assert "has not been started on this computer" in html
