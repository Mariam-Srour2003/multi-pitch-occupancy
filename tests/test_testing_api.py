"""The booking-day test over HTTP: upload, background job, saved run, CSV, bookings.

The classifier is scripted by frame colour - dark frames are play, light frames are an empty
pitch - so a two-minute generated video is one used slot and one unused one, and the route's
arithmetic can be checked by hand. What is pinned beyond the numbers is what the page promises
the client: the footage is gone once the job ends, only results are saved, and a run id
cannot be used to read anything but a run.
"""

from __future__ import annotations

import sys
import threading
import time as clock
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy.api import clip_pair, clip_review, testing
from pitch_occupancy.bookings import read_bookings
from pitch_occupancy.config import settings
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.pipeline import Pipeline
from pitch_occupancy.vision.detector import PERSON, SPORTS_BALL, Detection, DetectorSpec
from pitch_occupancy.vision.pitch_classifier import DetectorFirstClassifier
from pitch_occupancy.vision.rules import RuleConfig

DARK, LIGHT = 40, 200


def _video(path: Path, greys: list[int], fps: int = 2) -> Path:
    """One second of video per entry in ``greys``, each a constant colour."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for grey in greys:
        for _ in range(fps):
            writer.write(np.full((64, 64, 3), grey, dtype=np.uint8))
    writer.release()
    return path


def _scripted(image, **_kw):
    return (Class3.ACTIVE_PLAY if int(image[32, 32, 0]) < 128 else Class3.EMPTY), 0.9


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    import tempfile

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(clip_review, "_classifier", lambda: _scripted)
    monkeypatch.setattr(clip_review, "_derive_boundary", lambda path: None)
    app = FastAPI()
    app.include_router(testing.router)
    return TestClient(app)


def _wait(client: TestClient, job_id: str, timeout: float = 120) -> dict:
    deadline = clock.monotonic() + timeout
    while clock.monotonic() < deadline:
        job = client.get(f"/api/v1/testing/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        clock.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish")


def _uploads(tmp_path: Path) -> list[Path]:
    return [p for p in tmp_path.iterdir() if p.suffix == ".upload"]


def test_a_day_test_runs_saves_and_deletes_the_footage(client, tmp_path) -> None:
    video = _video(tmp_path / "src.mp4", [DARK] * 60 + [LIGHT] * 60).read_bytes()
    r = client.post("/api/v1/testing/day?start=08:00&slot_minutes=1&interval_s=5"
                    "&field_id=field_01&day=2026-07-13&name=morning.mp4", content=video)
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["progress"]["done"] == job["progress"]["total"] > 0
    run = job["result"]
    assert [s["label"] for s in run["slots"]] == ["08:00-08:01", "08:01-08:02"]
    assert [s["status"] for s in run["slots"]] == ["USED", "NOT_USED"]
    assert run["counts"] == {"slots": 2, "USED": 1, "NOT_USED": 1, "REVIEW": 0}
    assert run["video_name"] == "morning.mp4" and run["boundaries"] == ["none"]

    assert not _uploads(tmp_path), "the footage was left on disk"
    saved = list((tmp_path / "data" / "test_runs").glob("*.json"))
    assert [p.stem for p in saved] == [run["id"]]

    listed = client.get("/api/v1/testing/runs").json()
    assert listed[0]["id"] == run["id"] and listed[0]["counts"]["USED"] == 1
    assert client.get(f"/api/v1/testing/runs/{run['id']}").json()["slots"] == run["slots"]

    csv = client.get(f"/api/v1/testing/runs/{run['id']}.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    lines = csv.text.strip().splitlines()
    assert len(lines) == 3 and "USED" in lines[1] and "NOT_USED" in lines[2]


def test_comparing_a_saved_run_with_an_uploaded_booking_file(client, tmp_path,
                                                            monkeypatch) -> None:
    def parse_csv_text(text: str, source: str = "upload"):
        path = tmp_path / "bookings.csv"
        path.write_text(text, encoding="utf-8")
        return read_bookings(path, source=source)

    fake = types.ModuleType("pitch_occupancy.booking_sources")
    fake.parse_csv_text = parse_csv_text
    fake.load_source = lambda path=None: None
    monkeypatch.setitem(sys.modules, "pitch_occupancy.booking_sources", fake)

    video = _video(tmp_path / "src.mp4", [DARK] * 60 + [LIGHT] * 60).read_bytes()
    job_id = client.post("/api/v1/testing/day?start=08:00&slot_minutes=1&interval_s=5",
                         content=video).json()["job_id"]
    run_id = _wait(client, job_id)["result"]["id"]

    sheet = ("field_id,date,start,end,customer_ref,status,entered_by\n"
             "field_01,2026-07-13,08:01,08:02,BK-9,confirmed,a\n")
    # no field id or date given and none on the run: refused with a reason
    assert client.post(f"/api/v1/testing/runs/{run_id}/compare", content=sheet
                       ).status_code == 422
    r = client.post(f"/api/v1/testing/runs/{run_id}/compare?field_id=field_01"
                    f"&day=2026-07-13", content=sheet)
    assert r.status_code == 200, r.text
    assert [c["outcome"] for c in r.json()["checks"]] == ["used_not_booked", "booked_not_used"]
    assert r.json()["checks"][1]["customer_refs"] == ["BK-9"]

    stored = client.get(f"/api/v1/testing/runs/{run_id}").json()
    assert stored["comparison"]["checks"][0]["outcome"] == "used_not_booked"
    assert "booked_not_used" in client.get(f"/api/v1/testing/runs/{run_id}.csv").text

    bad = client.post(f"/api/v1/testing/runs/{run_id}/compare?field_id=f&day=2026-07-13",
                      content="not,a,booking,file\n1,2,3,4\n")
    assert bad.status_code == 422
    none = client.post(f"/api/v1/testing/runs/{run_id}/compare?field_id=f&day=2026-07-13"
                       f"&use_configured=true")
    assert none.status_code == 404 and "configured" in none.json()["detail"]


def test_two_cameras_run_through_the_detector_pipeline(client, tmp_path, monkeypatch) -> None:
    class Detector:
        spec = DetectorSpec("scripted", "none.pt", False)

        def detect(self, image_bgr, **_):
            dark = int(image_bgr[32, 32, 0]) < 128
            found = [Detection(PERSON, (2 + 6 * i, 10, 6 + 6 * i, 40), 0.9)
                     for i in range(3 if dark else 0)]
            if dark:
                found.append(Detection(SPORTS_BALL, (30, 50, 34, 54), 0.5))
            return found

    def pipeline() -> Pipeline:
        cfg = RuleConfig(burst_frames=1, require_ball=False)
        classifier = DetectorFirstClassifier(Detector(), cfg)
        return Pipeline("scripted", classifier, require_boundary=False,
                        rules=cfg, classifier=classifier)

    monkeypatch.setattr(clip_pair, "_pipeline", pipeline)
    a = _video(tmp_path / "a.mp4", [DARK] * 60 + [LIGHT] * 60).read_bytes()
    b = _video(tmp_path / "b.mp4", [DARK] * 60 + [LIGHT] * 60).read_bytes()
    r = client.post(f"/api/v1/testing/day?start=08:00&slot_minutes=1&interval_s=10"
                    f"&size_a={len(a)}", content=a + b)
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job["error"]
    run = job["result"]
    assert run["cameras"] == 2
    # three people on each camera: six on the pitch, which is play; then nobody
    assert [s["status"] for s in run["slots"]] == ["USED", "NOT_USED"]
    assert not _uploads(tmp_path)


def test_an_unreadable_video_fails_the_job_and_is_still_deleted(client, tmp_path) -> None:
    r = client.post("/api/v1/testing/day?start=08:00", content=b"not a video at all")
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "failed" and job["error"]
    assert not _uploads(tmp_path)


def test_bad_requests_are_refused_and_leave_nothing(client, tmp_path) -> None:
    assert client.post("/api/v1/testing/day?start=8am", content=b"x").status_code == 422
    assert client.post("/api/v1/testing/day?start=08:00", content=b"").status_code == 422
    assert client.post("/api/v1/testing/day?start=08:00&slot_minutes=0",
                       content=b"x").status_code == 422
    r = client.post("/api/v1/testing/day?start=08:00&size_a=5", content=b"abc")
    assert r.status_code == 422
    assert not _uploads(tmp_path)


@pytest.mark.parametrize("run_id", ["..%2F..%2Fsecret", "..", "20260101-000000-abcdeg",
                                    "x" * 30])
def test_a_run_id_cannot_reach_outside_the_runs_folder(client, tmp_path, run_id) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "secret.json").write_text("{}", encoding="utf-8")
    for url in (f"/api/v1/testing/runs/{run_id}", f"/api/v1/testing/runs/{run_id}.csv"):
        assert client.get(url).status_code == 404
    assert client.get("/api/v1/testing/jobs/nope").status_code == 404


def test_the_pages_and_aliases(client) -> None:
    hub = client.get("/testing")
    assert hub.status_code == 200
    for href in ('href="/clip"', 'href="/clip/pair"', 'href="/images"',
                 'href="/testing/day"'):
        assert href in hub.text
    day = client.get("/testing/day")
    assert day.status_code == 200 and "/api/v1/testing/day" in day.text
    assert "Download CSV" in day.text and "Compare with bookings" in day.text
    for alias, target in (("/testing/video", "/clip"), ("/testing/two-cameras", "/clip/pair"),
                          ("/testing/images", "/images")):
        r = client.get(alias, follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"] == target


# --- judging rules --------------------------------------------------------------------------


def _fake_bookings(monkeypatch, tmp_path) -> None:
    def parse_csv_text(text: str, source: str = "upload"):
        path = tmp_path / "bookings.csv"
        path.write_text(text, encoding="utf-8")
        return read_bookings(path, source=source)

    fake = types.ModuleType("pitch_occupancy.booking_sources")
    fake.parse_csv_text = parse_csv_text
    fake.load_source = lambda path=None: None
    monkeypatch.setitem(sys.modules, "pitch_occupancy.booking_sources", fake)


def test_rule_options_are_validated_before_anything_is_received(client, tmp_path) -> None:
    for query in ("min_seen_pct=101", "close_low_pct=-1", "close_high_pct=200",
                  "close_low_pct=60&close_high_pct=40", "close_low_pct=50&close_high_pct=50",
                  "close_low_pct=70"):
        r = client.post(f"/api/v1/testing/day?start=08:00&{query}", content=b"x")
        assert r.status_code == 422, query
    r = client.post("/api/v1/testing/day?start=08:00&close_low_pct=70", content=b"x")
    assert "close-call band" in r.json()["detail"]
    assert not _uploads(tmp_path)


def test_rules_are_stored_shown_in_the_csv_and_a_run_can_be_rejudged(client, tmp_path,
                                                                     monkeypatch) -> None:
    _fake_bookings(monkeypatch, tmp_path)
    video = _video(tmp_path / "src.mp4", [DARK] * 60 + [LIGHT] * 60).read_bytes()
    r = client.post("/api/v1/testing/day?start=08:00&slot_minutes=1&interval_s=5"
                    "&min_seen_pct=75&close_low_pct=30&close_high_pct=70", content=video)
    run = _wait(client, r.json()["job_id"])["result"]
    assert run["rules"] == {"min_seen_share": 0.75, "close_low": 0.3, "close_high": 0.7}
    assert run["rules_text"] == "at least 75% seen; 30-70% is a close call"
    assert [s["status"] for s in run["slots"]] == ["USED", "NOT_USED"]
    lines = client.get(f"/api/v1/testing/runs/{run['id']}.csv").text.strip().splitlines()
    assert lines[0].endswith(",rules") and "at least 75% seen" in lines[1]

    sheet = ("field_id,date,start,end,customer_ref,status,entered_by\n"
             "field_01,2026-07-13,08:01,08:02,BK-9,confirmed,a\n")
    cmp = client.post(f"/api/v1/testing/runs/{run['id']}/compare?field_id=field_01"
                      f"&day=2026-07-13", content=sheet).json()
    assert [c["outcome"] for c in cmp["checks"]] == ["used_not_booked", "booked_not_used"]

    bad = client.post(f"/api/v1/testing/runs/{run['id']}/rejudge?close_low_pct=90"
                      f"&close_high_pct=80")
    assert bad.status_code == 422
    assert client.post("/api/v1/testing/runs/20260101-000000-abcdef/rejudge").status_code == 404
    # Playing 100% of slot 1 is now a close call; slot 2 (0%) is still not used.
    again = client.post(f"/api/v1/testing/runs/{run['id']}/rejudge?close_low_pct=90"
                        f"&close_high_pct=100")
    assert again.status_code == 200, again.text
    data = again.json()
    assert [s["status"] for s in data["slots"]] == ["REVIEW", "NOT_USED"]
    assert data["counts"] == {"slots": 2, "USED": 0, "NOT_USED": 1, "REVIEW": 1}
    assert data["rules"] == {"min_seen_share": 0.5, "close_low": 0.9, "close_high": 1.0}
    assert data["rejudged"]["slots_changed"] == 1
    assert data["rejudged"]["previous_rules_text"] == run["rules_text"]
    # The stored comparison follows the new results: a REVIEW slot is never accused.
    assert [c["outcome"] for c in data["comparison"]["checks"]] == ["review", "booked_not_used"]
    assert data["comparison"]["counts"] == {"review": 1, "booked_not_used": 1}
    stored = client.get(f"/api/v1/testing/runs/{run['id']}").json()
    assert stored["slots"] == data["slots"] and stored["comparison"] == data["comparison"]
    # Back to the defaults: the original results return exactly.
    back = client.post(f"/api/v1/testing/runs/{run['id']}/rejudge").json()
    assert [s["status"] for s in back["slots"]] == ["USED", "NOT_USED"]
    assert back["rules_text"] == "at least 50% seen; 40-60% is a close call"


def test_the_day_page_has_the_rule_options_and_prints(client) -> None:
    text = client.get("/testing/day").text
    for needle in ('id="minSeen"', 'id="closeLow"', 'id="closeHigh"', "@media print",
                   "window.print()", "Print / save as PDF", "/rejudge",
                   "Use a recording on this computer", "/api/v1/testing/recordings"):
        assert needle in text, needle


# --- big uploads and recordings already on this machine ------------------------------------


def test_the_upload_limit_comes_from_the_setting(client, monkeypatch) -> None:
    monkeypatch.setattr(settings, "testing_max_upload_mb", 1)
    r = client.post("/api/v1/testing/day?start=08:00", content=b"x" * (1024 * 1024 + 10))
    assert r.status_code == 413 and "limit is 1 MB" in r.json()["detail"]
    r = client.post("/api/v1/testing/day?start=08:00&size_a=5",
                    content=b"x" * (2 * 1024 * 1024 + 10))
    assert r.status_code == 413


def test_an_upload_that_would_not_fit_on_disk_is_refused(client, tmp_path,
                                                         monkeypatch) -> None:
    from pitch_occupancy import retention

    monkeypatch.setattr(retention, "free_bytes", lambda path: 1000)
    r = client.post("/api/v1/testing/day?start=08:00", content=b"x" * 5000)
    assert r.status_code == 507 and "disk space" in r.json()["detail"]
    assert not _uploads(tmp_path)


@pytest.fixture
def recordings(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "recordings"
    (folder / "day2").mkdir(parents=True)
    monkeypatch.setattr(settings, "recordings_dir", folder)
    return folder


def test_recordings_are_listed(client, recordings) -> None:
    _video(recordings / "morning.mp4", [DARK] * 4)
    _video(recordings / "day2" / "cam_b.mp4", [LIGHT] * 2)
    (recordings / "notes.txt").write_text("x", encoding="utf-8")
    data = client.get("/api/v1/testing/recordings").json()
    assert data["exists"] and data["folder"] == str(recordings)
    assert [r["name"] for r in data["recordings"]] == ["day2/cam_b.mp4", "morning.mp4"]
    assert data["recordings"][1]["duration_s"] == pytest.approx(4, abs=0.6)
    assert data["recordings"][1]["size_mb"] >= 0 and data["recordings"][1]["modified"]


def test_a_missing_recordings_folder_lists_nothing(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "recordings_dir", tmp_path / "nowhere")
    data = client.get("/api/v1/testing/recordings").json()
    assert data == {"folder": str(tmp_path / "nowhere"), "exists": False, "recordings": []}


def test_a_local_recording_is_analysed_and_kept(client, recordings) -> None:
    video = _video(recordings / "morning.mp4", [DARK] * 60 + [LIGHT] * 60)
    before = video.read_bytes()
    r = client.post("/api/v1/testing/day/local", json={
        "file_a": "morning.mp4", "start": "08:00", "slot_minutes": 1, "interval_s": 5,
        "close_low_pct": 30})
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["job_id"])
    assert job["status"] == "done", job["error"]
    run = job["result"]
    assert [s["status"] for s in run["slots"]] == ["USED", "NOT_USED"]
    assert run["video_name"] == "morning.mp4" and run["rules"]["close_low"] == 0.3
    assert video.exists() and video.read_bytes() == before, "a client's recording was touched"


@pytest.mark.parametrize("name", ["../secret.mp4", "..\\secret.mp4", "/etc/passwd",
                                  "C:/Windows/win.ini", "day2/../../secret.mp4",
                                  "notes.txt", "missing.mp4", ""])
def test_a_local_recording_must_be_a_video_inside_the_folder(client, recordings, tmp_path,
                                                             name) -> None:
    _video(tmp_path / "secret.mp4", [DARK] * 2)
    (recordings / "notes.txt").write_text("x", encoding="utf-8")
    r = client.post("/api/v1/testing/day/local", json={"file_a": name, "start": "08:00"})
    assert r.status_code in (404, 422), (name, r.status_code)
    assert (tmp_path / "secret.mp4").exists()


def test_a_link_out_of_the_recordings_folder_is_refused(client, recordings, tmp_path) -> None:
    target = _video(tmp_path / "secret.mp4", [DARK] * 2)
    link = recordings / "link.mp4"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("this system does not allow creating symlinks")
    r = client.post("/api/v1/testing/day/local", json={"file_a": "link.mp4", "start": "08:00"})
    assert r.status_code == 422
    names = [x["name"] for x in client.get("/api/v1/testing/recordings").json()["recordings"]]
    assert "link.mp4" not in names


# --- listing and cancelling tests -----------------------------------------------------------


def _wait_status(client: TestClient, job_id: str, *statuses: str, timeout: float = 30) -> dict:
    deadline = clock.monotonic() + timeout
    while clock.monotonic() < deadline:
        job = client.get(f"/api/v1/testing/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        clock.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {statuses}")


def _endless(monkeypatch) -> None:
    """Replace the one-video analysis with one that samples until it is cancelled."""

    def single(path, *, interval_s, camera, whole_frame, progress):
        assert Path(path).exists()
        for i in range(100_000):
            progress(i + 1, 100_000)
            clock.sleep(0.005)
        raise AssertionError("never cancelled")

    monkeypatch.setattr(testing, "_single", single)


def test_a_running_upload_can_be_cancelled_and_its_footage_is_deleted(client, tmp_path,
                                                                    monkeypatch) -> None:
    _endless(monkeypatch)
    r = client.post("/api/v1/testing/day?start=08:00&name=evening.mp4", content=b"video bytes")
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    deadline = clock.monotonic() + 30
    while client.get(f"/api/v1/testing/jobs/{job_id}").json()["progress"]["done"] == 0:
        assert clock.monotonic() < deadline
        clock.sleep(0.02)
    assert _uploads(tmp_path), "the upload should exist while the test runs"

    listed = client.get("/api/v1/testing/jobs").json()
    row = next(j for j in listed if j["job_id"] == job_id)
    assert row["status"] == "running" and row["label"] == "evening.mp4"
    assert "result" not in row

    r = client.post(f"/api/v1/testing/jobs/{job_id}/cancel")
    assert r.status_code == 200 and r.json()["cancel_requested"] is True
    job = _wait_status(client, job_id, "cancelled", "done", "failed")
    assert job["status"] == "cancelled"
    assert not _uploads(tmp_path), "a cancelled upload left its footage on disk"
    assert not list((tmp_path / "data" / "test_runs").glob("*.json")), "a run was saved"

    assert client.post(f"/api/v1/testing/jobs/{job_id}/cancel").status_code == 409
    assert client.post("/api/v1/testing/jobs/nope/cancel").status_code == 404


def test_a_queued_upload_cancelled_before_it_starts_is_deleted(client, tmp_path,
                                                             monkeypatch) -> None:
    from pitch_occupancy.api import jobs

    release = threading.Event()

    def blocker(progress):
        while not release.is_set():
            progress(0, 1)
            clock.sleep(0.01)
        return {}

    first = jobs.submit("day", blocker)
    try:
        _wait_status(client, first, "running")
        r = client.post("/api/v1/testing/day?start=08:00", content=b"queued video")
        job_id = r.json()["job_id"]
        assert client.get(f"/api/v1/testing/jobs/{job_id}").json()["status"] == "queued"
        assert _uploads(tmp_path)
        r = client.post(f"/api/v1/testing/jobs/{job_id}/cancel")
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        assert not _uploads(tmp_path), "a cancelled queued upload left its footage on disk"
    finally:
        release.set()
    _wait_status(client, first, "done")


def test_a_cancelled_local_recording_is_kept(client, recordings, monkeypatch) -> None:
    _endless(monkeypatch)
    video = _video(recordings / "morning.mp4", [DARK] * 4)
    before = video.read_bytes()
    job_id = client.post("/api/v1/testing/day/local", json={
        "file_a": "morning.mp4", "start": "08:00"}).json()["job_id"]
    deadline = clock.monotonic() + 30
    while client.get(f"/api/v1/testing/jobs/{job_id}").json()["progress"]["done"] == 0:
        assert clock.monotonic() < deadline
        clock.sleep(0.02)
    assert client.post(f"/api/v1/testing/jobs/{job_id}/cancel").status_code == 200
    assert _wait_status(client, job_id, "cancelled", "done", "failed")["status"] == "cancelled"
    assert video.exists() and video.read_bytes() == before, "a client's recording was touched"


def test_the_pages_offer_cancelling(client) -> None:
    hub = client.get("/testing").text
    assert "Running and queued" in hub and "/api/v1/testing/jobs" in hub
    assert "/cancel" in hub and "5000" in hub
    day = client.get("/testing/day").text
    assert 'id="cancel"' in day and "/cancel" in day
