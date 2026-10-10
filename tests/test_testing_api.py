"""The booking-day test over HTTP: upload, background job, saved run, CSV, bookings.

The classifier is scripted by frame colour - dark frames are play, light frames are an empty
pitch - so a two-minute generated video is one used slot and one unused one, and the route's
arithmetic can be checked by hand. What is pinned beyond the numbers is what the page promises
the client: the footage is gone once the job ends, only results are saved, and a run id
cannot be used to read anything but a run.
"""

from __future__ import annotations

import sys
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
    monkeypatch.setattr(clip_review, "_gates", lambda: (None, None))
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
        return Pipeline("scripted", "detector", classifier, require_boundary=False,
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
