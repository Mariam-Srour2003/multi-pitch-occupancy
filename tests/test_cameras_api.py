"""The `/cameras` page's API, on a fresh app with every camera, file and model faked.

The capture opener and the TCP connect are replaced so no camera is ever contacted, the
pipeline is scripted because the real detector cannot load in this environment, and every
path the routes write to is a temporary one. Each test that sees a response also checks the
saved password is not in it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import cameras
from pitch_occupancy.api import cameras_page
from pitch_occupancy.config import settings
from pitch_occupancy.vision import roi

SECRET = "s3cretPW"


class FakeCapture:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def isOpened(self) -> bool:  # noqa: N802
        return self.ok

    def read(self):
        return (True, np.full((360, 640, 3), 90, np.uint8)) if self.ok else (False, None)

    def grab(self) -> bool:
        return self.ok

    def get(self, prop) -> float:
        return 25.0

    def set(self, *args) -> bool:
        return True

    def release(self) -> None:
        pass


class ScriptedPipeline:
    """Counts scripted per camera: people and ball, as the detector path would report."""

    model_key = "scripted"
    burst = (3, 0.0)

    def __init__(self, counts: dict[str, tuple[int, bool]]) -> None:
        self.counts = counts
        self.seen: list[tuple[str, int, object]] = []

    def observe_minute(self, camera, frames, previous=None, *, polygon=None, venue=None,
                       slot_key=None):
        self.seen.append((camera, len(frames), polygon))
        people, ball = self.counts[camera]
        verdict = SimpleNamespace(state="C1_EMPTY", confidence=0.9, people=people, ball=ball,
                                  trace=(f"{camera}: {people} people",))
        return SimpleNamespace(verdict=verdict, evidence_index=len(frames) - 1, instances=())

    def fuse(self, observations):
        people = sum(o.verdict.people for o in observations.values())
        ball = any(o.verdict.ball for o in observations.values())
        state = "C2_ACTIVE_PLAY" if people > 4 and ball else "C3_MAINTENANCE_NON_SPORTING"
        return SimpleNamespace(state=state, confidence=0.8, people_inside=people,
                               ball_seen=ball, trace=(f"{people} people, ball {ball}",))


@pytest.fixture
def env(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    cam_file = configs / "cameras.json"
    cam_file.write_text(json.dumps({
        "_comment": ["test"],
        "venue_01": {
            "camera_A": f"rtsp://admin:{SECRET}@10.0.0.5:554/Streaming/Channels/102",
            "camera_B": f"rtsp://admin:{SECRET}@10.0.0.6:554/Streaming/Channels/102",
        },
    }), encoding="utf-8")
    schedule = configs / "slots_schedule.json"
    schedule.write_text(json.dumps([
        {"venue_id": "venue_01", "field_id": "field_01", "start": "10:00",
         "duration_minutes": 60, "cameras": ["camera_A", "camera_B"], "days": []},
        {"venue_id": "venue_02", "field_id": "field_09", "start": "10:00",
         "duration_minutes": 60, "cameras": ["cam1"], "days": ["sat"]},
    ]), encoding="utf-8")
    monkeypatch.setattr(cameras, "CONFIG_PATH", cam_file)
    monkeypatch.setattr(cameras_page, "SCHEDULE_PATH", schedule)
    # No database yet, unless a test makes one: never the real one.
    monkeypatch.setattr(settings, "db_path", tmp_path / "db" / "pitch_monitor.db")

    opened: list[str] = []
    broken: set[str] = set()

    def open_capture(url, timeout_ms=5000):
        opened.append(url)
        return FakeCapture(ok=not any(b in url for b in broken))

    monkeypatch.setattr(cameras, "open_capture", open_capture)
    monkeypatch.setattr(cameras, "_tcp_connect", lambda host, port, timeout: None)
    boundaries = {"camera_A": [[0, 0], [1, 0], [1, 1]]}
    monkeypatch.setattr(roi, "resolve_with_key", lambda camera, venue=None, slot_key=None: (
        (boundaries[camera], f"{venue}/{camera}") if camera in boundaries else (None, None)))
    pipeline = ScriptedPipeline({"camera_A": (3, False), "camera_B": (3, True)})
    monkeypatch.setattr(cameras_page, "_pipeline", lambda: pipeline)

    app = FastAPI()
    app.include_router(cameras_page.router)
    return SimpleNamespace(client=TestClient(app), cam_file=cam_file, schedule=schedule,
                           opened=opened, broken=broken, pipeline=pipeline)


def _clean(response) -> dict:
    assert SECRET not in response.text, "a saved password leaked into a response"
    return response.json()


# --- page and registry --------------------------------------------------------------------

def test_page_renders_all_four_sections(env) -> None:
    r = env.client.get("/cameras")
    assert r.status_code == 200
    for text in ("Your cameras", "What is happening now?", "Start recording a day",
                 "How to connect a camera", "Streaming/Channels/101", "usb:0", "VLC"):
        assert text in r.text
    assert SECRET not in r.text


def test_list_is_masked(env) -> None:
    data = _clean(env.client.get("/api/v1/cameras/registry"))
    assert [c["camera_id"] for c in data["cameras"]] == ["camera_A", "camera_B"]
    assert data["cameras"][0]["url"].startswith("rtsp://admin:***@10.0.0.5")
    assert data["cameras"][0]["boundary"] == "venue_01/camera_A"
    assert data["cameras"][1]["boundary"] is None


def test_save_with_blank_password_keeps_the_stored_one(env) -> None:
    r = env.client.put("/api/v1/cameras", json={
        "venue": "venue_01", "camera_id": "camera_A",
        "url": "rtsp://admin:***@10.0.0.5:554/Streaming/Channels/101"})
    data = _clean(r)
    assert data["camera"]["url"] == "rtsp://admin:***@10.0.0.5:554/Streaming/Channels/101"
    stored = json.loads(env.cam_file.read_text(encoding="utf-8"))
    assert stored["venue_01"]["camera_A"] == \
        f"rtsp://admin:{SECRET}@10.0.0.5:554/Streaming/Channels/101"
    assert stored["_comment"] == ["test"]


def test_save_refuses_bad_input(env) -> None:
    r = env.client.put("/api/v1/cameras", json={"venue": "venue_01", "camera_id": "bad id",
                                                "url": "usb:0"})
    assert r.status_code == 422
    r = env.client.put("/api/v1/cameras", json={"venue": "venue_01", "camera_id": "c",
                                                "url": "ftp://x/y"})
    assert r.status_code == 422


def test_delete_warns_about_scheduled_slots(env) -> None:
    data = _clean(env.client.delete("/api/v1/cameras/venue_01/camera_B"))
    assert data["ok"] and "camera_B" in data["warning"]
    assert env.client.delete("/api/v1/cameras/venue_01/camera_B").status_code == 404


# --- test ---------------------------------------------------------------------------------

def test_testing_a_saved_camera(env) -> None:
    data = _clean(env.client.post("/api/v1/cameras/test",
                                  json={"venue": "venue_01", "camera_id": "camera_A"}))
    assert data["ok"] and data["stage"] == "ok" and data["width"] == 640
    assert data["snapshot"] and data["url"].startswith("rtsp://admin:***@")
    assert env.opened == [f"rtsp://admin:{SECRET}@10.0.0.5:554/Streaming/Channels/102"]


def test_testing_an_unsaved_url(env) -> None:
    env.broken.add("10.0.0.99")
    data = env.client.post("/api/v1/cameras/test",
                           json={"url": "rtsp://u:other@10.0.0.99/x"}).json()
    assert data["stage"] == "stream" and data["snapshot"] is None
    assert "other" not in json.dumps(data)


def test_testing_an_edit_uses_the_stored_password(env) -> None:
    _clean(env.client.post("/api/v1/cameras/test", json={
        "venue": "venue_01", "camera_id": "camera_A",
        "url": "rtsp://admin:***@10.0.0.5:554/Streaming/Channels/101"}))
    assert env.opened[-1] == f"rtsp://admin:{SECRET}@10.0.0.5:554/Streaming/Channels/101"


def test_test_needs_something_to_test(env) -> None:
    assert env.client.post("/api/v1/cameras/test", json={}).status_code == 422
    assert env.client.post("/api/v1/cameras/test",
                           json={"venue": "venue_01", "camera_id": "nope"}).status_code == 404


# --- predict ------------------------------------------------------------------------------

def test_predict_adds_people_and_shares_the_ball(env) -> None:
    data = _clean(env.client.post("/api/v1/cameras/predict", json={"venue": "venue_01"}))
    assert data["pitch"]["state"] == "C2_ACTIVE_PLAY"
    assert data["pitch"]["label"] == "Playing"
    assert (data["pitch"]["people"], data["pitch"]["ball"]) == (6, True)
    assert data["recorded"] is False
    by_cam = {c["camera_id"]: c for c in data["cameras"]}
    assert by_cam["camera_A"]["people"] == 3 and by_cam["camera_B"]["ball"] is True
    assert all(c["snapshot"] for c in data["cameras"])
    # A burst per camera, camera_A inside its stored boundary, camera_B on the whole frame.
    assert [(cam, n) for cam, n, _ in env.pipeline.seen] == [("camera_A", 3), ("camera_B", 3)]
    assert env.pipeline.seen[0][2] is not None and env.pipeline.seen[1][2] is None
    assert any("camera_B has no pitch boundary" in n for n in data["notes"])


def test_predict_one_camera_and_a_dead_one(env) -> None:
    env.broken.add("10.0.0.6")
    data = _clean(env.client.post("/api/v1/cameras/predict", json={"venue": "venue_01"}))
    by_cam = {c["camera_id"]: c for c in data["cameras"]}
    assert by_cam["camera_B"]["readable"] is False and by_cam["camera_B"]["problem"]
    assert data["pitch"]["people"] == 3
    assert any("part of the pitch" in n for n in data["notes"])

    data = _clean(env.client.post("/api/v1/cameras/predict",
                                  json={"venue": "venue_01", "cameras": ["camera_A"]}))
    assert [c["camera_id"] for c in data["cameras"]] == ["camera_A"]


def test_predict_without_a_detector_is_a_verdict_not_a_500(env, monkeypatch) -> None:
    def broken():
        raise RuntimeError("torchvision::nms does not exist")
    monkeypatch.setattr(cameras_page, "_pipeline", broken)
    r = env.client.post("/api/v1/cameras/predict", json={"venue": "venue_01"})
    data = _clean(r)
    assert r.status_code == 200
    assert data["pitch"]["state"] == "UNAVAILABLE"
    assert "detector could not be loaded" in data["pitch"]["trace"][0]
    assert all(c["snapshot"] for c in data["cameras"])


def test_predict_unknown_venue_or_camera(env) -> None:
    assert env.client.post("/api/v1/cameras/predict",
                           json={"venue": "nowhere"}).status_code == 404
    assert env.client.post("/api/v1/cameras/predict",
                           json={"venue": "venue_01", "cameras": ["x"]}).status_code == 422


# --- schedule -----------------------------------------------------------------------------

BODY = {"venue_id": "venue_01", "field_id": "field_01", "cameras": ["camera_A", "camera_B"],
        "start": "08:00", "slot_minutes": 90, "count": 6, "days": ["mon", "tue"]}


def test_plan_slots() -> None:
    slots, left = cameras_page.plan_slots("08:00", 90, count=3)
    assert slots == [(480, 570), (570, 660), (660, 750)] and left == 0
    slots, left = cameras_page.plan_slots("08:00", 90, end="23:00")
    assert len(slots) == 10 and left == 0
    slots, left = cameras_page.plan_slots("08:00", 90, end="10:00")
    assert len(slots) == 1 and left == 30
    assert len(cameras_page.plan_slots("22:30", 90, end="00:00")[0]) == 1
    for kwargs in ({"count": 4, "start": "20:00"}, {"end": "07:00", "start": "08:00"},
                   {"start": "25:00", "count": 1}, {"start": "08:00"}):
        start = kwargs.pop("start")
        with pytest.raises(ValueError):
            cameras_page.plan_slots(start, 90, **kwargs)


def test_preview_writes_nothing(env) -> None:
    before = env.schedule.read_text(encoding="utf-8")
    data = _clean(env.client.post("/api/v1/cameras/schedule/preview", json=BODY))
    assert data["ok"], data["problems"]
    assert [(s["start"], s["end"]) for s in data["slots"]] == [
        ("08:00", "09:30"), ("09:30", "11:00"), ("11:00", "12:30"),
        ("12:30", "14:00"), ("14:00", "15:30"), ("15:30", "17:00")]
    assert (data["replaced"], data["kept"]) == (1, 1)
    assert "--source live" in data["commands"]["start"]
    assert any("camera_B has no pitch boundary" in w for w in data["warnings"])
    assert env.schedule.read_text(encoding="utf-8") == before


def test_save_merges_and_keeps_other_pitches(env) -> None:
    data = _clean(env.client.put("/api/v1/cameras/schedule", json=BODY))
    assert data["saved"] and data["previous_version"]
    entries = json.loads(env.schedule.read_text(encoding="utf-8"))
    ours = [e for e in entries if e["venue_id"] == "venue_01"]
    assert [e["start"] for e in ours] == ["08:00", "09:30", "11:00", "12:30", "14:00", "15:30"]
    assert all(e["duration_minutes"] == 90 and e["days"] == ["mon", "tue"] for e in ours)
    assert [e["venue_id"] for e in entries if e["venue_id"] != "venue_01"] == ["venue_02"]
    from pitch_occupancy.scheduler import load_schedule
    assert len(load_schedule(env.schedule).slots) == 7
    assert list((env.schedule.parent / "schedule_history").glob("*.json"))


def test_overlap_with_another_pitch_sharing_a_camera_is_refused(env) -> None:
    entries = json.loads(env.schedule.read_text(encoding="utf-8"))
    entries.append({"venue_id": "venue_01", "field_id": "field_02", "start": "09:00",
                    "duration_minutes": 60, "cameras": ["camera_A"], "days": ["mon"]})
    env.schedule.write_text(json.dumps(entries), encoding="utf-8")
    before = env.schedule.read_text(encoding="utf-8")

    data = env.client.post("/api/v1/cameras/schedule/preview", json=BODY).json()
    assert not data["ok"] and any("overlaps" in p for p in data["problems"])
    r = env.client.put("/api/v1/cameras/schedule", json=BODY)
    assert r.status_code == 422 and "overlaps" in r.json()["detail"]
    assert env.schedule.read_text(encoding="utf-8") == before

    # On days the other pitch does not run, there is no clash.
    data = env.client.post("/api/v1/cameras/schedule/preview",
                           json={**BODY, "days": ["sat"]}).json()
    assert data["ok"], data["problems"]


def test_crossing_midnight_and_unknown_cameras_are_refused(env) -> None:
    r = env.client.post("/api/v1/cameras/schedule/preview",
                        json={**BODY, "start": "20:00", "count": 4})
    assert r.status_code == 422 and "midnight" in r.json()["detail"]
    data = env.client.post("/api/v1/cameras/schedule/preview",
                           json={**BODY, "cameras": ["camera_Z"]}).json()
    assert not data["ok"] and "camera_Z" in data["problems"][0]
    assert env.client.put("/api/v1/cameras/schedule",
                          json={**BODY, "cameras": ["camera_Z"]}).status_code == 422


def test_end_time_mode(env) -> None:
    body = {**BODY, "count": None, "end": "23:00"}
    data = env.client.post("/api/v1/cameras/schedule/preview", json=body).json()
    assert len(data["slots"]) == 10 and data["slots"][-1]["end"] == "23:00"


# --- the daily check ----------------------------------------------------------------------

def test_test_all_tries_every_camera_and_says_how_many_answered(env, monkeypatch) -> None:
    timeouts: list[int] = []
    real_probe = cameras.probe

    def probe(url, **kw):
        timeouts.append(kw["timeout_ms"])
        return real_probe(url, **kw)

    monkeypatch.setattr(cameras, "probe", probe)
    env.broken.add("10.0.0.6")
    data = _clean(env.client.post("/api/v1/cameras/test-all"))
    assert data["summary"] == "1 of 2 cameras answered"
    assert (data["answered"], data["total"]) == (1, 2)
    by_cam = {c["camera_id"]: c for c in data["cameras"]}
    assert by_cam["camera_A"]["ok"] and by_cam["camera_A"]["width"] == 640
    assert by_cam["camera_B"]["stage"] == "stream" and by_cam["camera_B"]["hint"]
    assert all(c["url"].startswith("rtsp://admin:***@") for c in data["cameras"])
    assert all("snapshot" not in c and "frame" not in c for c in data["cameras"])
    assert {"venue", "elapsed_ms", "message", "height"} <= set(by_cam["camera_A"])
    assert timeouts == [cameras_page.TEST_ALL_TIMEOUT_MS] * 2
    assert len(env.opened) == 2


def test_test_all_with_no_cameras(env) -> None:
    env.cam_file.write_text("{}", encoding="utf-8")
    data = env.client.post("/api/v1/cameras/test-all").json()
    assert data["total"] == 0 and data["summary"] == "No cameras saved yet"


def test_no_database_yet_means_never_recorded_and_creates_nothing(env) -> None:
    data = _clean(env.client.get("/api/v1/cameras/registry"))
    rec = data["cameras"][0]["recording"]
    assert rec["last_recorded"] is None and rec["captured_24h"] == 0
    assert rec["scheduled"] is True and rec["stale"] is True  # scheduled, never recorded
    health = env.client.get("/api/v1/cameras/health").json()
    assert health["database_exists"] is False
    assert [c["last_recorded"] for c in health["cameras"]] == [None, None]
    assert not settings.db_path.exists(), "looking must not create the database"


def test_last_picture_comes_from_the_database(env) -> None:
    from datetime import UTC, datetime, timedelta

    from pitch_occupancy.db.schema import connect, initialise
    from pitch_occupancy.db.store import ensure_slot

    conn = connect(settings.db_path)
    initialise(conn)
    now = datetime.now()
    begin = now - timedelta(minutes=30)
    ensure_slot(conn, slot_id="s1", venue_id="venue_01", field_id="field_01",
                cameras=["camera_A"], slot_date=begin.date().isoformat(),
                start_time=f"{begin:%H:%M}", end_time=f"{now:%H:%M}")
    # camera_B is known to the database at another venue: not this registry's camera_B
    ensure_slot(conn, slot_id="s2", venue_id="venue_09", field_id="field_09",
                cameras=["camera_B"], slot_date=begin.date().isoformat(),
                start_time=f"{begin:%H:%M}", end_time=f"{now:%H:%M}")
    utc = datetime.now(UTC)
    for i, cam, slot in ((0, "camera_A", "s1"), (1, "camera_A", "s1"), (0, "camera_B", "s2")):
        stamp = (utc - timedelta(minutes=3 + i)).isoformat(timespec="seconds")
        conn.execute("INSERT INTO frame_samples (slot_id, camera_id, captured_at, "
                     "minute_index, predicted, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                     (slot, cam, stamp, i, "C1_EMPTY", 0.9))
    conn.commit()
    conn.close()

    by_cam = {c["camera_id"]: c["recording"]
              for c in _clean(env.client.get("/api/v1/cameras/registry"))["cameras"]}
    a = by_cam["camera_A"]
    assert 2 <= a["minutes_since"] <= 4 and a["stale"] is False
    assert a["captured_24h"] == 2 and 29 <= a["expected_24h"] <= 31
    b = by_cam["camera_B"]
    assert b["last_recorded"] is None and "venue_09" in b["note"] and b["stale"] is True


def test_the_page_has_the_daily_check(env) -> None:
    text = env.client.get("/cameras").text
    for needle in ("Test all cameras", "Last picture", "/api/v1/cameras/test-all",
                   "answered in", "never"):
        assert needle in text
