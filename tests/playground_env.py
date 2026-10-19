"""A playground test setup shared by `test_playgrounds.py` and `test_simple_pages.py`.

Four saved cameras (``site/a``, ``b``, ``c`` seeing 3, 5 and 4 people; ``site/dead`` never
answering), a temporary database and playground files, and no camera contacted or model
loaded. Each seam is patched on the module that reads it (`playgrounds.check._grab`, ...).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import cameras, playgrounds
from pitch_occupancy.api import cameras_page, results_page, simple_pages
from pitch_occupancy.config import settings

SECRET = "s3cretPW"
FRAME = np.full((360, 640, 3), 90, np.uint8)


class CountingPipeline:
    """Reports a scripted number of people per camera, as the detector would."""

    burst = (1, 0.0)

    def __init__(self, people: dict[str, int]) -> None:
        self.people = people
        self.calls: list[str] = []

    def observe_minute(self, camera, frames, previous=None, *, polygon=None, venue=None,
                       slot_key=None):
        self.calls.append(camera)
        n = self.people.get(camera, 0)
        verdict = SimpleNamespace(state="C1_EMPTY" if n == 0 else "C2_ACTIVE_PLAY",
                                  confidence=0.9, people=n, ball=False, trace=())
        return SimpleNamespace(verdict=verdict, evidence_index=0, instances=())


def make_env(tmp_path, monkeypatch) -> SimpleNamespace:
    cam_file = tmp_path / "cameras.json"
    cam_file.write_text(json.dumps({"site": {
        name: f"rtsp://admin:{SECRET}@10.0.0.{i}:554/1/1"
        for i, name in enumerate(("a", "b", "c", "dead"), start=5)
    }}), encoding="utf-8")
    monkeypatch.setattr(cameras, "CONFIG_PATH", cam_file)
    monkeypatch.setattr(settings, "db_path", tmp_path / "db" / "pitch.db")
    monkeypatch.setattr(playgrounds.store, "CONFIG_PATH", tmp_path / "playgrounds.json")
    monkeypatch.setattr(playgrounds.capacity, "SPEED_PATH", tmp_path / "speed.json")
    monkeypatch.setattr(playgrounds.snapshots, "SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(playgrounds.snapshots, "_snap_bytes", None)
    monkeypatch.setattr(playgrounds.check, "play_min", lambda: 5)
    pipeline = CountingPipeline({"a": 3, "b": 5, "c": 4})
    monkeypatch.setattr(playgrounds.check, "_pipeline", lambda: pipeline)
    monkeypatch.setattr(cameras_page, "_pipeline", lambda: pipeline)

    def grab(url, timeout_s):
        if "10.0.0.8" in url:  # the "dead" camera
            return None, "The camera did not answer.", 3000
        return FRAME, None, 120

    monkeypatch.setattr(playgrounds.check, "_grab", grab)
    app = FastAPI()
    app.include_router(cameras_page.router)
    app.include_router(simple_pages.router)
    app.include_router(results_page.router)
    return SimpleNamespace(client=TestClient(app), tmp=tmp_path, pipeline=pipeline)


def new_playground(name="Playground 1", cams=("site/a",), every=30, counting="add",
                   **kw) -> dict:
    return playgrounds.save_playground({"name": name, "cameras": list(cams),
                                        "every_seconds": every, "counting": counting, **kw})
