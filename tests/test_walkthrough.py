"""Stepping through a clip with the detector's drawing.

The claim most easily broken silently is that the upload is deleted. It is harder here than
on `/clip/analyse`: a StreamingResponse's body runs *after* the handler returns, so the
obvious `finally` would remove the file before a single frame had been read.

The classifier is scripted, so nothing here loads the detector.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from pitch_occupancy.api import clip_walkthrough as cw
from pitch_occupancy.api.app import app
from pitch_occupancy.data.taxonomy import Class3

P = Class3.ACTIVE_PLAY


def _video(path: Path, seconds: int = 12, fps: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for i in range(seconds * fps):
        writer.write(np.full((64, 64, 3), i % 256, dtype=np.uint8))
    writer.release()
    return path


class _StubClassifier:
    def __init__(self) -> None:
        self.previous: list[object] = []

    def __call__(self, image_bgr, *, polygon=None, previous=None, **_kw):
        self.previous.append(previous)
        return P, 0.9


def _stub_records(client, video: bytes, query: str = "?interval_s=2&explain_n=0"):
    with client.stream("POST", "/api/v1/clip/walkthrough" + query, content=video,
                       headers={"Content-Type": "application/octet-stream"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in response.iter_lines() if line.strip()]


def test_frames_are_sampled_at_the_interval_and_never_past_the_cap(tmp_path) -> None:
    frames = list(cw.sample_frames(_video(tmp_path / "c.mp4"), 2.0))
    assert [i for i, _, _ in frames] == list(range(len(frames)))
    assert [round(t) for _, t, _ in frames][:3] == [0, 2, 4]
    assert len(list(cw.sample_frames(tmp_path / "c.mp4", 0.5, max_samples=5))) == 5


def test_each_step_gets_the_previous_sample_for_the_motion_cue(tmp_path, monkeypatch) -> None:
    stub = _StubClassifier()
    monkeypatch.setattr("pitch_occupancy.api.clip_review._classifier", lambda: stub)
    walked = cw.walk_records(_video(tmp_path / "c.mp4"), interval_s=2.0, explain_n=0,
                             redact=False, whole_frame=True)
    records = [json.loads(r) for r in walked]
    assert stub.previous[0] is None
    assert all(p is not None for p in stub.previous[1:])
    assert records[0]["type"] == "meta" and records[0]["detector"]


def test_the_stream_is_newline_delimited_json_with_a_meta_and_a_done(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("pitch_occupancy.api.clip_review._classifier",
                        lambda: _StubClassifier())
    records = _stub_records(TestClient(app), _video(tmp_path / "c.mp4").read_bytes())
    assert records[0]["type"] == "meta"
    assert records[-1]["type"] == "done"
    steps = [r for r in records if r["type"] == "step"]
    assert steps and records[-1]["n"] == len(steps)
    assert all("clock" in s and "predicted" in s for s in steps)
    assert not any("boxes" in s for s in steps), "explain_n=0 draws nothing"


def test_the_walkthrough_deletes_the_upload_after_streaming(tmp_path, monkeypatch) -> None:
    """Named in `ALLOWED_MUTATING`'s reason. The `finally` has to live in the generator."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr("pitch_occupancy.api.clip_review._classifier",
                        lambda: _StubClassifier())
    video = _video(tmp_path / "src.mp4").read_bytes()

    records = _stub_records(TestClient(app), video)

    assert [r for r in records if r["type"] == "step"], "nothing streamed"
    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".upload"]
    assert not leftover, f"the upload was left on disk: {sorted(p.name for p in leftover)}"


def test_an_empty_body_is_refused_before_anything_streams() -> None:
    response = TestClient(app).post("/api/v1/clip/walkthrough", content=b"",
                                    headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 422


def test_an_unreadable_upload_reports_an_error_record(tmp_path, monkeypatch) -> None:
    """The stream has already begun by the time the decode fails, so it cannot become a 422.
    It becomes a record the page can render instead of a truncated stream."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr("pitch_occupancy.api.clip_review._classifier",
                        lambda: _StubClassifier())
    monkeypatch.setattr("pitch_occupancy.api.limits.check_video_pixels",
                        lambda *a, **k: None)
    records = _stub_records(TestClient(app), b"definitely not a video")
    assert records[-1]["type"] == "error"
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".upload"]


def test_the_page_offers_the_walkthrough_and_is_honest_about_the_button() -> None:
    """A reader who thinks the skip button sped the model up has learnt something false about
    the system. The page has to say what it actually does."""
    text = TestClient(app).get("/clip").text
    assert "Watch it work" in text
    assert "Continue without slow motion" in text
    assert "does not make the model" in text
    assert "What the detector found" in text
