"""The two-camera reviewer: both videos, one pitch, the rule applied to the two together.

What is pinned here is what the page claims about crossing the cameras - people added, the
ball from either side, the overlap check - and that neither upload survives the request. The
detector is scripted by frame colour, so each "camera" shows a fixed scene and the arithmetic
can be checked by hand.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api import clip_pair
from pitch_occupancy.api.app import app
from pitch_occupancy.pair_analysis import analyse_pair
from pitch_occupancy.pipeline import Pipeline
from pitch_occupancy.vision.detector import PERSON, SPORTS_BALL, Detection, DetectorSpec
from pitch_occupancy.vision.pitch_classifier import DetectorFirstClassifier
from pitch_occupancy.vision.rules import MinuteState, RuleConfig

#: Frame grey level -> (people, ball). Each test video is one constant colour.
SCENES = {40: (3, False), 80: (2, True), 120: (0, False), 160: (6, True), 200: (1, False)}


def _video(path: Path, grey: int, seconds: int = 30, fps: int = 5) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for _ in range(seconds * fps):
        writer.write(np.full((64, 64, 3), grey, dtype=np.uint8))
    writer.release()
    return path


class ScriptedDetector:
    spec = DetectorSpec("scripted", "none.pt", False)

    def detect(self, image_bgr, **_):
        grey = int(image_bgr[32, 32, 0])
        key = min(SCENES, key=lambda g: abs(g - grey))  # survives codec rounding
        people, ball = SCENES[key]
        found = [Detection(PERSON, (2 + 6 * i, 10, 6 + 6 * i, 40), 0.9) for i in range(people)]
        if ball:
            found.append(Detection(SPORTS_BALL, (30, 50, 34, 54), 0.5))
        return found


def _pipeline(**rules) -> Pipeline:
    cfg = RuleConfig(burst_frames=1, require_ball=False, **rules)
    classifier = DetectorFirstClassifier(ScriptedDetector(), cfg)
    return Pipeline("scripted", "detector", classifier, require_boundary=False, rules=cfg,
                    classifier=classifier)


def test_people_on_both_cameras_are_added_before_the_rule(tmp_path) -> None:
    """Three on A and two on B is five on the pitch: play. Each camera alone says not."""
    a = _video(tmp_path / "a.mp4", 40)
    b = _video(tmp_path / "b.mp4", 80)
    result = analyse_pair(a, b, _pipeline(), interval_s=10)

    assert result.samples
    for s in result.samples:
        assert [side.people for side in s.sides] == [3, 2]
        assert {side.state for side in s.sides} == {MinuteState.MAINTENANCE_NON_SPORTING}
        assert s.people == 5
        assert s.raw is MinuteState.ACTIVE_PLAY


def test_a_ball_on_one_side_is_the_pitchs_ball(tmp_path) -> None:
    a = _video(tmp_path / "a.mp4", 40)
    b = _video(tmp_path / "b.mp4", 80)
    s = analyse_pair(a, b, _pipeline(), interval_s=10).samples[0]
    assert s.ball is True
    assert s.ball_sides == ("B",)
    assert s.ball_one_side


def test_requiring_the_ball_is_satisfied_by_either_camera(tmp_path) -> None:
    """With `require_ball` on, A has no ball and B has one: the pitch has one, so play."""
    a = _video(tmp_path / "a.mp4", 40)
    b = _video(tmp_path / "b.mp4", 80)
    cfg = RuleConfig(burst_frames=1, require_ball=True)
    classifier = DetectorFirstClassifier(ScriptedDetector(), cfg)
    pipeline = Pipeline("scripted", "detector", classifier, require_boundary=False,
                        rules=cfg, classifier=classifier)
    s = analyse_pair(a, b, pipeline, interval_s=10).samples[0]
    assert s.raw is MinuteState.ACTIVE_PLAY


def test_a_verdict_that_only_the_sum_reaches_is_flagged(tmp_path) -> None:
    """If the views overlapped completely the count would be max(3, 2) = 3: not play."""
    a = _video(tmp_path / "a.mp4", 40)
    b = _video(tmp_path / "b.mp4", 80)
    s = analyse_pair(a, b, _pipeline(), interval_s=10).samples[0]
    assert s.if_overlapping is MinuteState.MAINTENANCE_NON_SPORTING
    assert s.depends_on_sum


def test_one_camera_already_enough_is_not_flagged(tmp_path) -> None:
    a = _video(tmp_path / "a.mp4", 160)
    b = _video(tmp_path / "b.mp4", 120)
    s = analyse_pair(a, b, _pipeline(), interval_s=10).samples[0]
    assert s.raw is MinuteState.ACTIVE_PLAY
    assert not s.depends_on_sum
    assert s.split_view  # A plays, B is empty: the half-pitch case fusion exists for


def test_both_empty_is_empty_and_one_person_anywhere_is_not(tmp_path) -> None:
    empty = analyse_pair(_video(tmp_path / "a.mp4", 120), _video(tmp_path / "b.mp4", 120),
                         _pipeline(), interval_s=10).samples[0]
    assert empty.raw is MinuteState.EMPTY
    one = analyse_pair(_video(tmp_path / "c.mp4", 120), _video(tmp_path / "d.mp4", 200),
                       _pipeline(), interval_s=10).samples[0]
    assert one.raw is MinuteState.MAINTENANCE_NON_SPORTING


def test_the_offset_samples_only_where_both_files_run(tmp_path) -> None:
    """B starts 10 s after A and both are 30 s long: A's clock covers 10-30 s."""
    a = _video(tmp_path / "a.mp4", 40)
    b = _video(tmp_path / "b.mp4", 80)
    result = analyse_pair(a, b, _pipeline(), offset_s=10, interval_s=5)
    assert result.start_s == pytest.approx(10)
    assert all(10 <= s.t_s < 30 for s in result.samples)
    with pytest.raises(ValueError, match="do not overlap"):
        analyse_pair(a, b, _pipeline(), offset_s=40)


# --- the route ------------------------------------------------------------------------------


def test_the_pair_route_keeps_neither_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(clip_pair, "_pipeline", _pipeline)
    monkeypatch.setattr(clip_pair, "_derive_boundary", lambda path: None)
    made: list[Path] = []
    real = clip_pair.tempfile.mkstemp

    def tracking(*args, **kwargs):
        fd, name = real(*args, **kwargs)
        made.append(Path(name))
        return fd, name

    monkeypatch.setattr(clip_pair.tempfile, "mkstemp", tracking)
    a = _video(tmp_path / "a.mp4", 40).read_bytes()
    b = _video(tmp_path / "b.mp4", 80).read_bytes()
    r = TestClient(app).post(f"/api/v1/clip/pair/analyse?size_a={len(a)}&interval_s=10",
                             content=a + b)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["samples"][0]["people"] == 5
    assert body["samples"][0]["sides"][0]["people"] == 3
    assert body["n_depends_on_sum"] == body["n_samples"]
    assert [b["source"] for b in body["boundaries"]] == ["none", "none"]
    assert len(made) == 2 and not any(p.exists() for p in made)


def test_the_route_refuses_a_body_with_no_second_video(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(clip_pair, "_pipeline", _pipeline)
    a = _video(tmp_path / "a.mp4", 40).read_bytes()
    r = TestClient(app).post(f"/api/v1/clip/pair/analyse?size_a={len(a)}", content=a)
    assert r.status_code == 422
    assert "second video" in r.json()["detail"]


def test_the_page_is_served_and_carries_the_site_navigation() -> None:
    client = TestClient(app)
    page = client.get("/clip/pair")
    assert page.status_code == 200 and "/api/v1/clip/pair/analyse" in page.text
    assert 'href="/testing"' in page.text
