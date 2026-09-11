"""Stepping through a clip with its evidence maps (WP4-T5).

Two claims carry this feature and both are cheap to get wrong silently.

The first is that the walkthrough shows *the deployed model*. It derives the prediction from
the pre-pooling features it needs for the explanation rather than calling the classifier
again, which halves the cost and would be worthless if the two disagreed. That is asserted
against the real backbone, because a stub cannot be wrong in the way this could.

The second is that the upload is deleted. It is harder here than on `/clip/analyse`: a
StreamingResponse's body runs *after* the handler returns, so the obvious `finally` would
remove the file before a single frame had been read, and the failure would look like a
decode error rather than a deleted file.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api.app import app
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.walkthrough import EXPLAIN_ALL, Step, walk_clip

E, P = Class3.EMPTY, Class3.ACTIVE_PLAY


def _video(path: Path, seconds: int = 12, fps: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for i in range(seconds * fps):
        writer.write(np.full((64, 64, 3), i % 256, dtype=np.uint8))
    writer.release()
    return path


class _StubClassifier:
    """Enough of `ProbeClassifier` for the unexplained path, which touches only these two."""

    backbone = "stub"
    n_train = 0

    class probe:  # noqa: N801 - mimics the attribute, not a class being used as one
        classes_ = [E.value, P.value]

    def __call__(self, image_bgr):
        return P, 0.9


# --- the plain path -------------------------------------------------------------------------


def test_steps_carry_their_position_and_timestamp(tmp_path) -> None:
    steps = list(walk_clip(_video(tmp_path / "c.mp4"), _StubClassifier(),
                           interval_s=2.0, explain_n=0))
    assert [s.index for s in steps] == list(range(len(steps)))
    assert [s.t_s for s in steps] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert all(not s.explained for s in steps)


def test_explain_n_bounds_how_many_are_explained(tmp_path) -> None:
    """Explaining costs about twice a bare prediction, so this is the control that changes
    the work - unlike the page's slow-motion button, which only changes the display."""
    steps = list(walk_clip(_video(tmp_path / "c.mp4"), _StubClassifier(),
                           interval_s=2.0, explain_n=0))
    assert sum(s.explained for s in steps) == 0


def test_a_file_that_is_not_video_is_refused(tmp_path) -> None:
    bad = tmp_path / "no.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="could not open"):
        list(walk_clip(bad, _StubClassifier(), explain_n=0))


# --- what a Step reports ---------------------------------------------------------------------


def test_focus_ratio_is_none_when_nobody_was_detected() -> None:
    """A ratio over zero area is not a small number, it is not a number. Returning 0.0 would
    read as "the evidence avoided the people" when there were none to avoid."""
    step = Step(index=0, t_s=0.0, predicted=P.value, confidence=0.9, elapsed_ms=1.0,
                evidence=np.zeros((2, 2)), evidence_on_people=0.0, people_area=0.0)
    assert step.focus_ratio is None


def test_focus_ratio_divides_evidence_share_by_area_share() -> None:
    step = Step(index=0, t_s=0.0, predicted=P.value, confidence=0.9, elapsed_ms=1.0,
                evidence=np.zeros((2, 2)), evidence_on_people=0.6, people_area=0.2)
    assert step.focus_ratio == pytest.approx(3.0)


def test_reconstruction_error_is_none_without_an_explanation() -> None:
    step = Step(index=0, t_s=0.0, predicted=P.value, confidence=0.9, elapsed_ms=1.0)
    assert step.reconstruction_error is None and not step.explained


# --- against the real backbone ----------------------------------------------------------------


@pytest.mark.slow
def test_the_walkthrough_predicts_what_the_deployed_classifier_predicts(tmp_path) -> None:
    """The single-pass claim. `walk_clip` pools the features it already needed rather than
    running the backbone a second time; if that pooling did not match `embed_batch`, the page
    would be explaining a model nobody deploys - and it would look entirely plausible.
    """
    from pitch_occupancy.vision.classifier import load_classifier

    classifier = load_classifier()
    video = _video(tmp_path / "c.mp4", seconds=4)

    steps = list(walk_clip(video, classifier, interval_s=2.0, explain_n=EXPLAIN_ALL))
    assert steps

    capture = cv2.VideoCapture(str(video))
    for step in steps:
        capture.set(cv2.CAP_PROP_POS_MSEC, step.t_s * 1000.0)
        ok, frame = capture.read()
        assert ok
        state, confidence = classifier(frame)
        assert step.predicted == str(state)
        assert step.confidence == pytest.approx(confidence, abs=1e-6)
    capture.release()


@pytest.mark.slow
def test_the_evidence_map_sums_to_the_score(tmp_path) -> None:
    """The claim that makes this an explanation rather than a heatmap. The probe is linear
    over mean-pooled features, so the map is the summands of the score - not an attribution
    method with choices in it. If this drifts, the page's wording becomes false."""
    from pitch_occupancy.vision.classifier import load_classifier

    steps = list(walk_clip(_video(tmp_path / "c.mp4", seconds=4), load_classifier(),
                           interval_s=2.0, explain_n=EXPLAIN_ALL))
    assert steps
    for step in steps:
        assert step.explained
        assert step.reconstruction_error < 1e-9, (
            f"step {step.index} reconstructs to {step.score_from_map} against a direct "
            f"score of {step.score_direct}; the decomposition is no longer exact"
        )


# --- the streaming route -----------------------------------------------------------------------


def _stub_records(client, video: bytes, query: str = "?interval_s=2&explain_n=0"):
    with client.stream("POST", "/api/v1/clip/walkthrough" + query, content=video,
                       headers={"Content-Type": "application/octet-stream"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in response.iter_lines() if line.strip()]


def test_the_stream_is_newline_delimited_json_with_a_meta_and_a_done(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("pitch_occupancy.api.clip_review._CLASSIFIER", _StubClassifier())
    records = _stub_records(TestClient(app), _video(tmp_path / "c.mp4").read_bytes())
    assert records[0]["type"] == "meta"
    assert records[-1]["type"] == "done"
    steps = [r for r in records if r["type"] == "step"]
    assert steps and records[-1]["n"] == len(steps)
    assert all("clock" in s and "predicted" in s for s in steps)


def test_the_walkthrough_deletes_the_upload_after_streaming(tmp_path, monkeypatch) -> None:
    """Named in `ALLOWED_MUTATING`'s reason. The `finally` has to live in the generator: a
    StreamingResponse's body runs after the handler returns, so unlinking in the handler
    would delete the file before the first frame was read."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr("pitch_occupancy.api.clip_review._CLASSIFIER", _StubClassifier())
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
    monkeypatch.setattr("pitch_occupancy.api.clip_review._CLASSIFIER", _StubClassifier())
    records = _stub_records(TestClient(app), b"definitely not a video")
    assert records[-1]["type"] == "error"
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".upload"]


# --- the page ------------------------------------------------------------------------------------


def test_the_page_offers_the_walkthrough_and_is_honest_about_the_button() -> None:
    """A reader who thinks the skip button sped the model up has learnt something false about
    the system. The page has to say what it actually does."""
    text = TestClient(app).get("/clip").text
    assert "Watch it work" in text
    assert "Continue without slow motion" in text
    assert "not a saliency heuristic" in text
    assert "it does not make the model faster" in text or "does not make the model" in text
