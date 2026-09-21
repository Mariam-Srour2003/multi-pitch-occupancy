"""Explaining still images, one or many (WP6-T6).

The image reviewer shares its explanation with the clip walkthrough - `explain_frame` is
called by both - so the tests that matter here are the ones about what is *different*:

* stills have no neighbours, so nothing is smoothed, and the page has to say so rather than
  imply the clip's temporal correction is at work;
* a batch arrives as JSON with base64 rather than as raw bytes, which is a wider door than
  `/clip/analyse` has and needs its bounds asserted;
* a file that is not an image yields nothing rather than a guess, and the count that reaches
  the page has to reconcile with the number of files submitted - otherwise a skipped upload
  is invisible.

The shared half is asserted once, against the real backbone: if `explain_frame` ever stopped
agreeing with the deployed classifier, both pages would be wrong together and neither would
say so.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api.app import app
from pitch_occupancy.api.image_walkthrough import MAX_IMAGES
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.walkthrough import Shot, walk_images

E, P = Class3.EMPTY, Class3.ACTIVE_PLAY


def _image(path: Path, value: int = 120) -> Path:
    cv2.imwrite(str(path), np.full((64, 64, 3), value, dtype=np.uint8))
    return path


def _b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _real_frames(n: int = 2) -> list[str]:
    """Real labelled frames, or a skip. A synthetic grey square exercises the plumbing but
    says nothing about whether the explanation is meaningful."""
    found = sorted(glob.glob("data/processed/2_playing/*.jpg"))[:n]
    if len(found) < n:
        pytest.skip("labelled frames not present in this checkout")
    return found


class _StubClassifier:
    """Enough of `ProbeClassifier` for the unexplained path, which touches only these two."""

    backbone = "stub"
    n_train = 0

    class probe:  # noqa: N801 - mimics the attribute, not a class being used as one
        classes_ = [E.value, P.value]

    def __call__(self, image_bgr):
        return P, 0.9


# --- walk_images ----------------------------------------------------------------------


def test_shots_carry_their_position_and_name(tmp_path) -> None:
    paths = [_image(tmp_path / f"{i}.jpg") for i in range(3)]
    shots = list(walk_images(paths, _StubClassifier(), explain_n=0))
    assert [s.index for s in shots] == [0, 1, 2]
    assert [s.name for s in shots] == ["0.jpg", "1.jpg", "2.jpg"]


def test_shots_carry_the_size_the_image_actually_was(tmp_path) -> None:
    """The page prints it beside the prediction, and a 64x64 thumbnail and a 4K frame are
    not the same evidence even when they score the same."""
    cv2.imwrite(str(tmp_path / "wide.jpg"), np.zeros((90, 160, 3), dtype=np.uint8))
    shot = next(iter(walk_images([tmp_path / "wide.jpg"], _StubClassifier(), explain_n=0)))
    assert (shot.width, shot.height) == (160, 90)


def test_a_file_that_is_not_an_image_yields_nothing_rather_than_a_guess(tmp_path) -> None:
    """The same rule `walk_clip` applies to a frame that will not decode.

    Yielding a prediction for a file the decoder rejected would be inventing an observation,
    which is the one thing this system is built never to do.
    """
    (tmp_path / "notes.txt").write_text("this is not an image", encoding="utf-8")
    good = _image(tmp_path / "real.jpg")
    shots = list(walk_images([tmp_path / "notes.txt", good], _StubClassifier(), explain_n=0))
    assert len(shots) == 1
    assert shots[0].name == "real.jpg"


def test_explain_n_bounds_how_many_get_an_evidence_map(tmp_path) -> None:
    paths = [_image(tmp_path / f"{i}.jpg") for i in range(4)]
    shots = list(walk_images(paths, _StubClassifier(), explain_n=0))
    assert not any(s.explained for s in shots)


def test_max_images_is_enforced_in_the_generator_not_only_the_route(tmp_path) -> None:
    """The route rejects an oversized batch, but `walk_images` is importable on its own and
    a caller that bypassed the route should not be able to start an unbounded run."""
    paths = [_image(tmp_path / f"{i}.jpg") for i in range(6)]
    shots = list(walk_images(paths, _StubClassifier(), explain_n=0, max_images=4))
    assert len(shots) == 4


# --- the shared explanation -----------------------------------------------------------


def test_the_image_walkthrough_predicts_what_the_deployed_classifier_predicts() -> None:
    """`explain_frame` derives its prediction from pre-pooling features rather than calling
    the classifier again. That halves the cost and is worthless if the two disagree - and it
    now backs two pages, so a divergence would be wrong in both."""
    from pitch_occupancy.vision.classifier import load_classifier

    frame_path = _real_frames(1)[0]
    classifier = load_classifier()
    shot = next(iter(walk_images([frame_path], classifier)))

    direct_state, direct_conf = classifier(cv2.imread(frame_path))
    assert shot.predicted == str(direct_state)
    assert shot.confidence == pytest.approx(direct_conf, abs=1e-6)


def test_the_evidence_map_sums_to_the_score() -> None:
    """The decomposition claims to be exact rather than a saliency heuristic, so the page
    prints both scores and their difference. If that error is not tiny, the claim is false."""
    from pitch_occupancy.vision.classifier import load_classifier

    shot = next(iter(walk_images(_real_frames(1), load_classifier())))
    assert shot.explained
    assert shot.reconstruction_error == pytest.approx(0.0, abs=1e-6)


def test_focus_ratio_is_none_when_nobody_was_detected() -> None:
    """A ratio over zero area is not a small number - it is not a number, and rendering it
    as 0.00x would read as "the evidence missed the people" on a frame that has none."""
    assert Shot(index=0, name="x", predicted=P.value, confidence=1.0, elapsed_ms=0.0,
                people_area=0.0, evidence_on_people=0.0).focus_ratio is None


# --- the route ------------------------------------------------------------------------


def test_the_stream_is_newline_delimited_json_with_a_meta_and_a_done(tmp_path) -> None:
    client = TestClient(app)
    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    response = client.post("/api/v1/images/walkthrough?explain_n=0", json=body)
    assert response.status_code == 200

    records = [json.loads(line) for line in response.text.strip().splitlines()]
    assert records[0]["type"] == "meta"
    assert records[0]["n_submitted"] == 1
    assert records[-1]["type"] == "done"


def test_a_skipped_file_is_reconcilable_from_the_done_record(tmp_path) -> None:
    """`n` and `n_submitted` can differ, and a page showing only what it managed to read
    would drop the difference silently."""
    junk = base64.b64encode(b"not an image at all").decode()
    body = {"images": [{"name": "junk.txt", "data": junk}]}
    response = TestClient(app).post("/api/v1/images/walkthrough", json=body)

    records = [json.loads(line) for line in response.text.strip().splitlines()]
    assert records[0]["n_submitted"] == 1
    # Exact, not a subset: the done record is what a page reconciles its counts from, and a
    # field appearing in it unannounced is a field nobody accounted for. `n_gated` arrived on
    # 2026-09-19, when the walkthrough endpoints started applying the deployed gates - it is
    # 0 here because nothing was readable, so there was no verdict for a gate to overrule.
    assert records[-1] == {"type": "done", "n": 0, "n_unreadable": 1, "n_gated": 0}


def test_a_data_url_prefix_is_accepted(tmp_path) -> None:
    """`FileReader.readAsDataURL` produces the prefixed form, so the browser's natural
    output has to work without every caller stripping it first."""
    payload = "data:image/jpeg;base64," + _b64(_image(tmp_path / "a.jpg"))
    response = TestClient(app).post(
        "/api/v1/images/walkthrough?explain_n=0", json={"images": [{"name": "a", "data": payload}]}
    )
    records = [json.loads(line) for line in response.text.strip().splitlines()]
    assert records[-1]["n"] == 1


def test_the_batch_size_is_capped(tmp_path) -> None:
    data = _b64(_image(tmp_path / "a.jpg"))
    body = {"images": [{"name": f"{i}.jpg", "data": data} for i in range(MAX_IMAGES + 1)]}
    assert TestClient(app).post("/api/v1/images/walkthrough", json=body).status_code == 413


def test_an_empty_batch_is_refused() -> None:
    assert TestClient(app).post(
        "/api/v1/images/walkthrough", json={"images": []}
    ).status_code == 422


def test_invalid_base64_is_refused_before_anything_streams() -> None:
    assert TestClient(app).post(
        "/api/v1/images/walkthrough", json={"images": [{"name": "x", "data": "not!!b64"}]}
    ).status_code == 422


def test_the_uploads_are_deleted_after_streaming(tmp_path) -> None:
    """Same guarantee as the clip route, and it has the same trap: a StreamingResponse's
    body runs after the handler returns, so deleting in the handler would remove the files
    before a single one had been read."""
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "shots-*")))
    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    TestClient(app).post("/api/v1/images/walkthrough?explain_n=0", json=body)
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "shots-*")))
    assert after == before


def test_two_uploads_sharing_a_name_do_not_collide(tmp_path) -> None:
    """A browser can send two files called `frame.jpg` from different folders. Writing both
    to one path would analyse the second twice and report the first's name against it."""
    a = _b64(_image(tmp_path / "a.jpg", value=10))
    b = _b64(_image(tmp_path / "b.jpg", value=240))
    body = {"images": [{"name": "frame.jpg", "data": a}, {"name": "frame.jpg", "data": b}]}
    response = TestClient(app).post("/api/v1/images/walkthrough?explain_n=0", json=body)

    shots = [r for r in (json.loads(x) for x in response.text.strip().splitlines())
             if r["type"] == "shot"]
    assert len(shots) == 2
    assert [s["index"] for s in shots] == [0, 1]


# --- the page -------------------------------------------------------------------------


def test_the_page_is_served_and_reaches_the_batch_endpoint() -> None:
    html = TestClient(app).get("/images").text
    assert "/api/v1/images/walkthrough" in html
    assert 'id="file"' in html and "multiple" in html


def test_the_page_says_stills_are_not_smoothed() -> None:
    """The one claim this page must not let a reader assume. The clip reviewer corrects an
    isolated misread from its neighbours; a set of stills has none, so a page that looked
    like the clip page without saying this would imply a correction that never happened.
    """
    html = " ".join(TestClient(app).get("/images").text.split())
    assert "Each image stands on its own" in html
    assert "no neighbours" in html


def test_the_page_is_honest_that_slow_motion_is_not_the_model() -> None:
    """Inherited from the clip page for the same reason: a reader who thinks they sped the
    model up has learnt something false about the system.

    Asserted on whitespace-normalised text, because the claim is the sentence and not where
    the source happens to wrap it - the first version of this test failed on a line break in
    the middle of the very phrase it was protecting.
    """
    html = " ".join(TestClient(app).get("/images").text.split())
    assert "the model runs at the same speed either way" in html


def test_the_two_reviewers_link_to_each_other() -> None:
    client = TestClient(app)
    assert 'href="/images"' in client.get("/clip").text
    assert 'href="/clip"' in client.get("/images").text


def test_a_failure_mid_stream_becomes_a_record_rather_than_a_truncated_body(
    tmp_path, monkeypatch
) -> None:
    """The response is committed as 200 before the generator runs.

    So an exception the generator does not catch escapes as an unhandled ASGI error and the
    client gets a truncated NDJSON body with no `error` record in it - a page that spins
    forever while the server console holds the only account of what happened. This was
    observed for real: Windows Application Control blocked torch's `_C` DLL, the model
    failed to load with an `ImportError` on the very first record, and the browser showed
    nothing at all.

    `ImportError` specifically, because the model load is both the likeliest failure here
    and the one furthest from the `ValueError` the handler used to catch.
    """
    from pitch_occupancy.api import image_walkthrough as iw

    def _boom(*args, **kwargs):
        raise ImportError("DLL load failed while importing _C")

    monkeypatch.setattr(iw, "_classifier", _boom)

    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    response = TestClient(app).post("/api/v1/images/walkthrough", json=body)

    assert response.status_code == 200
    records = [json.loads(line) for line in response.text.strip().splitlines()]
    errors = [r for r in records if r["type"] == "error"]
    assert errors, f"the failure never reached the client: {records}"
    assert "ImportError" in errors[0]["detail"]
    # `fatal` tells the page not to suggest re-uploading: the model is missing, and no
    # different image would help.
    assert errors[0]["fatal"] is True


def test_the_uploads_are_still_deleted_when_the_stream_fails(tmp_path, monkeypatch) -> None:
    """The cleanup is in a `finally`, and a broadened `except` must not have moved it."""
    from pitch_occupancy.api import image_walkthrough as iw

    def _boom(*args, **kwargs):
        raise ImportError("no model")

    monkeypatch.setattr(iw, "_classifier", _boom)
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "shots-*")))

    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    TestClient(app).post("/api/v1/images/walkthrough", json=body)

    assert set(glob.glob(os.path.join(tempfile.gettempdir(), "shots-*"))) == before


# --- both models on the same frame (WP9-T4a) -------------------------------------------------


@pytest.mark.slow
def test_an_explained_image_carries_the_detectors_boxes_beside_the_probes_heatmap(
        tmp_path) -> None:
    """The complaint that started A35 was two pages giving opposite answers on one clip.

    The fix was to show the deployed path rather than the probe alone; this is the other
    half of it. A heatmap can only say *where* a score came from, so "the model is reading
    the floodlights" and "the model found six people and a ball" look alike on it. The
    detector's pane says which, in its own currency: a box per person, a ring round the
    ball, and the rule row that fired.
    """
    client = TestClient(app)
    body = {"images": [{"name": "a.jpg", "data": _b64(_image(tmp_path / "a.jpg"))}]}
    response = client.post("/api/v1/images/walkthrough?explain_n=1", json=body)
    assert response.status_code == 200

    shots = [json.loads(line) for line in response.text.strip().splitlines()
             if json.loads(line)["type"] == "shot"]
    explained = [s for s in shots if s["explained"]]
    assert explained, "nothing was explained, so there is nothing to check"
    for shot in explained:
        assert shot["boxes"].startswith("data:image/jpeg;base64,"), "a picture, not a promise"
        assert shot["boxes"] != shot["heat"], "two models, two different explanations"
        found = shot["detector"]
        assert found["model"] and found["state"]
        assert found["rule"] >= 1, "which row of the table fired, so the state is checkable"
        assert isinstance(found["people_inside"], int)
        assert found["trace"], "the steps that led to the state, in words"
    for shot in shots:
        if not shot["explained"]:
            assert "boxes" not in shot, (
                "a detector pass costs what a heatmap costs; both are bounded by explain_n")


def test_the_detector_pane_says_not_checked_rather_than_nobody(monkeypatch) -> None:
    """`None` from a detector means it did not run. It has never meant an empty pitch, and
    this pane is the newest place that distinction could be lost (`vision/explain.py`)."""
    import numpy as np

    from pitch_occupancy.vision import detector, overlay

    class Unavailable:
        def detect(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(detector.Detector, "load", staticmethod(lambda *a, **k: Unavailable()))
    frame = np.zeros((90, 160, 3), np.uint8)
    canvas, found = overlay.detector_pane(frame)
    assert found["checked"] is False
    assert found["state"] == "UNCERTAIN" and found["rule"] == 1
    assert canvas.shape == frame.shape
    assert np.array_equal(canvas, frame), "nothing found is not something to draw"
