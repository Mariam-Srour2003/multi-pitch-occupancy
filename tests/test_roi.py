"""Pitch boundaries: saying what counts as this pitch (WP3-T1).

The feature exists for one failure: pitches are built in rows, so a camera watching one sees
its neighbours down the sides of the frame, and a match on the next pitch over puts real
players into frames where *this* pitch is empty. The model is right about what it sees and
wrong about what was asked.

That failure is invisible in the verdict, which is what makes these tests worth having. A
boundary that is subtly wrong does not raise, does not look wrong, and produces confident
answers about the wrong region - so the properties asserted here are the ones a person
cannot check by looking at the page:

* a polygon is stored and returned in **normalised** coordinates, so a camera swapped for a
  higher-resolution one keeps its boundary rather than silently masking the wrong region;
* a mis-click is refused at save time rather than persisted;
* a malformed store degrades to "no boundary" rather than raising on the live path;
* applying a boundary actually removes what is outside it - asserted on pixels, not on the
  absence of an exception.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api.app import app
from pitch_occupancy.vision import roi

#: The middle half of the frame, as a square. Coverage 0.25, which is the arithmetic the
#: shoelace formula has to reproduce.
MIDDLE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """Every test gets its own store. The real one is a committed config file, and a test
    that wrote to it would leave the repository dirty and the next run non-deterministic."""
    monkeypatch.setattr(roi, "STORE", tmp_path / "roi.json")


# --- geometry -------------------------------------------------------------------------


def test_coverage_is_the_area_the_boundary_keeps() -> None:
    assert roi.coverage(MIDDLE) == pytest.approx(0.25)


def test_coverage_does_not_depend_on_winding_direction() -> None:
    """A boundary clicked clockwise and one clicked anticlockwise are the same region, and
    the shoelace formula is signed - so an operator drawing the other way round must not get
    a negative coverage and a refusal."""
    assert roi.coverage(list(reversed(MIDDLE))) == pytest.approx(roi.coverage(MIDDLE))


def test_a_boundary_needs_at_least_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        roi.validate([[0.1, 0.1], [0.9, 0.9]])


def test_points_outside_the_frame_are_refused() -> None:
    """Coordinates are fractions, so a value above 1 is a pixel coordinate that slipped
    through - which would mask a region far outside the frame and silently keep nothing."""
    with pytest.raises(ValueError, match="0-1"):
        roi.validate([[0, 0], [1920, 0], [1920, 1080]])


def test_a_misclick_is_refused_rather_than_saved() -> None:
    """Three points clicked by accident produce a boundary that keeps almost nothing. Saved,
    it would leave the model looking at a sliver and answering confidently about it."""
    with pytest.raises(ValueError, match="mis-click"):
        roi.validate([[0.5, 0.5], [0.51, 0.5], [0.51, 0.51]])


# --- the store ------------------------------------------------------------------------


def test_a_boundary_survives_a_round_trip() -> None:
    roi.save("venue_01/camera_A", MIDDLE)
    assert roi.get("venue_01/camera_A") == MIDDLE


def test_saving_one_camera_leaves_the_others_alone() -> None:
    roi.save("camera_A", MIDDLE)
    roi.save("camera_B", [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    assert set(roi.load_all()) == {"camera_A", "camera_B"}
    assert roi.get("camera_A") == MIDDLE


def test_a_camera_with_no_boundary_reads_as_none_not_as_an_empty_one() -> None:
    """None means *no boundary*; an empty list would be a boundary that keeps nothing, and
    `apply` treats those completely differently."""
    assert roi.get("never-drawn") is None


def test_removing_a_boundary_reports_whether_there_was_one() -> None:
    roi.save("camera_A", MIDDLE)
    assert roi.remove("camera_A") is True
    assert roi.remove("camera_A") is False


def test_a_malformed_store_degrades_to_no_boundary_rather_than_raising() -> None:
    """This file is read on the live path. Someone breaking it while hand-editing should cost
    the boundaries, which is the behaviour the system had before boundaries existed - not a
    night's capture."""
    roi.STORE.write_text("{ not json at all", encoding="utf-8")
    assert roi.load_all() == {}


def test_an_invalid_entry_is_dropped_without_taking_the_valid_ones_with_it() -> None:
    roi.STORE.write_text(
        json.dumps({"good": MIDDLE, "bad": [[0, 0], [2, 2], [3, 3]]}), encoding="utf-8"
    )
    loaded = roi.load_all()
    assert set(loaded) == {"good"}


def test_the_comment_key_is_not_read_back_as_a_camera() -> None:
    """The store writes an explanatory `_comment` block, as `configs/cameras.example.json`
    does. Reading it back as a camera would put a boundary named `_comment` in the editor."""
    roi.save("camera_A", MIDDLE)
    assert "_comment" in json.loads(roi.STORE.read_text(encoding="utf-8"))
    assert set(roi.load_all()) == {"camera_A"}


# --- masking --------------------------------------------------------------------------


def test_no_polygon_is_a_no_op() -> None:
    """The reason `roi` stayed out of the preprocessing search: a no-op that looks like a
    switch gets measured, and the finding is recorded about ROI masking rather than about
    the absence of a polygon."""
    frame = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    assert np.array_equal(roi.apply(frame, None), frame)


@pytest.mark.parametrize("fill", roi.FILLS)
def test_the_inside_of_the_boundary_is_untouched(fill) -> None:
    """Whatever happens outside, the pitch itself must arrive at the model unaltered."""
    frame = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = roi.apply(frame, MIDDLE, fill=fill)
    assert np.array_equal(out[40:60, 40:60], frame[40:60, 40:60])


def test_black_fill_zeroes_the_outside() -> None:
    frame = np.full((100, 100, 3), 200, np.uint8)
    out = roi.apply(frame, MIDDLE, fill="black")
    assert out[0, 0].tolist() == [0, 0, 0]


def test_blur_fill_destroys_detail_outside_without_zeroing_it() -> None:
    """The point of `blur`: a neighbouring pitch has to stop being recognisable as people,
    but a large black region is a distribution shift of its own."""
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    out = roi.apply(frame, MIDDLE, fill="blur")
    outside = out[0:40, 0:40]
    assert outside.std() < frame[0:40, 0:40].std()
    assert outside.mean() > 0


def test_an_unknown_fill_is_refused() -> None:
    frame = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError, match="unknown fill"):
        roi.apply(frame, MIDDLE, fill="feather")


def test_preprocess_roi_mask_delegates_rather_than_reimplementing() -> None:
    """One masking implementation. Two would eventually disagree about what "inside the
    pitch" means, and the disagreement would be between the editor's preview and the live
    worker - the two places it would be hardest to notice."""
    from pitch_occupancy.vision.preprocess import roi_mask

    frame = np.random.default_rng(3).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    assert np.array_equal(roi_mask(frame, MIDDLE), roi.apply(frame, MIDDLE, fill="black"))


# --- the API --------------------------------------------------------------------------


def test_a_camera_key_containing_a_slash_round_trips() -> None:
    """`venue_01/camera_A` is the shape `configs/cameras.json` already uses, and a slash
    cannot survive a URL path segment - `PUT /api/v1/roi/venue_01%2Fcamera_A` is a 404.
    So the key travels in the body, and this is the test that says why."""
    client = TestClient(app)
    saved = client.put(
        "/api/v1/roi", json={"camera": "venue_01/camera_A", "polygon": MIDDLE}
    )
    assert saved.status_code == 200
    assert "venue_01/camera_A" in client.get("/api/v1/roi").json()["cameras"]
    assert client.delete(
        "/api/v1/roi", params={"camera": "venue_01/camera_A"}
    ).json()["removed"] is True


def test_the_api_refuses_a_misclick_with_a_reason() -> None:
    response = TestClient(app).put(
        "/api/v1/roi",
        json={"camera": "x", "polygon": [[0.5, 0.5], [0.51, 0.5], [0.51, 0.51]]},
    )
    assert response.status_code == 422
    assert "mis-click" in response.json()["detail"]


def test_the_editor_page_is_served() -> None:
    html = TestClient(app).get("/roi").text
    assert "/api/v1/roi" in html
    assert "canvas" in html


def test_the_page_explains_the_neighbouring_pitch() -> None:
    """The page has to say what it is for. An editor offering a polygon tool with no reason
    attached gets used to crop out whatever looks untidy, which is not the same job."""
    html = " ".join(TestClient(app).get("/roi").text.split())
    assert "neighbours" in html or "neighbouring" in html


def test_the_image_reviewer_reports_a_boundary_that_was_asked_for_and_not_found() -> None:
    """Silently analysing the whole frame would produce precisely the answers the operator
    applied a boundary to avoid, and nothing would say so."""
    import base64
    import io

    import cv2

    frame = np.full((64, 64, 3), 120, np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    payload = base64.b64encode(io.BytesIO(buf).getvalue()).decode()

    response = TestClient(app).post(
        "/api/v1/images/walkthrough",
        params={"camera": "never-drawn", "explain_n": 0},
        json={"images": [{"name": "a.jpg", "data": payload}]},
    )
    meta = json.loads(response.text.strip().splitlines()[0])
    assert meta["camera"] == "never-drawn"
    assert meta["boundary"] is False


# --- boundaries on video --------------------------------------------------------------


def _video(path, seconds: int = 3, fps: int = 10, size=(64, 64)):
    """A clip whose frames differ, so "the first frame" is a checkable claim."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        writer.write(np.full((size[1], size[0], 3), (i * 7) % 256, dtype=np.uint8))
    writer.release()
    return path


def test_the_first_frame_of_a_video_comes_back_as_an_image(tmp_path) -> None:
    """The drawing surface for a clip's boundary.

    Server-side rather than a `<video>` element painting frame 0 onto a canvas: the browser
    and OpenCV need not agree on which frame a seek to 0 lands on, on rotation metadata, or
    on colour conversion, and a boundary drawn against a frame the analysis never sees is
    wrong by however much they differ - with nothing anywhere reporting it.
    """
    response = TestClient(app).post(
        "/api/v1/roi/first-frame", content=_video(tmp_path / "clip.mp4").read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["frame"].startswith("data:image/jpeg;base64,")
    assert (body["width"], body["height"]) == (64, 64)


def test_the_first_frame_is_returned_at_full_width(tmp_path) -> None:
    """`_jpeg` downscales to 640 by default, which is right for a sixty-step walkthrough and
    wrong here: the operator is about to place points on this by eye, and a downscaled
    surface costs precision they cannot get back."""
    response = TestClient(app).post(
        "/api/v1/roi/first-frame",
        content=_video(tmp_path / "wide.mp4", size=(960, 540)).read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.json()["width"] == 960


def test_a_file_that_is_not_a_video_is_refused(tmp_path) -> None:
    (tmp_path / "notes.txt").write_bytes(b"not a video")
    response = TestClient(app).post(
        "/api/v1/roi/first-frame", content=(tmp_path / "notes.txt").read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422


def test_an_empty_body_is_refused_before_anything_is_decoded() -> None:
    response = TestClient(app).post(
        "/api/v1/roi/first-frame", content=b"",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422


def test_the_first_frame_upload_is_deleted(tmp_path, monkeypatch) -> None:
    """Footage of people, so the same `finally` guarantee every other clip route carries."""
    import tempfile as tf

    workspace = tmp_path / "uploads"
    workspace.mkdir()
    monkeypatch.setattr(tf, "tempdir", str(workspace))

    TestClient(app).post(
        "/api/v1/roi/first-frame", content=_video(tmp_path / "clip.mp4").read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert list(workspace.iterdir()) == []


def test_a_clip_walkthrough_reports_the_boundary_it_applied(tmp_path, monkeypatch) -> None:
    """One outline drawn on frame 0 governs the whole recording, so the clip route has to
    say which boundary it used - and say so when the camera it was given has none."""
    from pitch_occupancy.api import clip_walkthrough as cw

    monkeypatch.setattr(cw, "_classifier", lambda: _StubForClip())
    roi.save("camera_A", MIDDLE)

    client = TestClient(app)
    video = _video(tmp_path / "clip.mp4").read_bytes()
    for camera, expected in (("camera_A", True), ("never-drawn", False)):
        response = client.post(
            f"/api/v1/clip/walkthrough?explain_n=0&camera={camera}",
            content=video, headers={"Content-Type": "application/octet-stream"},
        )
        meta = json.loads(response.text.strip().splitlines()[0])
        assert meta["camera"] == camera
        assert meta["boundary"] is expected


class _StubForClip:
    """Enough of `ProbeClassifier` for the unexplained path."""

    backbone = "stub"
    n_train = 0

    class probe:  # noqa: N801 - mimics the attribute
        classes_ = ["C1_EMPTY", "C2_ACTIVE_PLAY"]

    def __call__(self, image_bgr):
        return "C2_ACTIVE_PLAY", 0.9
