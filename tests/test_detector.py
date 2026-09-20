"""The detector registry and the counting arithmetic (A36, WP9-T2/T3).

No real model runs here: the registry, the cache, the tile merge and the counts are all
testable with detections written by hand, which is why `vision/counting.py` takes detections
rather than frames.
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from pitch_occupancy.vision import counting, detector
from pitch_occupancy.vision.detector import (
    DEFAULT_DETECTOR,
    DETECTORS,
    PERSON,
    SPORTS_BALL,
    Detection,
    Detector,
    merge,
)

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
LEFT_HALF = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
SHAPE = (100, 200)  # height, width


def person(x1, y1, x2, y2, conf=0.9) -> Detection:
    return Detection(PERSON, (x1, y1, x2, y2), conf)


def ball(x1, y1, x2, y2, conf=0.3) -> Detection:
    return Detection(SPORTS_BALL, (x1, y1, x2, y2), conf)


# --- the registry -------------------------------------------------------------------------


def test_the_default_detector_is_the_one_every_published_count_rests_on() -> None:
    assert DEFAULT_DETECTOR == "yolov8n"
    assert DETECTORS[DEFAULT_DETECTOR].weights == "yolov8n.pt"
    assert DETECTORS[DEFAULT_DETECTOR].imgsz == 1280, "A16's setting"


def test_every_entry_names_itself_and_segmentation_is_marked() -> None:
    for key, spec in DETECTORS.items():
        assert spec.key == key
        assert spec.segment == key.endswith("-seg"), key
        assert spec.weights.endswith(".pt")


def test_an_unknown_key_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown detector"):
        Detector.load("not-a-detector")


def test_missing_weights_mean_not_checked_and_never_a_download(monkeypatch, tmp_path) -> None:
    """None, not [] - and no network. A worker that reaches for the network on its first
    frame at 3 am fails at 3 am."""
    monkeypatch.setattr(detector, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setattr(detector, "_MODELS", threading.local())

    class Boom:
        def __init__(self, name):
            raise AssertionError(f"tried to construct/download {name}")

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Boom))
    det = Detector.load("yolo11n")
    assert not det.available
    assert det.detect(np.zeros((8, 8, 3), np.uint8)) is None


def test_the_model_is_built_once_per_thread_and_reused(monkeypatch, tmp_path) -> None:
    """A21: constructing the model cost more than the inference."""
    monkeypatch.setattr(detector, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setattr(detector, "_MODELS", threading.local())
    (tmp_path / "yolov8n.pt").write_bytes(b"not really weights")
    built: list[str] = []

    class FakeYOLO:
        def __init__(self, name):
            built.append(name)

        def predict(self, _image, **_kwargs):
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    det = Detector.load("yolov8n")
    frame = np.zeros((16, 16, 3), np.uint8)
    for _ in range(4):
        assert det.detect(frame) == []
    assert len(built) == 1 and built[0].endswith("yolov8n.pt")


# --- detections ---------------------------------------------------------------------------


def test_a_detection_stands_on_its_feet() -> None:
    det = person(10, 20, 30, 80)
    assert det.foot == (20, 80)
    assert det.centre == (20, 50)
    assert det.height == 60 and det.area == 20 * 60


def test_merge_keeps_the_most_confident_of_overlapping_boxes_per_class() -> None:
    """Tiles overlap by design, so the same player appears in two crops; the merge must
    keep one of them and must not let a ball suppress a person."""
    a = person(10, 10, 50, 90, 0.8)
    b = person(12, 12, 52, 92, 0.6)  # the same person from the neighbouring tile
    c = person(120, 10, 160, 90, 0.7)  # a different person
    d = ball(12, 12, 52, 92, 0.9)  # a ball box lying over the person - different class
    kept = merge([b, a, c, d])
    assert a in kept and c in kept and d in kept
    assert b not in kept
    assert len(kept) == 3


def test_tiling_runs_the_whole_and_the_quarters_and_merges(monkeypatch, tmp_path) -> None:
    """Assert on the crops asked for and the merge, with `_predict` scripted: the whole frame
    plus four overlapping quarters, and a person seen in two of them counted once."""
    monkeypatch.setattr(detector, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setattr(detector, "_MODELS", threading.local())
    (tmp_path / "yolov8n.pt").write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "ultralytics",
                        types.SimpleNamespace(YOLO=lambda name: object()))
    det = Detector.load("yolov8n")
    calls: list[tuple[int, int]] = []

    def scripted(self, model, image, confidence, classes, imgsz, want_masks, *,
                 offset=(0, 0), frame_shape=None):
        calls.append(image.shape[:2])
        ox, oy = offset
        # one person standing at frame (100, 60)-(110, 90); report it wherever the crop
        # contains it, in frame coordinates
        x1, y1, x2, y2 = 100, 60, 110, 90
        h, w = image.shape[:2]
        if ox <= x1 and x2 <= ox + w and oy <= y1 and y2 <= oy + h:
            return [person(x1, y1, x2, y2, 0.5 + 0.01 * len(calls))]
        return []

    monkeypatch.setattr(Detector, "_predict", scripted)
    frame = np.zeros((100, 200, 3), np.uint8)
    found = det.detect(frame, tiles=2)
    assert calls[0] == (100, 200), "the whole frame first"
    assert len(calls) == 5, "then the four quarters"
    assert found is not None and len(found) == 1, found


# --- counting -----------------------------------------------------------------------------


def test_people_are_placed_by_their_feet_and_balls_by_their_centre() -> None:
    # Centre at x=100, on the left half's edge; feet at (105, 60), outside it. The person
    # is standing on the touchline and must not be counted as on the pitch.
    at_touchline = person(95, 10, 115, 60)
    on_pitch = person(20, 10, 40, 60)  # feet at (30, 60) inside
    high_ball = ball(40, 0, 60, 10)  # centre (50, 5) inside
    count = counting.count_inside([at_touchline, on_pitch, high_ball], LEFT_HALF, SHAPE)
    assert count.people_inside == 1 and count.people_total == 2
    assert count.ball_seen and count.ball_confidence == pytest.approx(0.3)
    assert count.bounded


def test_without_a_boundary_everything_counts_and_it_says_so() -> None:
    count = counting.count_inside([person(150, 10, 170, 60)], None, SHAPE)
    assert count.people_inside == 1 and not count.bounded


def test_confidence_floors_apply_per_class() -> None:
    faint_person = person(20, 10, 40, 60, conf=0.2)
    faint_ball = ball(40, 40, 50, 50, conf=0.05)
    count = counting.count_inside([faint_person, faint_ball], SQUARE, SHAPE)
    assert count.people_inside == 0 and count.people_total == 0 and not count.ball_seen
    count = counting.count_inside([faint_person, faint_ball], SQUARE, SHAPE,
                                  person_conf=0.1, ball_conf=0.01)
    assert count.people_inside == 1 and count.ball_seen


def test_the_minimum_height_filter_drops_a_shadow_but_remembers_it() -> None:
    """Row 3 of the table lowers the EMPTY confidence by what the filter dropped."""
    tall = person(20, 10, 40, 60)  # height 50
    short = person(60, 55, 70, 60)  # height 5
    count = counting.count_inside([tall, short], SQUARE, SHAPE, min_height_at=lambda y: 20.0)
    assert count.people_inside == 1 and count.raw_inside == 2
    assert count.people == (tall,)


def test_vehicles_inside_are_counted_by_their_feet() -> None:
    truck = Detection(7, (20, 10, 60, 60), 0.8)
    outside = Detection(2, (150, 10, 190, 60), 0.8)
    count = counting.count_inside([truck, outside], LEFT_HALF, SHAPE)
    assert count.vehicles_inside == 1


def test_a_person_whose_feet_the_frame_cuts_off_is_still_standing_on_the_pitch() -> None:
    """A detection touching the bottom edge used to vanish the moment a boundary existed.

    `Detection.foot` is the bottom-centre of the box, so a person close enough to the camera
    for the frame to cut their feet off gets ``foot_y == frame_height`` - one row past the
    last row there is. :func:`counting.inside` tested ``0 <= y < height``, so that point fell
    outside *every* polygon, including one covering the whole frame: counting with no boundary
    found them and counting with a boundary found none, which is the wrong way round.

    Reported from use on 2026-09-20. Two people standing in the foreground of a clip the
    operator had labelled `not playing` were read as **C1_EMPTY at zero people**. EMPTY is the
    one class this project promises not to miss; the failure was silent; and it only appears
    once a boundary exists, so every measurement taken without one had been blind to it.
    """
    shape = (496, 864)
    whole_frame = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cut_off = [person(532, 331, 630, 496, conf=0.94), person(782, 312, 863, 496, conf=0.87)]

    assert cut_off[0].foot == (581, 496), "one row past the last row of a 496-row frame"
    bounded = counting.count_inside(cut_off, whole_frame, shape)
    assert bounded.people_inside == 2, "a boundary must not delete the foreground"
    assert bounded.people_inside == counting.count_inside(cut_off, None, shape).people_inside


def test_the_edge_clamp_does_not_move_anybody_onto_the_pitch() -> None:
    """It pulls a point onto the last visible row. It does not pull anyone over a line."""
    top_half = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]]
    on_the_edge = person(20, 60, 40, 100)     # feet cut off, in the bottom half
    well_inside = person(20, 10, 40, 40)      # feet at y=40, in the top half
    count = counting.count_inside([on_the_edge, well_inside], top_half, SHAPE)
    assert count.people_inside == 1, "the edge person stands at the bottom, outside the polygon"
    assert count.people_total == 2
    # and the same clamp on the x axis, for a box the frame cuts off at the right
    right_edge = person(180, 10, 200, 40)
    assert counting.count_inside([right_edge], SQUARE, SHAPE).people_inside == 1


def test_spread_is_relative_to_the_boundary() -> None:
    a, b = person(0, 0, 10, 99), person(190, 0, 200, 99)  # feet at x=5 and x=195, y=99
    count = counting.count_inside([a, b], SQUARE, SHAPE)
    assert count.people_inside == 2
    assert count.spread == pytest.approx(190 / np.hypot(200, 100))
    assert counting.count_inside([a], SQUARE, SHAPE).spread is None


def test_hi_vis_reads_saturated_yellow_and_orange_pixels() -> None:
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[10:60, 20:40] = (0, 255, 255)  # BGR yellow: hue 30, saturated, bright
    frame[10:60, 60:80] = (200, 200, 200)  # grey: not hi-vis
    bright = person(20, 10, 40, 60)
    grey = person(60, 10, 80, 60)
    assert counting.hi_vis_fraction(frame, bright) > 0.9
    assert counting.hi_vis_fraction(frame, grey) == 0.0
    count = counting.count_inside([bright, grey], SQUARE, SHAPE, frame_bgr=frame,
                                  hi_vis_min_fraction=0.15)
    assert count.hi_vis_people == 1
    off = counting.count_inside([bright, grey], SQUARE, SHAPE)
    assert off.hi_vis_people == 0, "off unless asked, since it is unevaluated"


def test_persist_is_the_median_count_and_any_ball() -> None:
    """A detection in one frame of three does not survive; a person missed in one of three
    does; a ball seen once counts."""
    one_shadow = [counting.count_inside([person(20, 10, 40, 60)], SQUARE, SHAPE),
                  counting.count_inside([], SQUARE, SHAPE),
                  counting.count_inside([], SQUARE, SHAPE)]
    assert counting.persist(one_shadow).people_inside == 0
    assert counting.persist(one_shadow).raw_inside == 1, "what was dropped is remembered"

    missed_once = [counting.count_inside([person(20, 10, 40, 60), person(60, 10, 80, 60)],
                                         SQUARE, SHAPE),
                   counting.count_inside([person(20, 10, 40, 60)], SQUARE, SHAPE),
                   counting.count_inside([person(20, 10, 40, 60), person(60, 10, 80, 60),
                                          ball(90, 40, 100, 50)], SQUARE, SHAPE)]
    persisted = counting.persist(missed_once)
    assert persisted.people_inside == 2
    assert persisted.ball_seen and persisted.ball_confidence == pytest.approx(0.3)
    assert len(persisted.people) == 2, "the frame the verdict rests on"

    with pytest.raises(ValueError):
        counting.persist([])
    single = counting.count_inside([], SQUARE, SHAPE)
    assert counting.persist([single]) is single
