"""People on the pitch make a still's EMPTY verdict C3 (image reviewer only).

The image reviewer called a pitch EMPTY with people standing on it, because the person gate
only weakens and EMPTY is the weakest verdict. `PersonGate(people_overrule_empty=True)` turns
that into C3 - never into ACTIVE_PLAY - and the default gate, which the worker and the clip
pages use, is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.people import Counted, PersonGate

PLAY, EMPTY, C3 = Class3.ACTIVE_PLAY, Class3.EMPTY, Class3.MAINTENANCE_NON_SPORTING
FRAME = np.zeros((8, 8, 3), np.uint8)
STILLS = PersonGate(always_count=True, people_overrule_empty=True)


def detector(monkeypatch, found: Counted | None) -> None:
    import pitch_occupancy.vision.people as people

    monkeypatch.setattr(people, "detect_inside", lambda image_bgr, polygon: found)


@pytest.mark.parametrize("people", [1, 4, 5, 12])
@pytest.mark.parametrize("ball", [False, True])
def test_empty_with_people_inside_becomes_c3_never_play(monkeypatch, people, ball) -> None:
    detector(monkeypatch, Counted(people=people, ball=ball))
    state, counted = STILLS.inspect(EMPTY, FRAME)
    assert state is C3
    assert counted.people == people


def test_empty_with_nobody_inside_stays_empty(monkeypatch) -> None:
    detector(monkeypatch, Counted(people=0, ball=True))
    assert STILLS.inspect(EMPTY, FRAME)[0] is EMPTY


def test_a_detector_that_did_not_run_leaves_empty_alone(monkeypatch) -> None:
    """Not checked is not "found somebody" either."""
    detector(monkeypatch, None)
    assert STILLS.inspect(EMPTY, FRAME) == (EMPTY, None)


@pytest.mark.parametrize("state", [PLAY, C3])
@pytest.mark.parametrize("found", [Counted(people=0, ball=False), Counted(people=3, ball=False),
                                   Counted(people=9, ball=True), None])
def test_other_verdicts_are_decided_exactly_as_before(monkeypatch, state, found) -> None:
    detector(monkeypatch, found)
    assert (STILLS.inspect(state, FRAME)[0]
            is PersonGate(always_count=True).inspect(state, FRAME)[0])


def test_the_default_gate_still_never_strengthens_empty(monkeypatch) -> None:
    """The worker and the clip pages: unchanged."""
    detector(monkeypatch, Counted(people=6, ball=True))
    assert PersonGate().inspect(EMPTY, FRAME)[0] is EMPTY
    assert PersonGate(always_count=True).inspect(EMPTY, FRAME)[0] is EMPTY


def test_the_image_route_turns_it_on(monkeypatch, tmp_path) -> None:
    """End to end through `walk_records`: a still the probe calls EMPTY, with people found
    inside, comes out C3 and is reported as gated."""
    import json

    import cv2

    import pitch_occupancy.api.image_walkthrough as iw

    detector(monkeypatch, Counted(people=2, ball=False))

    def classify(frame):
        return EMPTY, 0.99

    monkeypatch.setattr(iw, "_classifier", lambda: classify)
    path = tmp_path / "a.jpg"
    cv2.imwrite(str(path), np.full((48, 64, 3), 90, np.uint8))
    lines = [json.loads(x) for x in iw.walk_records([path], names=["a.jpg"], explain_n=0,
                                                     redact=False)]
    shot = next(x for x in lines if x["type"] == "shot")
    assert shot["predicted"] == C3.value
    assert shot["n_inside"] == 2
    assert next(x for x in lines if x["type"] == "done")["n_gated"] == 1


# --- the gate's own inside test and weights (audit VI-2, VI-3) ------------------------------


def test_a_person_cut_off_by_the_bottom_edge_is_inside_a_whole_frame_boundary(monkeypatch):
    """`detector` clips y2 to the frame height, and the gate's private copy of the inside test
    read ``y < height`` - so a foreground player was outside every polygon, the whole frame
    included, and real play was overruled to EMPTY. It now uses `counting.count_inside`."""
    from pitch_occupancy.vision import counting, explain, people

    whole = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    frame = np.zeros((100, 200, 3), np.uint8)
    boxes = [(0, (20, 40, 50, 100), 0.9), (0, (120, 30, 150, 100), 0.8),
             (0, (60, 20, 80, 70), 0.9), (0, (90, 20, 110, 70), 0.9),
             (0, (160, 20, 180, 70), 0.9)]
    monkeypatch.setattr(explain, "detect_objects", lambda *_a, **_k: boxes)
    found = people.detect_inside(frame, whole)
    assert found.people == 5
    assert found.people == people.detect_inside(frame, None).people
    dets = [counting.Detection(c, b, p) for c, b, p in boxes]
    assert found.people == counting.count_inside(dets, whole, (100, 200)).people_inside
    state, _ = PersonGate().inspect(PLAY, frame, whole)
    assert state is PLAY


def test_the_gate_loads_its_weights_by_absolute_path(monkeypatch, tmp_path):
    """A bare ``yolov8n.pt`` was resolved against the working directory and, missing there,
    downloaded inside a prediction."""
    from pathlib import Path

    from pitch_occupancy.vision import detector, explain, people

    seen = {}
    monkeypatch.setattr(explain, "detect_objects",
                        lambda *_a, **k: seen.update(k) or [])
    monkeypatch.chdir(tmp_path)
    people.detect_inside(FRAME, None)
    assert Path(seen["model_name"]).is_absolute()
    assert Path(seen["model_name"]).parent == detector.WEIGHTS_DIR
