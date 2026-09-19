"""Counting on every frame reports more and decides nothing (A38).

The review pages skip the detector on a verdict that is already EMPTY, because the gate only
weakens and there is nothing weaker - so the count column showed a dash on most rows, which
reads as "nobody was found" when it means "nobody looked". Reported as confusing, twice.

`PersonGate(always_count=True)` runs the detector anyway. **The thing that must hold is that
the verdict does not move**: the review page and the worker have to answer the same about the
same footage, and A35 and A37 were both that invariant breaking. So the first test here is the
one that matters, and the rest are about what is now visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.people import Counted, PersonGate

PLAY, EMPTY, C3 = Class3.ACTIVE_PLAY, Class3.EMPTY, Class3.MAINTENANCE_NON_SPORTING
FRAME = np.zeros((8, 8, 3), np.uint8)


def detector(monkeypatch, found: Counted | None):
    """Script what one detector pass finds, and count how often it is asked."""
    calls = {"n": 0}

    def detect_inside(image_bgr, polygon):
        calls["n"] += 1
        return found

    import pitch_occupancy.vision.people as people

    monkeypatch.setattr(people, "detect_inside", detect_inside)
    return calls


# --- the invariant ------------------------------------------------------------------------


@pytest.mark.parametrize("state", [PLAY, EMPTY, C3])
@pytest.mark.parametrize("found", [
    Counted(people=0, ball=False),
    Counted(people=2, ball=False),
    Counted(people=2, ball=True),
    Counted(people=9, ball=False),
    Counted(people=9, ball=True),
    None,
])
def test_always_count_never_changes_the_verdict(monkeypatch, state, found) -> None:
    """The whole point. Every state, every thing the detector could find: the two gates
    agree on the answer and differ only in whether a count comes back with it.

    If this ever fails, the review page and the worker have started disagreeing about the
    same footage again - which is A35 and A37, and is the reason this file exists."""
    detector(monkeypatch, found)
    quiet, _ = PersonGate().inspect(state, FRAME)
    loud, counted = PersonGate(always_count=True).inspect(state, FRAME)
    assert quiet is loud, f"{state.name} with {found} moved: {quiet.name} -> {loud.name}"
    if found is not None:
        assert counted == found


def test_an_empty_verdict_stays_empty_even_when_the_detector_finds_a_crowd(monkeypatch):
    """The sharp case. The motion gate has already said EMPTY; the detector then finds nine
    people. The verdict must not move - the gates only weaken, and a gate that strengthened
    one would stop being a gate - but the count is reported so the disagreement is visible."""
    detector(monkeypatch, Counted(people=9, ball=True))
    state, counted = PersonGate(always_count=True).inspect(EMPTY, FRAME)
    assert state is EMPTY
    assert counted is not None and counted.people == 9


# --- what it costs and what it reports ------------------------------------------------------


def test_the_default_still_skips_the_detector_on_an_empty_verdict(monkeypatch) -> None:
    """The worker pays a fifth of a second per camera per minute; it must not pay it for a
    count that cannot change anything."""
    calls = detector(monkeypatch, Counted(people=0, ball=False))
    state, counted = PersonGate().inspect(EMPTY, FRAME)
    assert (state, counted) == (EMPTY, None)
    assert calls["n"] == 0, "the detector must not run"


def test_always_count_runs_the_detector_exactly_once_per_frame(monkeypatch) -> None:
    calls = detector(monkeypatch, Counted(people=0, ball=False))
    PersonGate(always_count=True).inspect(EMPTY, FRAME)
    assert calls["n"] == 1
    PersonGate(always_count=True).inspect(PLAY, FRAME)
    assert calls["n"] == 2, "one pass per frame, whatever the verdict"


def test_a_detector_that_cannot_load_still_reports_not_checked(monkeypatch) -> None:
    """`None` is "not checked" and `0` is "checked, found nobody". Turning the count on must
    not collapse that distinction - a missing weights file would otherwise read as an empty
    pitch, which is the failure `vision/explain.py` is built around."""
    detector(monkeypatch, None)
    state, counted = PersonGate(always_count=True).inspect(EMPTY, FRAME)
    assert state is EMPTY and counted is None


# --- through the clip page ------------------------------------------------------------------


def test_the_review_pages_count_on_every_frame_and_the_worker_does_not() -> None:
    from pitch_occupancy.api.clip_review import _gates
    from pitch_occupancy.pipeline import default_gates

    _, page_gate = _gates()
    _, worker_gate = default_gates()
    assert page_gate.always_count is True
    assert worker_gate.always_count is False
    assert page_gate.small_group_max == worker_gate.small_group_max, (
        "they must differ in what they report and in nothing else")


def test_a_sample_flags_the_frames_where_the_two_gates_disagree() -> None:
    from pitch_occupancy.clip_analysis import ClipSample

    disagreeing = ClipSample(index=0, t_s=0.0, raw=EMPTY, confidence=0.9, smoothed=EMPTY,
                             people=3, motion=0.2)
    assert disagreeing.gates_disagree

    for agreeing in (
        ClipSample(index=0, t_s=0.0, raw=EMPTY, confidence=0.9, smoothed=EMPTY, people=0),
        ClipSample(index=0, t_s=0.0, raw=EMPTY, confidence=0.9, smoothed=EMPTY, people=None),
        ClipSample(index=0, t_s=0.0, raw=PLAY, confidence=0.9, smoothed=PLAY, people=8),
        ClipSample(index=0, t_s=0.0, raw=C3, confidence=0.9, smoothed=C3, people=2),
    ):
        assert not agreeing.gates_disagree


def test_the_analyse_route_carries_the_count_and_the_cue_for_every_sample(tmp_path) -> None:
    """End to end through the route the page calls, with the gates scripted so the test is
    about the plumbing rather than about a real detector."""
    import itertools

    import cv2
    from fastapi.testclient import TestClient

    import pitch_occupancy.api.clip_review as review
    from pitch_occupancy.api.app import app

    path = tmp_path / "c.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(40):
        writer.write(np.full((48, 64, 3), 90, np.uint8))
    writer.release()

    class Counting:
        """Finds two people on every frame, whatever the verdict - `always_count`'s shape."""

        always_count = True
        small_group_max = 4

        def inspect(self, state, image, polygon=None):
            return state, Counted(people=2, ball=False)

    class Moving:
        def apply(self, state, previous, frame, polygon=None):
            return (EMPTY if state is PLAY else state), 0.05

    cycle = itertools.repeat(PLAY)
    original_classifier, original_gates = review._classifier, review._gates
    review._classifier = lambda: (lambda img, **_kw: (next(cycle), 0.9))
    review._gates = lambda: (Moving(), Counting())
    try:
        body = TestClient(app).post(
            "/api/v1/clip/analyse?interval_s=1&window=1",
            content=path.read_bytes(),
            headers={"Content-Type": "application/octet-stream"},
        ).json()
    finally:
        review._classifier, review._gates = original_classifier, original_gates

    samples = body["samples"]
    assert len(samples) >= 3
    assert all(s["people"] == 2 for s in samples), "counted on every frame, not just some"
    # The first sample has no predecessor, so the motion gate stays silent and its verdict
    # stands - which is the same reason the Analyse table shows a count on row 0 and the
    # reason that row is not flagged below.
    assert samples[0]["motion"] is None and samples[0]["raw"] == PLAY.value
    assert not samples[0]["gates_disagree"]
    # From the second on, the motion gate turns PLAY into EMPTY while the detector finds two
    # people: the exact frame the old behaviour hid, now counted and flagged.
    assert all(s["motion"] == 0.05 for s in samples[1:])
    assert all(s["raw"] == EMPTY.value for s in samples[1:])
    assert all(s["gates_disagree"] for s in samples[1:])
