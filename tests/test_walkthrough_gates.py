"""The walkthrough pages answer with the system, not the probe (A35's sequel, 2026-09-19).

Reported from use: the Analyse tab called a clip of an empty floodlit pitch EMPTY and noted
*58 verdicts were weakened by the gates*, while **Watch it work**, on the same clip and the
same boundary, stepped through it as `C2_ACTIVE_PLAY` at confidence 0.999 on almost every
frame. Two tabs, one clip, opposite answers - the defect A35 closed on the analyse path, on
the endpoint A35's own closing paragraph named as where it would surface next.

The cause was that `walk_clip` took no gates and therefore streamed the probe alone. What is
pinned here is the property, not the plumbing: **the same frames through both paths produce
the same verdicts**, and the pre-gate verdict is still reported so the page can show the
correction rather than quietly serving a different answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.people import Counted
from pitch_occupancy.vision.walkthrough import walk_clip, walk_images

PLAY, EMPTY, C3 = Class3.ACTIVE_PLAY, Class3.EMPTY, Class3.MAINTENANCE_NON_SPORTING


def _video(path, *, seconds: int = 6, fps: int = 10):
    """A clip of an unchanging scene - nothing moves, as on an empty pitch."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(seconds * fps):
        writer.write(np.full((48, 64, 3), 90, np.uint8))
    writer.release()
    return path


class AlwaysPlay:
    """The probe on an empty floodlit pitch: confident, and wrong (A20, A26)."""

    backbone = "stub"
    n_train = 0

    def __call__(self, frame, *, polygon=None):
        return PLAY, 0.999


class NobodyThere:
    """A person gate whose detector finds nobody inside the outline."""

    def inspect(self, state, frame, polygon=None):
        if state is EMPTY:
            return state, None
        return EMPTY, Counted(people=0, ball=False)


class SmallGroup:
    def inspect(self, state, frame, polygon=None):
        if state is EMPTY:
            return state, None
        return C3, Counted(people=2, ball=False)


class NothingMoved:
    def apply(self, state, previous, frame, polygon=None):
        return (EMPTY if state is PLAY else state), 0.01


# --- the clip page --------------------------------------------------------------------


def test_without_gates_it_streams_the_probe_alone_which_is_the_bug(tmp_path) -> None:
    """The behaviour this file exists to prevent, pinned so the fix is legible: passing no
    gates reproduces what the page did until 2026-09-19."""
    steps = list(walk_clip(_video(tmp_path / "c.mp4"), AlwaysPlay(), interval_s=1.0,
                           explain_n=0))
    assert steps and all(s.predicted == str(PLAY) for s in steps)
    assert all(s.probed is None for s in steps), "nothing was overruled, so nothing to report"


def test_with_the_gates_it_answers_what_the_system_answers(tmp_path) -> None:
    steps = list(walk_clip(_video(tmp_path / "c.mp4"), AlwaysPlay(), interval_s=1.0,
                           explain_n=0, motion_gate=NothingMoved(),
                           person_gate=NobodyThere()))
    assert len(steps) >= 5
    assert all(s.predicted == str(EMPTY) for s in steps), [s.predicted for s in steps]
    assert all(s.probed == str(PLAY) for s in steps), "the probe's verdict is still reported"


def test_the_two_tabs_agree_about_one_clip(tmp_path) -> None:
    """The property the report was about. `analyse_clip` and `walk_clip` are different
    functions over the same footage and they must not disagree."""
    from pitch_occupancy.clip_analysis import analyse_clip

    video = _video(tmp_path / "c.mp4")
    kwargs = {"motion_gate": NothingMoved(), "person_gate": NobodyThere()}
    analysed = analyse_clip(video, lambda f: AlwaysPlay()(f), interval_s=1.0, window=1,
                            **kwargs)
    walked = list(walk_clip(video, AlwaysPlay(), interval_s=1.0, explain_n=0, **kwargs))
    assert [str(s.raw) for s in analysed.samples] == [s.predicted for s in walked]


def test_the_first_frame_has_no_motion_cue_and_the_person_gate_still_speaks(tmp_path):
    """A camera's first sample has no predecessor, so the motion gate is silent - and the
    person gate is what catches it. The count is reported for exactly those frames, which is
    why the Analyse table shows a number on row 0 and a dash on most of the rest."""
    steps = list(walk_clip(_video(tmp_path / "c.mp4"), AlwaysPlay(), interval_s=1.0,
                           explain_n=0, motion_gate=NothingMoved(),
                           person_gate=NobodyThere()))
    assert steps[0].motion is None, "no previous frame, so no cue"
    assert steps[0].n_inside == 0 and steps[0].predicted == str(EMPTY)
    assert steps[1].motion is not None


def test_a_gate_that_did_not_run_leaves_its_count_none_not_zero(tmp_path) -> None:
    """`None` is "not checked" and `0` is "checked, found nobody" - the distinction the
    whole detector contract rests on (`vision/explain.py`)."""
    steps = list(walk_clip(_video(tmp_path / "c.mp4"), AlwaysPlay(), interval_s=1.0,
                           explain_n=0, motion_gate=NothingMoved()))
    assert all(s.n_inside is None for s in steps), "no person gate ran"
    assert steps[1].predicted == str(EMPTY), "the motion gate still overruled it"


def test_a_small_group_with_no_ball_becomes_maintenance(tmp_path) -> None:
    steps = list(walk_clip(_video(tmp_path / "c.mp4"), AlwaysPlay(), interval_s=1.0,
                           explain_n=0, person_gate=SmallGroup()))
    assert all(s.predicted == str(C3) for s in steps)
    assert all(s.n_inside == 2 and s.ball is False for s in steps)


# --- the image page -------------------------------------------------------------------


def _image(path, value: int = 90):
    import cv2

    cv2.imwrite(str(path), np.full((48, 64, 3), value, np.uint8))
    return path


def test_stills_get_the_person_gate_and_cannot_get_the_motion_gate(tmp_path) -> None:
    """A folder of stills has no neighbours, so there is no previous sample to compare
    against - the same absence that rules out smoothing. The page says so; this pins that
    the half which *can* run does."""
    paths = [_image(tmp_path / f"{i}.jpg") for i in range(3)]
    shots = list(walk_images(paths, AlwaysPlay(), explain_n=0, person_gate=NobodyThere()))
    assert len(shots) == 3
    assert all(s.predicted == str(EMPTY) and s.probed == str(PLAY) for s in shots)
    assert all(s.n_inside == 0 for s in shots)
    assert not any(hasattr(s, "motion") for s in shots), "a still has no motion cue at all"


def test_stills_without_a_gate_still_stream_the_probe(tmp_path) -> None:
    paths = [_image(tmp_path / "a.jpg")]
    (shot,) = walk_images(paths, AlwaysPlay(), explain_n=0)
    assert shot.predicted == str(PLAY) and shot.probed is None


# --- the endpoints --------------------------------------------------------------------


@pytest.mark.parametrize("module,factory", [
    ("pitch_occupancy.api.clip_walkthrough", "_gates"),
    ("pitch_occupancy.api.image_walkthrough", "_gates"),
])
def test_both_streaming_endpoints_take_their_gates_from_the_one_factory(module, factory):
    """`clip_review._gates` is where the deployed gates are constructed. Both walkthrough
    endpoints import it rather than building their own, so a change to what the system
    deploys reaches every page at once - which is the whole point of the seam."""
    import importlib

    assert getattr(importlib.import_module(module), factory) is not None
    from pitch_occupancy.api.clip_review import _gates as source

    assert getattr(importlib.import_module(module), factory) is source
