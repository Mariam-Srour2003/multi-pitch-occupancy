"""The detector's overlay (A36, WP9-T4).

What is pinned here is what a reader is entitled to see: the people in one colour and the
ball in another, a detection found outside the boundary drawn differently from one counted
inside it, the boundary itself, the state and the count, and - for anything written to disk -
the two layers of redaction underneath, with the drawing still sharp on top.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.vision import overlay
from pitch_occupancy.vision.counting import count_inside
from pitch_occupancy.vision.detector import PERSON, SPORTS_BALL, Detection
from pitch_occupancy.vision.rules import MinuteState, RuleConfig, decide

LEFT_HALF = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
CFG = RuleConfig()


def frame(value: int = 60) -> np.ndarray:
    """A textured frame, so a blur has something to destroy."""
    rng = np.random.default_rng(0)
    return (rng.integers(0, 255, (200, 400, 3))).astype(np.uint8) if value < 0 else np.full(
        (200, 400, 3), value, np.uint8)


def noisy() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (200, 400, 3), dtype=np.uint8)


def person(x1, y1, x2, y2, conf=0.9, mask=None) -> Detection:
    return Detection(PERSON, (x1, y1, x2, y2), conf, mask)


def ball(x, y) -> Detection:
    return Detection(SPORTS_BALL, (x - 4, y - 4, x + 4, y + 4), 0.4)


def verdict_for(detections, polygon=LEFT_HALF, *, motion=None):
    count = count_inside(detections, polygon, (200, 400), person_conf=CFG.person_conf,
                         ball_conf=CFG.ball_conf)
    return decide(count, motion=motion, cfg=CFG)


def test_the_overlay_keeps_the_frames_shape_and_leaves_the_original_alone() -> None:
    original = frame()
    before = original.copy()
    drawn = overlay.draw_verdict(original, verdict_for([person(20, 100, 40, 180)]))
    assert drawn.shape == original.shape and drawn.dtype == original.dtype
    assert np.array_equal(original, before), "drawing must not touch the caller's frame"
    assert not np.array_equal(drawn, before), "something was drawn"


def test_people_and_the_ball_are_drawn_in_different_colours() -> None:
    drawn = overlay.draw_verdict(frame(), verdict_for([person(20, 100, 40, 180), ball(60, 120)]))
    colours = {tuple(int(c) for c in px) for px in drawn.reshape(-1, 3)}
    assert overlay.COLOURS["person"] in colours
    assert overlay.COLOURS["ball"] in colours
    assert overlay.COLOURS["person"] != overlay.COLOURS["ball"]


def test_a_person_outside_the_boundary_is_drawn_dimmed_rather_than_dropped() -> None:
    """"The detector found six and counted three" is the thing a reader most often needs
    explained, and an overlay that draws only the counted three cannot explain it."""
    inside, outside = person(20, 100, 40, 180), person(300, 100, 320, 180)
    verdict = verdict_for([inside, outside])
    assert verdict.people == 1 and verdict.count.people_total == 2
    drawn = overlay.draw_verdict(frame(), verdict, instances=[inside, outside])
    colours = {tuple(int(c) for c in px) for px in drawn.reshape(-1, 3)}
    assert overlay.COLOURS["person"] in colours
    assert overlay.COLOURS["person_outside"] in colours


def test_the_boundary_is_outlined() -> None:
    plain = overlay.draw_verdict(frame(), verdict_for([], polygon=None), polygon=None)
    outlined = overlay.draw_verdict(frame(), verdict_for([]), polygon=LEFT_HALF)
    yellow = overlay.COLOURS["boundary"]
    assert any(tuple(int(c) for c in px) == yellow for px in outlined.reshape(-1, 3))
    assert not any(tuple(int(c) for c in px) == yellow for px in plain.reshape(-1, 3))


def test_a_mask_is_filled_and_a_box_is_only_outlined() -> None:
    """A filled rectangle over a person claims a precision a box does not have."""
    mask = np.zeros((200, 400), bool)
    mask[100:180, 20:40] = True
    with_mask = overlay.draw_verdict(frame(), verdict_for([person(20, 100, 40, 180, mask=mask)]))
    without = overlay.draw_verdict(frame(), verdict_for([person(20, 100, 40, 180)]))
    tinted = int((with_mask[105:175, 25:35] != 60).any(axis=2).sum())
    plain = int((without[105:175, 25:35] != 60).any(axis=2).sum())
    assert tinted > plain, "the mask's interior is tinted; the box's is not"


def test_the_badge_names_the_state_and_the_count_and_the_state_sets_its_colour() -> None:
    playing = verdict_for([person(10 + 20 * i, 100, 25 + 20 * i, 180) for i in range(6)])
    assert playing.state is MinuteState.PLAYING
    drawn = overlay.draw_verdict(frame(), playing)
    corner = {tuple(int(c) for c in px) for px in drawn[0:3, 0:3].reshape(-1, 3)}
    assert overlay.STATE_COLOURS[MinuteState.PLAYING] in corner

    empty = overlay.draw_verdict(frame(), verdict_for([]))
    corner = {tuple(int(c) for c in px) for px in empty[0:3, 0:3].reshape(-1, 3)}
    assert overlay.STATE_COLOURS[MinuteState.EMPTY] in corner
    assert overlay.STATE_COLOURS[MinuteState.PLAYING] != overlay.STATE_COLOURS[MinuteState.EMPTY]


def test_an_uncertain_verdict_still_draws_and_is_its_own_colour() -> None:
    declined = decide(None, motion=None, cfg=CFG)
    drawn = overlay.draw_verdict(frame(), declined)
    corner = {tuple(int(c) for c in px) for px in drawn[0:3, 0:3].reshape(-1, 3)}
    assert overlay.STATE_COLOURS[MinuteState.UNCERTAIN] in corner


# --- redaction ----------------------------------------------------------------------------


def test_publishable_destroys_the_people_and_keeps_the_drawing_sharp() -> None:
    original = noisy()
    box = person(20, 100, 60, 180)
    verdict = verdict_for([box])
    picture = overlay.publishable(original, verdict, polygon=LEFT_HALF)
    assert picture.shape == original.shape

    def detail(image, region):
        y1, y2, x1, x2 = region
        return float(np.asarray(image[y1:y2, x1:x2], np.float64).std())

    person_region = (110, 170, 25, 55)
    assert detail(picture, person_region) < detail(original, person_region) * 0.7, (
        "the person must be pixelated and blurred")
    yellow = overlay.COLOURS["boundary"]
    assert any(tuple(int(c) for c in px) == yellow for px in picture.reshape(-1, 3)), (
        "the outline is drawn after the blur and stays sharp")


def test_the_whole_frame_gets_a_floor_of_blur_even_where_nobody_was_found() -> None:
    """One layer is not enough: a missed detection must still not be an identifiable face."""
    original = noisy()
    empty = verdict_for([])
    picture = overlay.publishable(original, empty, polygon=LEFT_HALF, trace=False)
    far_corner = (30, 90, 300, 380)  # no detection there, and outside the badge and the trace
    y1, y2, x1, x2 = far_corner
    assert float(np.asarray(picture[y1:y2, x1:x2], np.float64).std()) < float(
        np.asarray(original[y1:y2, x1:x2], np.float64).std()) * 0.7


def test_redact_frame_is_the_shared_two_layer_rule() -> None:
    from pitch_occupancy.vision.explain import redact_frame

    original = noisy()
    out = redact_frame(original, [(20, 100, 60, 180)])
    assert out.shape == original.shape
    assert float(np.asarray(out, np.float64).std()) < float(np.asarray(original, np.float64).std())
    # no boxes still gets the floor
    floor_only = redact_frame(original, [])
    assert not np.array_equal(floor_only, original)


def test_an_even_blur_kernel_is_accepted_rather_than_raising() -> None:
    from pitch_occupancy.vision.explain import redact_frame

    assert redact_frame(noisy(), [], floor_kernel=20).shape == (200, 400, 3)


def test_the_legend_names_every_colour_it_draws() -> None:
    labels = dict((text, colour) for text, colour in overlay.legend())
    assert set(labels.values()) >= {overlay.COLOURS["person"], overlay.COLOURS["ball"],
                                    overlay.COLOURS["boundary"],
                                    overlay.COLOURS["person_outside"]}
    assert all(isinstance(text, str) and text for text in labels)


@pytest.mark.parametrize("size", [(200, 400), (1080, 1920), (90, 160)])
def test_it_draws_at_any_frame_size(size) -> None:
    height, width = size
    image = np.full((height, width, 3), 80, np.uint8)
    detections = [person(int(width * 0.05), int(height * 0.5), int(width * 0.1),
                         int(height * 0.9))]
    count = count_inside(detections, LEFT_HALF, size, person_conf=0.25, ball_conf=0.1)
    drawn = overlay.draw_verdict(image, decide(count, motion=None, cfg=CFG), polygon=LEFT_HALF)
    assert drawn.shape == image.shape
