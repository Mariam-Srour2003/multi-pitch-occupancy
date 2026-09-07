"""Camera-view fingerprinting.

The properties that matter are the ones a venue label can get wrong: a view must stay itself
when the lighting changes and the players move, two different views must stay apart, a
genuinely new camera must come back as *new* rather than as the least-bad match, and an
exporter watermark must not be able to vote on where a camera is.

Distances here are compared *relative to each other*, not against absolute constants. The
absolute numbers belong to the real footage and live in `results/camera_fingerprint.csv`;
a synthetic scene has no business fixing them.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitch_occupancy.vision.fingerprint import (
    ViewFingerprint,
    chroma_descriptor,
    cluster_views,
    distance,
    distance_matrix,
    fingerprint_view,
    match_view,
    median_background,
    mutual_nearest_pairs,
    structure_descriptor,
    suggest_threshold,
    view_descriptor,
)


def scene(
    *,
    goal_x: int = 80,
    stand_y: int = 40,
    fence_spacing: int = 90,
    board: tuple[int, int, int] = (190, 90, 40),
    brightness: float = 1.0,
    people: int = 0,
    seed: int = 0,
    watermark: bool = False,
    size: tuple[int, int] = (1920, 1080),
) -> np.ndarray:
    """A synthetic pitch view. Geometry and board colour identify the venue; brightness must not.

    `goal_x`, `stand_y`, `fence_spacing` and `board` are the venue knobs — change one and it is
    a different place. `brightness` and `people` are the nuisance knobs — change them and it
    must still be the same place.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = 110  # turf
    img[:stand_y] = (70, 60, 55)  # skyline band
    img[stand_y : stand_y + h // 6] = board  # perimeter boards
    for x in range(0, w, fence_spacing):  # fence posts
        img[stand_y : stand_y + h // 3, x : x + 6] = (200, 200, 205)
    cv2.rectangle(img, (goal_x, h // 2), (goal_x + 260, h // 2 + 150), (245, 245, 245), 8)
    for i in range(people):  # figures, in different places each call
        cx = int(rng.integers(300, w - 300))
        cy = int(rng.integers(h // 2, h - 200))
        img[cy : cy + 150, cx : cx + 55] = (40, 40, 200)
    if watermark:
        cv2.rectangle(img, (w - 190, h - 90), (w - 20, h - 20), (255, 255, 255), -1)
    out = np.clip(img.astype(np.float32) * brightness, 0, 255)
    return np.clip(out + rng.normal(0, 2, out.shape), 0, 255).astype(np.uint8)


def other_venue(**kw) -> np.ndarray:
    """A scene that differs on every venue knob at once, as two real facilities do."""
    return scene(goal_x=1400, stand_y=180, fence_spacing=280, board=(80, 60, 200), **kw)


def desc(*frames: np.ndarray) -> np.ndarray:
    return view_descriptor(median_background(list(frames)))


# --- the background ---------------------------------------------------------


def test_the_median_removes_the_players() -> None:
    """The whole reason for a median: what identifies a camera is what stayed still."""
    crowded = desc(*[scene(people=4, seed=s) for s in range(9)])
    empty = desc(scene(people=0, seed=99))
    assert distance(crowded, empty) < 0.02


def test_median_background_needs_at_least_one_frame() -> None:
    with pytest.raises(ValueError):
        median_background([])


# --- the descriptor ---------------------------------------------------------


def test_a_view_is_itself() -> None:
    d = desc(scene(seed=1))
    assert distance(d, d) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("fn", [structure_descriptor, chroma_descriptor, view_descriptor])
def test_every_descriptor_block_is_unit_norm(fn) -> None:
    assert np.linalg.norm(fn(median_background([scene(seed=2)]))) == pytest.approx(1.0, abs=1e-5)


def test_lighting_moves_a_descriptor_less_than_geometry_does() -> None:
    """The ordering everything else rests on. Day versus night at one venue is the failure
    mode a plain colour histogram would hit — the audited `teal_pitch` group holds both."""
    base = desc(scene(seed=7))
    relit = desc(scene(brightness=0.3, seed=7))
    elsewhere = desc(other_venue(seed=7))
    assert distance(base, relit) < distance(base, elsewhere)


def test_chromaticity_survives_a_floodlight() -> None:
    """Chromaticity is the separator half of the descriptor, so it has to be the *colour* that
    identifies a venue and not the exposure."""
    day = chroma_descriptor(median_background([scene(brightness=1.0, seed=3)]))
    night = chroma_descriptor(median_background([scene(brightness=0.35, seed=4)]))
    other = chroma_descriptor(median_background([other_venue(seed=3)]))
    assert distance(day, night) < distance(day, other)


def test_each_block_is_sensitive_to_the_cue_it_carries() -> None:
    """Each block against its own noise floor, which is all that can be claimed here.

    Two earlier versions of this test asserted that a repaint moves chromaticity *more* than
    structure. Both were false, in opposite directions, because repainting a board also
    changes its luminance contrast against the turf and so moves the gradients too — by more
    than the coarse chromaticity cells move, in this scene. Cross-block magnitudes are
    scene-dependent and not something this test has measured, so it no longer claims them.
    What each block must do is respond to its own cue far above the noise of re-rendering the
    same scene.
    """
    noise_floor_struct = distance(
        structure_descriptor(median_background([scene(seed=5)])),
        structure_descriptor(median_background([scene(seed=6)])),
    )
    noise_floor_chroma = distance(
        chroma_descriptor(median_background([scene(seed=5)])),
        chroma_descriptor(median_background([scene(seed=6)])),
    )
    layout_change = distance(
        structure_descriptor(median_background([scene(goal_x=80, seed=5)])),
        structure_descriptor(median_background([scene(goal_x=1400, seed=5)])),
    )
    repaint = distance(
        chroma_descriptor(median_background([scene(board=(190, 90, 40), seed=5)])),
        chroma_descriptor(median_background([scene(board=(60, 200, 200), seed=5)])),
    )
    assert layout_change > 10 * noise_floor_struct
    assert repaint > 10 * noise_floor_chroma


def test_moving_the_goal_is_visible_to_the_structure_block() -> None:
    a = structure_descriptor(median_background([scene(goal_x=80, seed=5)]))
    b = structure_descriptor(median_background([scene(goal_x=1400, seed=5)]))
    same = structure_descriptor(median_background([scene(goal_x=80, seed=6)]))
    assert distance(a, b) > 10 * distance(a, same)


def test_the_watermark_corner_cannot_vote() -> None:
    """An exporter logo says who wrote the file, not where the camera is — and it is on every
    clip and absent from venue_01's frames, so it would bias exactly the comparison this
    module exists to make."""
    clean = desc(scene(seed=8))
    stamped = desc(scene(seed=8, watermark=True))
    assert distance(clean, stamped) < 0.02


# --- matching, separation, clustering ---------------------------------------


def fp(camera: str, venue: str, *, other: bool = False, **kw) -> ViewFingerprint:
    frames = [other_venue(**kw)] if other else [scene(**kw)]
    return fingerprint_view(camera, frames, venue=venue)


def two_venues() -> list[ViewFingerprint]:
    return [
        fp("v1_day", "v1", seed=1),
        fp("v1_night", "v1", brightness=0.35, people=3, seed=2),
        fp("v1_busy", "v1", people=5, seed=3),
        fp("v2_day", "v2", other=True, seed=4),
        fp("v2_night", "v2", other=True, brightness=0.4, people=4, seed=5),
    ]


def test_distance_matrix_is_symmetric_with_a_zero_diagonal() -> None:
    m = distance_matrix(two_venues())
    assert m.shape == (5, 5)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0, atol=1e-9)


@pytest.mark.parametrize("block", ["descriptor", "structure", "chroma"])
def test_distance_matrix_can_be_computed_per_block(block: str) -> None:
    """The thesis reports structure and chromaticity apart, so the block has to be selectable
    everywhere, not just in the combined descriptor."""
    assert distance_matrix(two_venues(), block).shape == (5, 5)


def test_suggest_threshold_finds_the_gap_when_there_is_one() -> None:
    sep = suggest_threshold(two_venues())
    assert not sep.overlaps
    assert sep.gap > 0
    assert sep.same_max < sep.threshold < sep.different_min
    assert sep.auc == 1.0


def test_suggest_threshold_reports_an_overlap_instead_of_hiding_it() -> None:
    """When same-venue and different-venue distances interleave, the honest output is
    `overlaps=True` — which is what the real footage gives. A mislabelled view produces it."""
    fps = [
        fp("a1", "v1", seed=1),
        fp("a2", "v1", other=True, seed=2),  # mislabelled on purpose
        fp("b1", "v2", seed=3),
        fp("b2", "v2", seed=4),
    ]
    sep = suggest_threshold(fps)
    assert sep.overlaps
    assert sep.auc < 1.0


def test_suggest_threshold_needs_two_venues() -> None:
    with pytest.raises(ValueError):
        suggest_threshold([fp("a", "v1"), fp("b", "v1", seed=2)])


def test_a_known_view_is_matched_to_its_own_venue() -> None:
    known = [fp("v1_cam", "v1"), fp("v2_cam", "v2", other=True)]
    probe = desc(scene(brightness=0.4, people=3, seed=11))
    m = match_view(probe, known, threshold=suggest_threshold(two_venues()).threshold)
    assert m is not None and m.venue == "v1"


def test_an_unseen_venue_comes_back_as_new_not_as_the_least_bad_match() -> None:
    known = [fp("v1_cam", "v1")]
    stranger = view_descriptor(
        median_background([scene(goal_x=1500, stand_y=250, fence_spacing=330, board=(30, 220, 120))])
    )
    assert match_view(stranger, known, threshold=0.05) is None


def test_match_view_with_nothing_known_is_new() -> None:
    assert match_view(desc(scene()), [], 0.5) is None


def test_mutual_nearest_pairs_finds_the_two_camera_views() -> None:
    """The venue_01 finding in miniature: two cameras, each recorded twice under different
    light, and the pairing that has to come out is same-camera, not same-day."""
    fps = [
        fp("camA_day", "v1", goal_x=80, seed=1),
        fp("camA_night", "v1", goal_x=80, brightness=0.35, seed=2),
        fp("camB_day", "v1", goal_x=1400, stand_y=180, seed=3),
        fp("camB_night", "v1", goal_x=1400, stand_y=180, brightness=0.35, seed=4),
    ]
    pairs = {frozenset((a, b)) for a, b, _ in mutual_nearest_pairs(fps)}
    assert frozenset(("camA_day", "camA_night")) in pairs
    assert frozenset(("camB_day", "camB_night")) in pairs


def test_mutual_nearest_pairs_are_ordered_closest_first() -> None:
    fps = two_venues()
    dists = [d for _, _, d in mutual_nearest_pairs(fps)]
    assert dists == sorted(dists)


def test_clustering_recovers_venues_from_pixels_alone() -> None:
    """The audit this module mechanises: do the views of one facility group together, and do
    two facilities stay apart, without reading anyone's venue label?"""
    fps = two_venues()
    labels = cluster_views(fps, threshold=suggest_threshold(fps).threshold)
    assert len({labels["v1_day"], labels["v1_night"], labels["v1_busy"]}) == 1
    assert len({labels["v2_day"], labels["v2_night"]}) == 1
    assert labels["v1_day"] != labels["v2_day"]


def test_clustering_one_view_is_not_an_error() -> None:
    assert cluster_views([fp("only", "v1")], threshold=0.1) == {"only": 0}
