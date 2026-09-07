"""Near-duplicate detection.

Two properties matter and both were got wrong first time: the grouping must not chain
unrelated frames together, and the hash must be read as a *candidate generator* for label
review rather than as an oracle.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitch_occupancy.data.dedup import (
    DEFAULT_THRESHOLD,
    dhash,
    distinct_subset,
    duplicate_rate,
    find_near_duplicates,
    hamming,
    near_duplicate_pairs,
)


def pitch(seed: int = 0, people: int = 0) -> np.ndarray:
    """A pitch frame, optionally with figures on it."""
    rng = np.random.default_rng(seed)
    img = np.zeros((360, 640, 3), np.uint8)
    img[:, :, 1] = 120
    for x in range(0, 640, 90):
        img[:, x : x + 3] = (230, 230, 230)
    for i in range(people):
        cx, cy = 80 + i * 110, 150 + (i % 3) * 50
        img[cy : cy + 46, cx : cx + 18] = (30, 30, 190)
    return np.clip(img.astype(int) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


# --- the hash ---------------------------------------------------------------


def test_the_same_frame_hashes_to_the_same_value() -> None:
    assert dhash(pitch(1)) == dhash(pitch(1))


def test_brightness_shifts_do_not_change_the_hash() -> None:
    """The whole reason for a gradient hash: a cloud crossing an outdoor pitch changes every
    pixel's value without changing the scene."""
    base = pitch(1)
    brighter = np.clip(base.astype(int) + 35, 0, 255).astype(np.uint8)
    assert hamming(dhash(base), dhash(brighter)) <= 2


def test_a_different_scene_hashes_far_away() -> None:
    other = np.zeros((360, 640, 3), np.uint8)
    other[:, :, 2] = 200
    other[100:300, 200:500] = (20, 20, 20)
    assert hamming(dhash(pitch(1)), dhash(other)) > DEFAULT_THRESHOLD


def test_hamming_is_symmetric_and_zero_on_itself() -> None:
    a, b = dhash(pitch(1)), dhash(pitch(2))
    assert hamming(a, a) == 0 and hamming(a, b) == hamming(b, a)


# --- what the hash can and cannot see ---------------------------------------


def test_several_figures_move_the_hash() -> None:
    """Measured on the real footage: frames with three or four players plus a cone sit 5-6
    bits from an empty frame of the same pitch."""
    assert hamming(dhash(pitch(1)), dhash(pitch(1, people=4))) >= 3


def test_one_small_figure_may_not_move_the_hash_at_all() -> None:
    """The limit that matters for label review, and the reason this is not an oracle.

    An 8x8 downscale erases a single person. On the real data three correctly-labelled
    MAINTENANCE frames sat at distance 0-1 from empty frames because each holds one person
    near the edge - flagged as suspicious, and correct on inspection.
    """
    lone = pitch(1).copy()
    lone[300:330, 600:612] = (30, 30, 190)  # one small figure near the edge
    assert hamming(dhash(pitch(1)), dhash(lone)) <= 2


# --- grouping, and the mistake it encodes -----------------------------------


def test_pairs_are_direct_and_never_transitive() -> None:
    """A chain must not become a duplicate.

    Single-link grouping reported 96.9% of the real dataset as redundant with "52 distinct
    scenes", because a fixed camera drifts and every frame is near its neighbour in time.
    Its own output refuted it: the largest group spanned 512 frames with a maximum internal
    distance of 19 against a threshold of 6, mixing EMPTY with ACTIVE_PLAY.
    """
    a, b, c = 0b0000, 0b0011, 0b1111  # a-b = 2, b-c = 2, a-c = 4
    hashes = {"a": a, "b": b, "c": c}
    pairs = near_duplicate_pairs(hashes, threshold=2)
    assert {tuple(sorted(p[:2])) for p in pairs} == {("a", "b"), ("b", "c")}
    # the chain still connects all three, which is why grouping overcounts
    assert len(find_near_duplicates(hashes, threshold=2)[0].members) == 3


def test_duplicate_rate_counts_frames_with_a_twin_not_chain_members() -> None:
    hashes = {"a": 0b0000, "b": 0b0001, "c": 0b1111_1111}
    assert duplicate_rate(hashes, threshold=1) == pytest.approx(2 / 3)


def test_a_frame_with_no_twin_is_not_counted() -> None:
    assert duplicate_rate({"only": 0b1010}, threshold=2) == 0.0


def test_empty_input_is_zero_not_an_error() -> None:
    assert duplicate_rate({}) == 0.0
    assert near_duplicate_pairs({}) == []


def test_threshold_zero_finds_only_identical_hashes() -> None:
    hashes = {"a": 0b1010, "b": 0b1010, "c": 0b1011}
    pairs = near_duplicate_pairs(hashes, threshold=0)
    assert {tuple(sorted(p[:2])) for p in pairs} == {("a", "b")}


def test_group_records_its_worst_internal_distance() -> None:
    """Without this the chaining problem is invisible in the output."""
    group = find_near_duplicates({"a": 0b0000, "b": 0b0011, "c": 0b1111}, threshold=2)[0]
    assert group.max_distance == 4  # larger than the threshold that built it
    assert group.redundant == 2


# --- distinct_subset: the effective-sample rule ----------------------------


def test_distinct_subset_keeps_one_frame_per_near_duplicate_cluster() -> None:
    """Two near-identical frames count as one observation, not two.

    This is the rule behind "243 held-out empty frames are three to ten distinct scenes",
    which is what killed six comparisons that had looked significant at p down to 8e-53.
    """
    hashes = {"a": 0b0000, "a_twin": 0b0001, "b": 0b11111111, "b_twin": 0b11111110}
    assert distinct_subset(hashes, threshold=2) == ["a", "b"]


def test_distinct_subset_keeps_everything_when_nothing_is_similar() -> None:
    hashes = {"a": 0x0000, "b": 0xFFFF, "c": 0x00FF}
    assert distinct_subset(hashes, threshold=2) == ["a", "b", "c"]


def test_distinct_subset_collapses_identical_frames_to_one() -> None:
    assert distinct_subset({"a": 7, "b": 7, "c": 7}) == ["a"]


def test_distinct_subset_is_empty_for_empty_input() -> None:
    assert distinct_subset({}) == []


def test_distinct_subset_respects_insertion_order_for_the_tie_break() -> None:
    """Greedy keeps the first of a cluster, so the caller controls which frame survives.

    Relied on by the audit, which wants a stable answer across runs rather than the
    largest possible independent set - a maximum independent set is NP-hard, and the count
    was checked to be stable across five shuffles anyway.
    """
    a_first = distinct_subset({"a": 0b0000, "b": 0b0001}, threshold=2)
    b_first = distinct_subset({"b": 0b0001, "a": 0b0000}, threshold=2)
    assert a_first == ["a"] and b_first == ["b"]
    assert len(a_first) == len(b_first)


def test_distinct_subset_threshold_is_exclusive_like_the_audit_expects() -> None:
    """`> threshold` to keep, so a pair exactly at the threshold is a duplicate.

    Matches `near_duplicate_pairs`, which treats `<= threshold` as similar. The two rules
    must agree on the boundary or a frame could be both a near-duplicate and a distinct
    scene.
    """
    hashes = {"a": 0b0000, "b": 0b0011}  # distance 2
    assert distinct_subset(hashes, threshold=2) == ["a"]
    assert distinct_subset(hashes, threshold=1) == ["a", "b"]
    assert len(near_duplicate_pairs(hashes, threshold=2)) == 1
    assert len(near_duplicate_pairs(hashes, threshold=1)) == 0
