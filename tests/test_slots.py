"""Fusion and slot aggregation.

These turn predictions into billing decisions, so their edge cases are the ones that
generate disputes: a half-empty camera during a real match, a slot with no footage, a
groundsman mistaken for a player."""

from __future__ import annotations

import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.slots.aggregate import (
    SlotVerdict,
    Thresholds,
    aggregate_slot,
    tune_thresholds,
)
from pitch_occupancy.slots.fusion import fuse

PLAY, EMPTY, MAINT = Class3.ACTIVE_PLAY, Class3.EMPTY, Class3.MAINTENANCE_NON_SPORTING


# --- fusion -----------------------------------------------------------------


def test_play_on_one_half_wins() -> None:
    """The documented failure case: a goalkeeper-only half during a live match."""
    f = fuse({"camA": (EMPTY, 0.97), "camB": (PLAY, 0.88)})
    assert f.state is PLAY
    assert f.disagreed


def test_empty_requires_both_halves() -> None:
    f = fuse({"camA": (EMPTY, 0.9), "camB": (EMPTY, 0.95)})
    assert f.state is EMPTY
    assert not f.disagreed


def test_maintenance_outranks_empty_but_not_play() -> None:
    assert fuse({"a": (MAINT, 0.8), "b": (EMPTY, 0.9)}).state is MAINT
    assert fuse({"a": (MAINT, 0.99), "b": (PLAY, 0.6)}).state is PLAY


def test_confidence_comes_from_the_winning_camera_not_an_average() -> None:
    """Averaging in a camera that saw nothing would dilute the one that saw the match."""
    f = fuse({"camA": (EMPTY, 0.99), "camB": (PLAY, 0.70)})
    assert f.confidence == pytest.approx(0.70)


def test_highest_confidence_wins_among_agreeing_cameras() -> None:
    assert fuse({"a": (PLAY, 0.6), "b": (PLAY, 0.9)}).confidence == pytest.approx(0.9)


def test_no_cameras_raises_rather_than_defaulting_to_empty() -> None:
    """Inventing EMPTY here would record real usage as an unused slot."""
    with pytest.raises(ValueError, match="no readable camera"):
        fuse({})


def test_accepts_raw_class_strings() -> None:
    assert fuse({"a": ("C2_ACTIVE_PLAY", 0.8)}).state is PLAY


def test_single_camera_still_fuses() -> None:
    f = fuse({"camA": (PLAY, 0.8)})
    assert f.state is PLAY
    assert not f.disagreed
    assert f.n_cameras == 1


# --- aggregation ------------------------------------------------------------


def test_sustained_play_is_used() -> None:
    v = aggregate_slot([PLAY] * 40 + [EMPTY] * 20)
    assert v.status is SlotStatus.USED


def test_an_empty_slot_is_notused() -> None:
    v = aggregate_slot([EMPTY] * 58 + [PLAY] * 2)
    assert v.status is SlotStatus.NOTUSED


def test_intermittent_activity_goes_to_review() -> None:
    """Neither band met - exactly what REVIEW is for."""
    v = aggregate_slot([PLAY] * 12 + [EMPTY] * 48)
    assert v.status is SlotStatus.REVIEW
    assert "intermittent" in v.reason


def test_maintenance_dominated_slot_goes_to_review_not_used() -> None:
    v = aggregate_slot([MAINT] * 40 + [EMPTY] * 20)
    assert v.status is SlotStatus.REVIEW
    assert "maintenance" in v.reason


def test_low_confidence_forces_review_regardless_of_ratios() -> None:
    """The audit must not accuse on footage it could barely read."""
    th = Thresholds(review_below_confidence=0.6)
    v = aggregate_slot([PLAY] * 60, [0.4] * 60, th)
    assert v.status is SlotStatus.REVIEW
    assert "confidence" in v.reason


def test_high_confidence_passes_the_gate() -> None:
    th = Thresholds(review_below_confidence=0.6)
    assert aggregate_slot([PLAY] * 60, [0.9] * 60, th).status is SlotStatus.USED


def test_no_samples_is_review_not_notused() -> None:
    """A camera outage is not evidence the pitch was empty."""
    v = aggregate_slot([])
    assert v.status is SlotStatus.REVIEW
    assert v.n_samples == 0


def test_ratios_sum_to_one() -> None:
    v = aggregate_slot([PLAY] * 30 + [EMPTY] * 20 + [MAINT] * 10)
    assert v.play_ratio + v.empty_ratio + v.maintenance_ratio == pytest.approx(1.0)


def test_verdict_reason_is_populated() -> None:
    assert aggregate_slot([PLAY] * 60).reason


def test_boundary_exactly_at_the_used_threshold_counts_as_used() -> None:
    v = aggregate_slot([PLAY] * 35 + [EMPTY] * 65)
    assert v.play_ratio == pytest.approx(0.35)
    assert v.status is SlotStatus.USED


# --- threshold tuning -------------------------------------------------------


def test_tuning_recovers_a_separable_rule() -> None:
    slots = [
        ([PLAY] * 50 + [EMPTY] * 10, SlotStatus.USED),
        ([PLAY] * 45 + [EMPTY] * 15, SlotStatus.USED),
        ([EMPTY] * 60, SlotStatus.NOTUSED),
        ([EMPTY] * 58 + [PLAY] * 2, SlotStatus.NOTUSED),
    ]
    _, acc = tune_thresholds(slots)
    assert acc == 1.0


def test_tuning_never_produces_overlapping_bands() -> None:
    slots = [([PLAY] * 60, SlotStatus.USED), ([EMPTY] * 60, SlotStatus.NOTUSED)]
    th, _ = tune_thresholds(slots)
    assert th.notused_max_play < th.used_min_play


def test_tuning_without_slots_raises() -> None:
    with pytest.raises(ValueError, match="without labelled slots"):
        tune_thresholds([])


def test_tuned_thresholds_beat_or_match_the_defaults() -> None:
    """The point of tuning: STAN must beat a *fitted* baseline, not a guessed one."""
    slots = [
        ([PLAY] * 20 + [EMPTY] * 40, SlotStatus.USED),  # only 33% play, but truly used
        ([EMPTY] * 60, SlotStatus.NOTUSED),
    ]
    default_acc = sum(
        aggregate_slot(s).status is truth for s, truth in slots
    ) / len(slots)
    _, tuned_acc = tune_thresholds(slots)
    assert tuned_acc >= default_acc
