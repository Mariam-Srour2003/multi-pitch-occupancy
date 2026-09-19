"""Abstained minutes in the slot aggregation (A36).

An UNCERTAIN minute is not a sample. It counts against the capture floor like a minute with
no frame, so it can push a slot toward REVIEW and nowhere else - the REVIEW-never-accuses
rule of `slots/reconcile.py` is untouched.
"""

from __future__ import annotations

import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.slots.aggregate import UNCERTAIN, SlotVerdict, aggregate_slot

PLAY, EMPTY = Class3.ACTIVE_PLAY, Class3.EMPTY


def test_abstentions_count_against_the_capture_floor_and_nothing_else() -> None:
    """A36 registers one mechanism: an abstained minute is a minute not captured. So 40%
    abstained still clears the 0.5 floor and the decided minutes decide; 60% abstained does
    not, and the slot is REVIEW with a reason that names the abstentions. No second
    threshold - that would be a hand-picked hyper-parameter, which the module's own header
    warns against."""
    verdict = aggregate_slot([PLAY] * 6 + [None] * 4, minutes_expected=10)
    assert verdict.status is SlotStatus.USED
    assert verdict.uncertain_ratio == pytest.approx(0.4)
    assert verdict.n_samples == 6, "the abstentions are not samples"

    verdict = aggregate_slot([PLAY] * 4 + [None] * 6, minutes_expected=10)
    assert verdict.status is SlotStatus.REVIEW
    assert "abstained" in verdict.reason and "4 of 10" in verdict.reason
    assert verdict.uncertain_ratio == pytest.approx(0.6)


def test_a_few_abstentions_do_not_change_a_clear_verdict() -> None:
    states = [EMPTY] * 9 + [UNCERTAIN]
    verdict = aggregate_slot(states, minutes_expected=10)
    assert verdict.status is SlotStatus.NOTUSED
    assert verdict.uncertain_ratio == pytest.approx(0.1)
    assert verdict.empty_ratio == 1.0, "ratios are over the decided minutes"


def test_abstention_never_manufactures_a_verdict_the_decided_minutes_would_not_give() -> None:
    """Whatever the abstentions, the verdict is either what the decided minutes alone say or
    REVIEW - never the other decided outcome."""
    for decided in ([PLAY] * 8, [EMPTY] * 8, [PLAY] * 4 + [EMPTY] * 4):
        alone = aggregate_slot(decided, minutes_expected=len(decided)).status
        for n_uncertain in (1, 2, 3, 8):
            with_abstentions = aggregate_slot(decided + [None] * n_uncertain,
                                              minutes_expected=len(decided) + n_uncertain).status
            assert with_abstentions in {alone, SlotStatus.REVIEW}, (decided, n_uncertain)


def test_all_abstained_is_review_with_a_reason_that_says_so() -> None:
    verdict = aggregate_slot([None, UNCERTAIN, None], minutes_expected=3)
    assert verdict.status is SlotStatus.REVIEW
    assert "abstained on all 3" in verdict.reason
    assert verdict.uncertain_ratio == 1.0 and verdict.n_samples == 0


def test_the_four_folder_names_are_accepted_alongside_the_three_classes() -> None:
    """The detector-first path speaks `MinuteState`; the probe path speaks `Class3`; a
    caller may hand either, and `3_people_not_playing` and `4_maintenance` both land on C3."""
    verdict = aggregate_slot(["2_playing"] * 5 + ["1_empty"] * 3 + ["3_people_not_playing",
                                                                     "4_maintenance"])
    assert verdict.status is SlotStatus.USED
    assert verdict.maintenance_ratio == pytest.approx(0.2)
    assert verdict.play_ratio == pytest.approx(0.5)


def test_without_minutes_expected_abstentions_count_against_the_observed_total() -> None:
    verdict = aggregate_slot([PLAY, PLAY, None, None])
    assert verdict.uncertain_ratio == pytest.approx(0.5)
    assert verdict.status is SlotStatus.USED, "no capture floor without minutes_expected"


def test_the_old_call_shape_is_unchanged() -> None:
    verdict = aggregate_slot([PLAY] * 6 + [EMPTY] * 4, [0.9] * 10)
    assert isinstance(verdict, SlotVerdict)
    assert verdict.status is SlotStatus.USED and verdict.uncertain_ratio == 0.0
    assert verdict.mean_confidence == pytest.approx(0.9)
