"""Transitions and slot conditions - the two ideas that make a verdict explainable.

A REVIEW that says only "intermittent activity" is a puzzle for whoever opens it. These
record what the run already knew and was throwing away."""

from __future__ import annotations

import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.slots.conditions import summarise_conditions
from pitch_occupancy.slots.evidence import (
    find_transitions,
    select_evidence_around_transitions,
)

PLAY, EMPTY, MAINT = Class3.ACTIVE_PLAY, Class3.EMPTY, Class3.MAINTENANCE_NON_SPORTING


# --- transitions ------------------------------------------------------------


def test_finds_an_abandoned_match() -> None:
    """Play then empty is a different event from never used, and only this tells them apart."""
    t = find_transitions([PLAY] * 22 + [EMPTY] * 38)
    assert len(t) == 1
    assert t[0].minute == 22
    assert (t[0].before, t[0].after) == (PLAY, EMPTY)


def test_finds_a_late_start() -> None:
    t = find_transitions([EMPTY] * 15 + [PLAY] * 45)
    assert t[0].after is PLAY and t[0].minute == 15


def test_a_single_stray_minute_is_not_a_transition() -> None:
    """A player walking through frame must not read as a match ending."""
    assert find_transitions([EMPTY] * 30 + [PLAY] + [EMPTY] * 29) == []


def test_short_runs_on_either_side_are_ignored() -> None:
    assert find_transitions([PLAY] * 3 + [EMPTY] * 57, min_stable=5) == []
    assert find_transitions([PLAY] * 57 + [EMPTY] * 3, min_stable=5) == []


def test_a_uniform_slot_has_no_transitions() -> None:
    assert find_transitions([PLAY] * 60) == []


def test_multiple_transitions_are_all_reported() -> None:
    t = find_transitions([EMPTY] * 10 + [PLAY] * 20 + [MAINT] * 10 + [EMPTY] * 20)
    assert len(t) == 3


def test_no_states_yields_nothing() -> None:
    assert find_transitions([]) == []


# --- evidence around a transition -------------------------------------------


def test_evidence_brackets_the_change() -> None:
    """"Here is minute 20 and here is minute 22" settles a dispute; three similar frames
    do not."""
    states = [PLAY] * 22 + [EMPTY] * 38
    samples = [(m, s, 0.9, f"f{m}.jpg") for m, s in enumerate(states)]
    ev = select_evidence_around_transitions(samples, states, SlotStatus.REVIEW)
    minutes = [e.minute_index for e in ev]
    assert minutes == [20, 22, 24]
    assert ev[0].predicted is PLAY and ev[-1].predicted is EMPTY


def test_falls_back_to_thirds_when_nothing_changed() -> None:
    states = [PLAY] * 60
    samples = [(m, s, 0.9, None) for m, s in enumerate(states)]
    ev = select_evidence_around_transitions(samples, states, SlotStatus.USED)
    assert len({e.third for e in ev}) == 3  # spread across the slot, as before


def test_a_transition_at_the_slot_edge_stays_in_range() -> None:
    states = [PLAY] * 6 + [EMPTY] * 54
    samples = [(m, s, 0.9, None) for m, s in enumerate(states)]
    ev = select_evidence_around_transitions(samples, states, SlotStatus.REVIEW)
    assert all(0 <= e.minute_index < 60 for e in ev)


# --- conditions -------------------------------------------------------------


def test_capture_rate_reflects_missing_minutes() -> None:
    c = summarise_conditions(
        minutes_expected=60, confidences=[0.9] * 48, disagreements=[False] * 48,
        cameras_seen=2,
    )
    assert c.minutes_missed == 12
    assert c.capture_rate == pytest.approx(0.8)


def test_concerns_name_a_partial_capture() -> None:
    c = summarise_conditions(
        minutes_expected=60, confidences=[0.9] * 20, disagreements=[False] * 20,
        cameras_seen=2,
    )
    assert any("20 of 60" in x for x in c.concerns())


def test_concerns_name_a_single_camera() -> None:
    c = summarise_conditions(
        minutes_expected=60, confidences=[0.95] * 60, disagreements=[False] * 60,
        cameras_seen=1,
    )
    assert any("half the pitch is unseen" in x for x in c.concerns())


def test_concerns_name_persistent_disagreement() -> None:
    """Halves that keep disagreeing can mean an occluded or dirty lens."""
    c = summarise_conditions(
        minutes_expected=60, confidences=[0.95] * 60,
        disagreements=[True] * 30 + [False] * 30, cameras_seen=2,
    )
    assert any("disagreed" in x for x in c.concerns())


def test_concerns_name_unreadable_footage() -> None:
    c = summarise_conditions(
        minutes_expected=60, confidences=[0.9] * 60, disagreements=[False] * 60,
        cameras_seen=2, contrasts=[18.0] * 60,
    )
    assert any("hard to read" in x for x in c.concerns())


def test_a_clean_slot_raises_no_concerns() -> None:
    """An empty list means the observation was unremarkable - the verdict may still be
    REVIEW because the pitch genuinely was ambiguous, which is a different conversation."""
    c = summarise_conditions(
        minutes_expected=60, confidences=[0.95] * 60, disagreements=[False] * 60,
        cameras_seen=2, contrasts=[55.0] * 60,
    )
    assert c.concerns() == []


def test_mixed_lighting_is_reported_as_mixed() -> None:
    c = summarise_conditions(
        minutes_expected=2, confidences=[0.9, 0.9], disagreements=[False, False],
        cameras_seen=2, lighting=["day", "night"],
    )
    assert c.lighting == "mixed"


def test_weather_is_reserved_and_unset() -> None:
    """Nothing measures it yet; the field exists so reconciliation has somewhere to look."""
    c = summarise_conditions(
        minutes_expected=1, confidences=[0.9], disagreements=[False], cameras_seen=2,
    )
    assert c.weather is None
    assert "weather" in c.as_dict()


def test_an_empty_slot_does_not_divide_by_zero() -> None:
    c = summarise_conditions(
        minutes_expected=60, confidences=[], disagreements=[], cameras_seen=0,
    )
    assert c.capture_rate == 0.0
    assert c.disagreement_rate == 0.0


# --- a camera gap must not move the clock ---------------------------------
#
# worker.py appends nothing for a minute no camera could be read, so the state sequence is
# compacted while the clock is not. `find_transitions` reported a list *position* in a field
# called `minute`, and the evidence selector looked it up in a dict keyed by real minute.
# Identical while every minute was observed; wrong for every slot with an outage.

PLAY, EMPTY = Class3.ACTIVE_PLAY, Class3.EMPTY

#: Play for minutes 0-19, then empty. The change is at minute 20.
_STATES = [PLAY] * 20 + [EMPTY] * 40


def _rows(pairs):
    return [(m, s, 0.9, f"f{m}.jpg") for m, s in pairs]


def test_a_transition_reports_a_minute_not_a_position() -> None:
    """With the first ten minutes lost, position 10 is minute 20."""
    kept = [(m, s) for m, s in enumerate(_STATES) if m >= 10]
    t = find_transitions([s for _, s in kept], minutes=[m for m, _ in kept])[0]
    assert t.minute == 20, "the clock time of the change"
    assert t.index == 10, "and its position in the compacted sequence"


def test_without_minutes_the_transition_falls_back_to_the_position() -> None:
    """Backwards compatible: a gapless caller needs no extra argument."""
    t = find_transitions(_STATES)[0]
    assert t.minute == 20 and t.index == 20


def test_evidence_straddles_the_change_even_when_minutes_are_missing() -> None:
    """The whole point of transition-aware evidence, under an outage.

    Before the fix this returned minutes 10 and 12 - two frames instead of three, both
    still showing ACTIVE_PLAY, for a slot whose play stopped at minute 20. A reviewer
    settling a dispute would have seen no evidence of the change at all.
    """
    kept = [(m, s) for m, s in enumerate(_STATES) if m >= 10]
    ev = select_evidence_around_transitions(
        _rows(kept), [s for _, s in kept], SlotStatus.REVIEW
    )
    assert [e.minute_index for e in ev] == [18, 20, 22]
    assert [e.predicted for e in ev] == [PLAY, EMPTY, EMPTY], "both states must appear"


def test_a_gap_and_no_gap_choose_the_same_evidence() -> None:
    """The outage removed minutes 0-9; it should not change what explains the slot."""
    kept = [(m, s) for m, s in enumerate(_STATES) if m >= 10]
    with_gap = select_evidence_around_transitions(
        _rows(kept), [s for _, s in kept], SlotStatus.REVIEW
    )
    without = select_evidence_around_transitions(
        _rows(list(enumerate(_STATES))), _STATES, SlotStatus.REVIEW
    )
    assert [e.minute_index for e in with_gap] == [e.minute_index for e in without]


def test_three_frames_survive_a_gap_around_the_transition_itself() -> None:
    """Neighbours come from the observed sequence, not from arithmetic on the clock.

    Minute-2 may simply not exist after an outage; the observation two *samples* earlier
    does, and it is the honest stand-in.
    """
    kept = [(m, s) for m, s in enumerate(_STATES) if not 14 <= m <= 19]
    ev = select_evidence_around_transitions(
        _rows(kept), [s for _, s in kept], SlotStatus.REVIEW
    )
    assert len(ev) == 3, "a gap next to the change must not cost a frame"
    assert {e.predicted for e in ev} == {PLAY, EMPTY}


def test_misaligned_minutes_raise_rather_than_silently_shift_the_clock() -> None:
    with pytest.raises(ValueError, match="align"):
        find_transitions(_STATES, minutes=list(range(len(_STATES) - 1)))
