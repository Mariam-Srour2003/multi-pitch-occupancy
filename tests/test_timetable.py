"""Booking slots from a day of verdicts: where the slots fall, how they are judged, and how
they meet the booking records.

Every number here can be checked by hand. That is the reason `timetable` takes verdicts rather
than footage: the rules that decide whether a customer is reported as a no-show should not
need a video to be tested.
"""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from pitch_occupancy.bookings import BookingRecord
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.timetable import (
    Rules,
    SlotResult,
    SlotStatus,
    compare,
    slot_windows,
    summarise,
    to_csv,
)
from pitch_occupancy.vision.rules import MinuteState

P, E, M = Class3.ACTIVE_PLAY, Class3.EMPTY, Class3.MAINTENANCE_NON_SPORTING


def _minutes(*runs: tuple[object, int], start_s: float = 0.0) -> list[tuple[float, object]]:
    """One sample a minute: ``(state, n)`` runs laid end to end."""
    out, t = [], start_s
    for state, n in runs:
        for _ in range(n):
            out.append((t, state))
            t += 60
    return out


# --- windows --------------------------------------------------------------------------------


def test_slots_follow_the_video_and_keep_a_partial_last_slot() -> None:
    windows = slot_windows(time(8, 0), 4 * 3600, 90)
    assert [w.label for w in windows] == ["08:00-09:30", "09:30-11:00", "11:00-12:30"]
    assert windows[0].coverage == 1.0
    assert windows[-1].coverage == pytest.approx(60 / 90)
    assert windows[-1].covered_end_s == 4 * 3600


def test_a_later_first_slot_leaves_the_minutes_before_it_unassigned() -> None:
    windows = slot_windows(time(7, 45), 2 * 3600, 60, first_slot=time(8, 0))
    assert windows[0].label == "08:00-09:00"
    assert windows[0].start_s == 15 * 60
    samples = _minutes((P, 15), (E, 60))  # the first 15 minutes are before any slot
    first = summarise(samples, windows, 60)[0]
    assert first.playing_s == 0 and first.empty_s == 3600


def test_an_earlier_first_slot_is_partly_covered() -> None:
    windows = slot_windows(time(8, 15), 3600, 30, first_slot=time(8, 0))
    assert windows[0].label == "08:00-08:30"
    assert windows[0].start_s == -15 * 60
    assert windows[0].coverage == pytest.approx(0.5)


def test_slots_cross_midnight_with_a_day_offset() -> None:
    windows = slot_windows(time(23, 0), 3 * 3600, 60)
    assert [w.label for w in windows] == ["23:00-00:00", "00:00-01:00 (+1d)",
                                          "01:00-02:00 (+1d)"]
    assert [w.day_offset for w in windows] == [0, 1, 1]
    assert windows[1].starts_on(date(2026, 7, 12)) == datetime(2026, 7, 13, 0, 0)
    # first_slot after midnight is read as the next morning, not the previous one
    later = slot_windows(time(23, 0), 3 * 3600, 60, first_slot=time(0, 30))
    assert later[0].start_s == 90 * 60 and later[0].day_offset == 1


def test_no_video_no_slots_and_a_bad_length_is_refused() -> None:
    assert slot_windows(time(8, 0), 0, 90) == []
    with pytest.raises(ValueError):
        slot_windows(time(8, 0), 3600, 0)


# --- summarise ------------------------------------------------------------------------------


def test_mostly_playing_is_used_and_mostly_empty_is_not() -> None:
    windows = slot_windows(time(8, 0), 2 * 3600, 60)
    used, empty = summarise(_minutes((P, 55), (E, 5), (E, 50), (M, 10)), windows, 60)
    assert used.status is SlotStatus.USED and used.main == P.value
    assert used.playing_s == 55 * 60 and used.seen_share == 1.0
    assert empty.status is SlotStatus.NOT_USED and empty.main == E.value
    assert "empty" in empty.reason


def test_less_than_half_seen_is_review() -> None:
    windows = slot_windows(time(8, 0), 3600, 60)
    (res,) = summarise(_minutes((P, 25), (MinuteState.UNCERTAIN, 20), (None, 15)), windows, 60)
    assert res.status is SlotStatus.REVIEW
    assert res.uncertain_s == 35 * 60
    assert "Less than half" in res.reason


def test_a_slot_mostly_outside_the_video_is_review_not_unused() -> None:
    windows = slot_windows(time(8, 0), 80 * 60, 60)
    last = summarise(_minutes((E, 80)), windows, 60)[-1]
    assert last.window.coverage == pytest.approx(1 / 3)
    assert last.status is SlotStatus.REVIEW
    assert "outside the video" in last.reason


def test_a_close_call_is_review() -> None:
    windows = slot_windows(time(8, 0), 3600, 60)
    (res,) = summarise(_minutes((P, 30), (E, 30)), windows, 60)
    assert res.status is SlotStatus.REVIEW and "Close call" in res.reason
    (res,) = summarise(_minutes((P, 37), (E, 23)), windows, 60)  # 62%: clear enough
    assert res.status is SlotStatus.USED


def test_a_tie_for_main_activity_is_mixed() -> None:
    windows = slot_windows(time(8, 0), 3600, 60)
    (res,) = summarise(_minutes((E, 30), (M, 30)), windows, 60)
    assert res.main == "mixed" and res.status is SlotStatus.NOT_USED
    (res,) = summarise(_minutes((P, 21), (M, 21), (E, 18)), windows, 60)
    assert res.main == "mixed" and res.status is SlotStatus.NOT_USED  # 35% play, not close


def test_states_may_be_strings_or_minute_states_and_samples_split_at_edges() -> None:
    windows = slot_windows(time(8, 0), 2 * 3600, 60)
    samples = [(t, "C2_ACTIVE_PLAY") for t in range(30, 7200, 60)]
    first, second = summarise(samples, windows, 60)
    assert first.playing_s == pytest.approx(3600 - 30)  # the first 30 s was never sampled
    assert second.playing_s == pytest.approx(3600)  # the 59:30 sample's tail lands here
    (res,) = summarise([(0, MinuteState.ACTIVE_PLAY)], slot_windows(time(8, 0), 60, 1), 60)
    assert res.playing_s == 60


def test_the_rules_can_be_changed_in_one_place() -> None:
    windows = slot_windows(time(8, 0), 3600, 60)
    samples = _minutes((P, 30), (E, 30))
    (res,) = summarise(samples, windows, 60, rules=Rules(close_low=0.2, close_high=0.3))
    assert res.status is SlotStatus.NOT_USED  # a tie is "mixed", and mixed is not playing


def test_a_result_round_trips_through_its_dict() -> None:
    windows = slot_windows(time(23, 30), 3600, 45)
    for res in summarise(_minutes((P, 60)), windows, 60):
        assert SlotResult.from_dict(res.to_dict()) == res


# --- compare --------------------------------------------------------------------------------


DAY = date(2026, 7, 13)


def _booking(start: str, end: str, status: str = "confirmed", ref: str = "BK-1",
             field: str = "field_01", day: date = DAY) -> BookingRecord:
    return BookingRecord(field_id=field, date=day, start=time.fromisoformat(start),
                         end=time.fromisoformat(end), customer_ref=ref, status=status,
                         entered_by="t", source="test")


def test_every_outcome() -> None:
    windows = slot_windows(time(8, 0), 6 * 3600, 60)
    states = [(P, 60), (E, 60), (P, 60), (E, 60), (P, 60), (P, 30)]
    results = summarise(_minutes(*[(s, n) for s, n in states], (E, 30)), windows, 60)
    assert [r.status for r in results] == [SlotStatus.USED, SlotStatus.NOT_USED,
                                           SlotStatus.USED, SlotStatus.NOT_USED,
                                           SlotStatus.USED, SlotStatus.REVIEW]
    records = [
        _booking("08:00", "09:00", ref="BK-A"),
        _booking("09:00", "10:00", status="no_show", ref="BK-B"),
        # 10:00 nothing booked, and play was seen
        # 11:00 nothing booked, and empty
        _booking("12:00", "13:00", status="maintenance", ref=""),
        _booking("13:00", "14:00", ref="BK-F"),
        _booking("08:00", "09:00", ref="OTHER", field="field_02"),
        _booking("10:00", "11:00", ref="YESTERDAY", day=date(2026, 7, 12)),
        _booking("11:00", "12:00", status="cancelled", ref="BK-X"),
    ]
    checks = compare(results, records, field_id="field_01", day=DAY)
    assert [c.outcome for c in checks] == [
        "booked_and_used", "booked_not_used", "used_not_booked", "not_booked_not_used",
        "off_sale_but_used", "review"]
    assert checks[0].customer_refs == ["BK-A"] and checks[0].ok
    assert checks[1].customer_refs == ["BK-B"] and not checks[1].ok
    assert "no-show" in checks[1].message
    assert checks[5].ok and "BK-F" in checks[5].message  # never an accusation


def test_a_booking_must_cover_half_the_slot() -> None:
    windows = slot_windows(time(8, 0), 90 * 60, 90)
    (res,) = summarise(_minutes((E, 90)), windows, 60)
    short = compare([res], [_booking("08:00", "08:30")], field_id="field_01", day=DAY)
    half = compare([res], [_booking("08:45", "10:00")], field_id="field_01", day=DAY)
    assert short[0].outcome == "not_booked_not_used"
    assert half[0].outcome == "booked_not_used"


def test_a_slot_after_midnight_meets_the_next_days_bookings() -> None:
    windows = slot_windows(time(23, 0), 2 * 3600, 60)
    results = summarise(_minutes((E, 60), (P, 60)), windows, 60)
    records = [_booking("00:00", "01:00", ref="LATE", day=date(2026, 7, 14))]
    checks = compare(results, records, field_id="field_01", day=DAY)
    assert [c.outcome for c in checks] == ["not_booked_not_used", "booked_and_used"]


def test_csv_has_one_row_per_slot_and_the_booking_columns_when_compared() -> None:
    windows = slot_windows(time(8, 0), 2 * 3600, 60)
    results = summarise(_minutes((P, 60), (E, 60)), windows, 60)
    plain = to_csv(results).strip().splitlines()
    assert len(plain) == 3 and plain[0].startswith("slot,") and "USED" in plain[1]
    checks = compare(results, [_booking("08:00", "09:00")], field_id="field_01", day=DAY)
    both = to_csv([r.to_dict() for r in results], [c.to_dict() for c in checks])
    assert "booking_outcome" in both and "booked_and_used" in both and "BK-1" in both
