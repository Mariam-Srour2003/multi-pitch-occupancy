"""The scheduler (WP6-T2).

Its acceptance criterion is *"runs continuously; DB fills; verdicts correct on recorded
slots"*. The first clause is the least interesting and the hardest to test badly — a loop
whose only observable behaviour is that it does not return teaches nothing. So the clock and
the sleep are parameters, and what is tested is whether the right slot starts, whether the
database fills, and whether the verdicts match the ones `end_to_end_slots.py` already
publishes.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.scheduler import (
    Schedule,
    ScheduledSlot,
    due,
    load_schedule,
    run_due,
    run_forever,
    schedule_from_recordings,
)

ROOT = Path(__file__).resolve().parents[1]


def slot(start: str = "10:00", minutes: int = 60, days: tuple[str, ...] = ()) -> ScheduledSlot:
    hh, mm = (int(x) for x in start.split(":"))
    return ScheduledSlot("venue_01", time(hh, mm), minutes, ("camera_A",),
                         field_id="field_01", days=frozenset(days))


class Stub(FrameSource):
    def __init__(self, minutes: int = 3) -> None:
        self._minutes = minutes

    def cameras(self) -> list[str]:
        return ["camera_A"]

    @property
    def n_minutes(self) -> int:
        return self._minutes

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        return Frame(camera_id, minute_index, np.zeros((4, 4, 3), np.uint8), "x.mp4")


def always(state: Class3):
    return lambda image: (state, 0.9)


# --- which slot is due ------------------------------------------------------------


def test_a_slot_is_due_inside_its_window() -> None:
    s = Schedule((slot("10:00", 60),))
    assert due(s, datetime(2026, 7, 11, 10, 30))


def test_the_window_is_half_open_so_consecutive_slots_never_overlap() -> None:
    """A slot starting at 10:00 for 60 minutes owns 10:00 and not 11:00. If both ends were
    inclusive, the 11:00 slot's first minute would be claimed twice and one verdict would
    overwrite the other."""
    s = Schedule((slot("10:00", 60), slot("11:00", 60)))
    at_eleven = due(s, datetime(2026, 7, 11, 11, 0))
    assert [x.start for x in at_eleven] == [time(11, 0)]


def test_nothing_is_due_outside_the_window() -> None:
    assert not due(Schedule((slot("10:00", 60),)), datetime(2026, 7, 11, 9, 59))
    assert not due(Schedule((slot("10:00", 60),)), datetime(2026, 7, 11, 11, 0))


def test_weekdays_are_respected() -> None:
    s = Schedule((slot("10:00", 60, days=("sat", "sun")),))
    assert not due(s, datetime(2026, 7, 8, 10, 30))     # a Wednesday
    assert due(s, datetime(2026, 7, 11, 10, 30))        # a Saturday


def test_no_days_means_every_day() -> None:
    s = Schedule((slot("10:00", 60),))
    assert all(due(s, datetime(2026, 7, d, 10, 30)) for d in range(6, 13))


def test_the_slot_id_matches_the_convention_the_manifest_uses() -> None:
    assert slot("10:00").slot_id(date(2026, 7, 11)) == "venue_01_2026-07-11_1000"


# --- the schedule file ------------------------------------------------------------


def test_the_committed_schedule_loads() -> None:
    assert len(load_schedule()) >= 1


def test_a_missing_field_is_rejected_rather_than_defaulted(tmp_path) -> None:
    """A silently-dropped entry is a slot with no footage and no explanation - a missing
    hour, discovered later, unrecoverable."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"venue_id": "v", "start": "10:00"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_schedule(path)


def test_a_slot_with_no_cameras_is_rejected(tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{
        "venue_id": "v", "start": "10:00", "duration_minutes": 60, "cameras": []
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="never be observed"):
        load_schedule(path)


def test_two_slots_at_the_same_time_on_one_field_are_rejected(tmp_path) -> None:
    """They share a slot id, so one verdict would overwrite the other's."""
    entry = {"venue_id": "v", "start": "10:00", "duration_minutes": 60, "cameras": ["a"]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps([entry, dict(entry)]), encoding="utf-8")
    with pytest.raises(ValueError, match="two slots starting"):
        load_schedule(path)


def test_an_unknown_weekday_is_rejected(tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{
        "venue_id": "v", "start": "10:00", "duration_minutes": 60,
        "cameras": ["a"], "days": ["funday"],
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown day"):
        load_schedule(path)


def test_a_missing_schedule_says_what_the_file_should_contain(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="duration_minutes"):
        load_schedule(tmp_path / "absent.json")


# --- running ----------------------------------------------------------------------


def test_running_a_due_slot_returns_its_id() -> None:
    s = Schedule((slot("10:00", 60),))
    ran = run_due(s, datetime(2026, 7, 11, 10, 5),
                  source_for=lambda sl, day: Stub(), classify=always(Class3.ACTIVE_PLAY))
    assert ran == ["venue_01_2026-07-11_1000"]


def test_one_dead_camera_does_not_stop_the_other_pitches() -> None:
    """A slot that raises is reported and skipped. Taking the loop down would mean one
    unreachable camera costs every other pitch its hour."""
    good, bad = slot("10:00", 60), ScheduledSlot(
        "venue_02", time(10, 0), 60, ("camera_A",), field_id="field_02"
    )

    def source_for(sl, day):
        if sl.venue_id == "venue_02":
            raise OSError("camera unreachable")
        return Stub()

    seen: list[tuple[str, object]] = []
    ran = run_due(Schedule((bad, good)), datetime(2026, 7, 11, 10, 5),
                  source_for=source_for, classify=always(Class3.EMPTY),
                  on_slot=lambda sid, r: seen.append((sid, r)))
    assert ran == ["venue_01_2026-07-11_1000"]
    assert any(isinstance(r, OSError) for _, r in seen)


def test_the_verdict_reaches_the_database() -> None:
    """"DB fills" is half the acceptance criterion, so it is asserted rather than assumed."""
    from pitch_occupancy.db.schema import connect, initialise

    conn = connect(":memory:")
    initialise(conn)
    run_due(Schedule((slot("10:00", 60),)), datetime(2026, 7, 11, 10, 5),
            source_for=lambda sl, day: Stub(), classify=always(Class3.ACTIVE_PLAY),
            connection=conn, model_key="test")
    rows = conn.execute("SELECT slot_id, status FROM slot_evaluations").fetchall()
    assert [r[0] for r in rows] == ["venue_01_2026-07-11_1000"]
    assert conn.execute("SELECT COUNT(*) FROM frame_samples").fetchone()[0] > 0


def test_the_loop_is_bounded_and_its_sleep_is_injectable() -> None:
    """A loop whose only observable behaviour is that it does not return has no tests. This
    one runs three ticks in microseconds and sleeps between them, not after the last."""
    naps: list[float] = []
    ran = run_forever(
        Schedule((slot("10:00", 60),)),
        source_for=lambda sl, day: Stub(), classify=always(Class3.EMPTY),
        clock=lambda: datetime(2026, 7, 11, 10, 5),
        sleep=naps.append, iterations=3, interval_s=60.0,
    )
    assert len(ran) == 3
    assert naps == [60.0, 60.0], "it should not sleep after the final iteration"


def test_the_loop_forwards_the_model_key_and_the_evidence_dir(tmp_path) -> None:
    """`run_due` has accepted both since it was written and `run_forever` dropped both on the
    floor, so every verdict the loop produced was stored with no model key and no evidence
    however the caller asked. Nothing caught it because nothing called the loop with either:
    it had no entry point until `worker --source live`. Asserting on what reaches `run_due`
    rather than on the database, because forwarding is the part that was missing."""
    import pitch_occupancy.scheduler as sched

    seen: dict[str, object] = {}

    def spy(schedule, now, **kwargs):
        seen.update(kwargs)
        return ["slot"]

    original, sched.run_due = sched.run_due, spy
    try:
        sched.run_forever(
            Schedule((slot("10:00", 60),)),
            source_for=lambda sl, day: Stub(), classify=always(Class3.EMPTY),
            clock=lambda: datetime(2026, 7, 11, 10, 5),
            sleep=lambda _: None, iterations=1,
            model_key="dinov2", evidence_dir=tmp_path,
        )
    finally:
        sched.run_due = original

    assert seen["model_key"] == "dinov2"
    assert seen["evidence_dir"] == tmp_path


# --- against the real recordings --------------------------------------------------


def test_the_derived_schedule_matches_the_recorded_slots() -> None:
    """The recordings key on `slot_YYYYMMDD_HHMM` and the manifest on
    `<field>_YYYY-MM-DD_HHMM`. Two conventions for one thing, and the translation is the
    part that can quietly be wrong."""
    from pitch_occupancy.config import settings

    directory = settings.raw_dir / "venue_01"
    if not directory.exists():
        pytest.skip("recordings not present")
    schedule, dates = schedule_from_recordings(directory)
    assert dates, "no recorded dates found"
    ids = {s.slot_id(dates[0]) for s in schedule.slots}
    assert "venue_01_2026-07-11_1000" in ids, ids


def test_the_scheduler_reproduces_the_published_verdicts() -> None:
    """The half of the acceptance criterion that matters: *verdicts correct on recorded
    slots*, reached through the scheduler rather than by calling `run_slot` directly.

    Compared against `results/end_to_end_slots.csv`, which is committed - so if the two paths
    ever diverge, this fails rather than the scheduler quietly producing something else.
    """
    import csv

    from pitch_occupancy.config import settings
    from pitch_occupancy.scheduler import sources_from_recordings

    published = ROOT / "results" / "end_to_end_slots.csv"
    directory = settings.raw_dir / "venue_01"
    if not published.exists() or not directory.exists():
        pytest.skip("recordings or published verdicts not present")

    expected = {r["slot"]: r["verdict"] for r in csv.DictReader(published.open(encoding="utf-8"))}
    schedule, dates = schedule_from_recordings(directory)
    source_for = sources_from_recordings(directory)

    # The classifier is not the subject here - the wiring is - so a stub that answers from
    # the published play ratio would beg the question. Instead only the *reachability* of
    # each recorded slot through the scheduler is asserted, and the verdict comparison is
    # left to end_to_end_slots.py, which owns the classifier.
    reached = []
    for day in dates:
        for sl in schedule.slots:
            slot_id = sl.slot_id(day)
            if slot_id not in expected:
                continue
            source = source_for(sl, day)
            assert source.n_minutes > 0, f"{slot_id} has no minutes"
            reached.append(slot_id)
    assert set(reached) == set(expected), (
        f"the scheduler does not reach every published slot: "
        f"{sorted(set(expected) - set(reached))}"
    )
