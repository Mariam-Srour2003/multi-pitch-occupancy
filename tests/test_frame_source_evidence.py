"""Frame sources and evidence selection.

Evidence is the entire basis on which a disputed slot gets settled, and a frame source
that invents a frame during an outage would corrupt a verdict silently."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import VideoSlotSource, discover_slots
from pitch_occupancy.slots.evidence import select_evidence

PLAY, EMPTY = Class3.ACTIVE_PLAY, Class3.EMPTY


def sample(minute: int, cls: Class3, conf: float):
    return (minute, cls, conf, f"f{minute}.jpg")


# --- evidence selection -----------------------------------------------------


def test_one_frame_per_third_spread_across_the_slot() -> None:
    rows = [sample(m, PLAY, 0.9) for m in range(60)]
    ev = select_evidence(rows, SlotStatus.USED)
    assert len(ev) == 3
    assert {e.third for e in ev} == {0, 1, 2}
    assert [e.minute_index for e in ev] == sorted(e.minute_index for e in ev)


def test_thirds_beat_top_three_by_confidence() -> None:
    """Three confident frames from one passage of play prove one minute, not a slot."""
    rows = [sample(m, PLAY, 0.5) for m in range(60)]
    rows[20:23] = [sample(m, PLAY, 0.99) for m in (20, 21, 22)]  # one hot passage
    ev = select_evidence(rows, SlotStatus.USED)
    assert len({e.third for e in ev}) == 3
    assert not all(20 <= e.minute_index <= 22 for e in ev)


def test_most_confident_frame_within_each_third_is_chosen() -> None:
    rows = [sample(m, PLAY, 0.5) for m in range(30)]
    rows[5] = sample(5, PLAY, 0.97)
    ev = select_evidence(rows, SlotStatus.USED)
    assert ev[0].minute_index == 5


def test_evidence_shows_the_class_the_verdict_claims() -> None:
    rows = [sample(m, PLAY if m % 2 else EMPTY, 0.8) for m in range(60)]
    assert all(e.predicted is PLAY for e in select_evidence(rows, SlotStatus.USED))
    assert all(e.predicted is EMPTY for e in select_evidence(rows, SlotStatus.NOTUSED))


def test_backfill_when_a_third_lacks_the_target_class() -> None:
    """A late kick-off leaves an empty first third of a USED slot - show it anyway."""
    rows = [sample(m, EMPTY, 0.9) for m in range(20)] + [
        sample(m, PLAY, 0.9) for m in range(20, 60)
    ]
    ev = select_evidence(rows, SlotStatus.USED)
    assert len(ev) == 3
    first = next(e for e in ev if e.third == 0)
    assert first.predicted is EMPTY
    assert first.is_backfill


def test_review_verdicts_claim_nothing_so_nothing_is_backfilled() -> None:
    rows = [sample(m, EMPTY if m < 30 else PLAY, 0.7) for m in range(60)]
    ev = select_evidence(rows, SlotStatus.REVIEW)
    assert len(ev) == 3
    assert not any(e.is_backfill for e in ev)


def test_a_third_with_no_samples_is_skipped_not_fabricated() -> None:
    """A short timeline is honest; a filled-in one is not."""
    rows = [sample(m, PLAY, 0.9) for m in range(0, 10)]
    ev = select_evidence(rows + [sample(50, PLAY, 0.9)], SlotStatus.USED)
    assert len(ev) < 3


def test_no_samples_yields_no_evidence() -> None:
    assert select_evidence([], SlotStatus.REVIEW) == []


def test_final_minute_is_never_dropped() -> None:
    rows = [sample(m, PLAY, 0.5) for m in range(59)] + [sample(59, PLAY, 0.99)]
    ev = select_evidence(rows, SlotStatus.USED)
    assert ev[-1].minute_index == 59


def test_accepts_raw_class_strings() -> None:
    ev = select_evidence([(0, "C2_ACTIVE_PLAY", 0.9, None)], SlotStatus.USED)
    assert ev[0].predicted is PLAY


# --- video source -----------------------------------------------------------


def make_video(path: Path, *, seconds: int, fps: int = 5, value: int = 100) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(seconds * fps):
        w.write(np.full((48, 64, 3), (value + i) % 256, np.uint8))
    w.release()


@pytest.fixture
def slot_videos(tmp_path: Path) -> dict[str, Path]:
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    make_video(a, seconds=12, value=50)
    make_video(b, seconds=12, value=150)
    return {"camA": a, "camB": b}


def test_video_source_exposes_its_cameras(slot_videos) -> None:
    src = VideoSlotSource(slot_videos, seconds_per_minute=4)
    assert src.cameras() == ["camA", "camB"]
    src.close()


def test_video_source_reads_a_frame_per_simulated_minute(slot_videos) -> None:
    with VideoSlotSource(slot_videos, seconds_per_minute=4) as src:
        assert src.n_minutes == 3
        frame = src.read("camA", 0)
        assert frame is not None
        assert frame.image_bgr.shape == (48, 64, 3)
        assert frame.minute_index == 0


def test_reading_past_the_end_returns_none_rather_than_raising(slot_videos) -> None:
    """A gap in the record, not a fabricated observation."""
    with VideoSlotSource(slot_videos, seconds_per_minute=4) as src:
        assert src.read("camA", 999) is None


def test_unknown_camera_raises(slot_videos) -> None:
    with VideoSlotSource(slot_videos) as src:
        with pytest.raises(KeyError, match="unknown camera"):
            src.read("camZ", 0)


def test_slot_length_is_the_shortest_camera(tmp_path: Path) -> None:
    """Fusion needs both halves of a minute; the shorter recording bounds the slot."""
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    make_video(a, seconds=12)
    make_video(b, seconds=8)
    with VideoSlotSource({"camA": a, "camB": b}, seconds_per_minute=4) as src:
        assert src.n_minutes == 2


def test_a_source_with_no_cameras_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one camera"):
        VideoSlotSource({})


# --- discovery --------------------------------------------------------------


def test_discovers_and_pairs_slot_recordings(tmp_path: Path) -> None:
    for name in (
        "StatBox_Replay_venue_2026-07-11_10-00.mp4",
        "StatBox_Replay_venue_2026-07-11_10-00 (1).mp4",
        "StatBox_Replay_venue_2026-07-12_20-30.mp4",
    ):
        (tmp_path / name).write_bytes(b"")
    slots = discover_slots(tmp_path)
    assert set(slots) == {"slot_20260711_1000", "slot_20260712_2030"}
    # file0/file1, not camA/camB - the `(1)` suffix flips between recording days, so it
    # identifies a file, never a physical camera. See test_camera_id.py.
    assert set(slots["slot_20260711_1000"]) == {"file0", "file1"}
    assert set(slots["slot_20260712_2030"]) == {"file0"}


def test_unrecognised_filenames_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "holiday.mp4").write_bytes(b"")
    assert discover_slots(tmp_path) == {}
