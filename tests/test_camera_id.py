"""Identifying a camera by what it sees.

The bug this guards against is silent: keying a camera off the `(1)` filename suffix is
correct on one recording day and wrong on the other, and swapped halves still fuse into a
plausible verdict. Nothing downstream would raise."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pitch_occupancy.frame_source import discover_slots
from pitch_occupancy.vision.camera_id import describe_view, match_cameras, similarity


def scene(*, bars: int, bright: int, seed: int = 0) -> np.ndarray:
    """A synthetic 'view': fixed vertical structure at a given illumination."""
    rng = np.random.default_rng(seed)
    img = np.full((90, 160, 3), bright, np.uint8)
    for i in range(bars):
        x = int((i + 1) * 160 / (bars + 1))
        img[:, x - 2 : x + 2] = min(255, bright + 90)
    img = np.clip(img.astype(int) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img


def make_video(path: Path, frames: list[np.ndarray], fps: int = 5) -> None:
    h, w = frames[0].shape[:2]
    w_ = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        w_.write(f)
    w_.release()


def recording(path: Path, *, bars: int, bright: int, movers: bool = True) -> Path:
    """A view with moving foreground, so the median has something to remove."""
    frames = []
    for t in range(40):
        f = scene(bars=bars, bright=bright, seed=t).copy()
        if movers:  # a "player" that wanders, and must not survive the median
            x = 10 + (t * 3) % 130
            f[40:60, x : x + 12] = 255 - bright
        frames.append(f)
    make_video(path, frames)
    return path


# --- descriptor -------------------------------------------------------------


def test_descriptor_is_normalised(tmp_path: Path) -> None:
    d = describe_view(recording(tmp_path / "a.mp4", bars=3, bright=120))
    assert np.linalg.norm(d.vector) == pytest.approx(1.0, abs=1e-5)


def test_same_view_under_different_lighting_matches(tmp_path: Path) -> None:
    """The property the whole approach rests on: day and night are the same camera."""
    day = describe_view(recording(tmp_path / "day.mp4", bars=3, bright=200))
    night = describe_view(recording(tmp_path / "night.mp4", bars=3, bright=45))
    assert similarity(day, night) > 0.9


def test_different_views_score_lower_than_the_same_view(tmp_path: Path) -> None:
    day_a = describe_view(recording(tmp_path / "da.mp4", bars=3, bright=200))
    night_a = describe_view(recording(tmp_path / "na.mp4", bars=3, bright=45))
    night_b = describe_view(recording(tmp_path / "nb.mp4", bars=8, bright=45))
    assert similarity(day_a, night_a) > similarity(day_a, night_b)


def test_moving_foreground_does_not_dominate(tmp_path: Path) -> None:
    """The median must leave the fixed background, not the players."""
    with_movers = describe_view(recording(tmp_path / "m.mp4", bars=3, bright=120))
    without = describe_view(recording(tmp_path / "s.mp4", bars=3, bright=120, movers=False))
    assert similarity(with_movers, without) > 0.9


def test_unreadable_video_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(OSError):
        describe_view(bad)


# --- matching ---------------------------------------------------------------


def test_matches_recordings_to_reference_cameras(tmp_path: Path) -> None:
    refs = {
        "camA": describe_view(recording(tmp_path / "ra.mp4", bars=3, bright=200)),
        "camB": describe_view(recording(tmp_path / "rb.mp4", bars=8, bright=200)),
    }
    # the same two views at night, deliberately in the opposite file order
    new = {
        "file0": describe_view(recording(tmp_path / "n0.mp4", bars=8, bright=45)),
        "file1": describe_view(recording(tmp_path / "n1.mp4", bars=3, bright=45)),
    }
    assert match_cameras(new, refs) == {"file0": "camB", "file1": "camA"}


def test_each_reference_is_used_once(tmp_path: Path) -> None:
    """A per-recording argmax could hand both halves of a pitch to one camera."""
    refs = {
        "camA": describe_view(recording(tmp_path / "ra.mp4", bars=3, bright=200)),
        "camB": describe_view(recording(tmp_path / "rb.mp4", bars=4, bright=200)),
    }
    new = {
        "file0": describe_view(recording(tmp_path / "n0.mp4", bars=3, bright=60)),
        "file1": describe_view(recording(tmp_path / "n1.mp4", bars=3, bright=70)),
    }
    assigned = match_cameras(new, refs)
    assert set(assigned.values()) == {"camA", "camB"}


def test_a_poor_match_is_left_unassigned(tmp_path: Path) -> None:
    """Better an obvious gap than a silently swapped pitch half."""
    refs = {"camA": describe_view(recording(tmp_path / "ra.mp4", bars=3, bright=200))}
    new = {"file0": describe_view(recording(tmp_path / "n0.mp4", bars=20, bright=45))}
    assert match_cameras(new, refs, min_similarity=0.999) == {"file0": None}


# --- the naming fix ---------------------------------------------------------


def test_discover_slots_does_not_claim_camera_identity(tmp_path: Path) -> None:
    """The `(1)` suffix flips between days, so it must not be reported as camA/camB."""
    for name in (
        "StatBox_Replay_v_2026-07-11_10-00.mp4",
        "StatBox_Replay_v_2026-07-11_10-00 (1).mp4",
    ):
        (tmp_path / name).write_bytes(b"")
    keys = set(discover_slots(tmp_path)["slot_20260711_1000"])
    assert keys == {"file0", "file1"}
    assert not keys & {"camA", "camB"}
