"""Half-drawn frames of every kind seen from the cameras are recognised; real ones are not."""

from __future__ import annotations

import numpy as np

from pitch_occupancy.frame_source import is_incomplete


def _scene(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(40, 200, (480, 640, 3), dtype=np.uint8)


def test_a_magenta_bottom_band_is_half_drawn() -> None:
    frame = _scene()
    frame[200:] = (255, 80, 255)  # BGR magenta
    assert is_incomplete(frame)


def test_a_smeared_bottom_half_is_half_drawn() -> None:
    frame = _scene()
    frame[240:] = frame[239]  # the last good row copied down
    assert is_incomplete(frame)


def test_a_real_scene_and_a_dark_night_are_whole() -> None:
    assert not is_incomplete(_scene(1))
    night = _scene(2)
    night[300:] = 6  # a black, featureless foreground at night
    assert not is_incomplete(night)
