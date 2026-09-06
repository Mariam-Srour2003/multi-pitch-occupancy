"""Train-time augmentation.

The distinction that matters: preprocessing removes information permanently and has a
floor, augmentation varies it at training time and does not. These tests pin the
properties that keep that true - and the ones that keep an audit trail intact."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from pitch_occupancy.vision.augment import (
    AUGMENTATIONS,
    AugmentConfig,
    augment,
    synthetic_fog,
    synthetic_rain,
)


@pytest.fixture
def frame() -> np.ndarray:
    """A pitch-like frame: green turf, white lines, a coloured figure."""
    rng = np.random.default_rng(0)
    img = np.zeros((240, 400, 3), np.uint8)
    img[:, :, 1] = 130
    for x in range(0, 400, 70):
        img[:, x : x + 3] = (235, 235, 235)
    img[120:150, 190:210] = (40, 40, 200)
    return np.clip(img.astype(int) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)


def diff(a, b) -> float:
    return float(np.mean(np.abs(a.astype(float) - b.astype(float))))


# --- the contract -----------------------------------------------------------


def test_default_config_is_a_no_op(frame) -> None:
    """A run has to opt in; augmentation must never happen by accident."""
    assert diff(augment(frame), frame) == 0.0
    assert AugmentConfig().enabled() == []


def test_the_input_is_never_mutated(frame) -> None:
    before = frame.copy()
    augment(frame, AUGMENTATIONS["full"], rng=np.random.default_rng(1))
    assert np.array_equal(frame, before)


def test_output_keeps_shape_and_dtype(frame) -> None:
    for name, cfg in AUGMENTATIONS.items():
        out = augment(frame, cfg, rng=np.random.default_rng(2))
        assert out.shape == frame.shape, name
        assert out.dtype == np.uint8, name


def test_same_seed_gives_the_same_draw(frame) -> None:
    a = augment(frame, AUGMENTATIONS["full"], rng=np.random.default_rng(7))
    b = augment(frame, AUGMENTATIONS["full"], rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_different_seeds_give_different_draws(frame) -> None:
    a = augment(frame, AUGMENTATIONS["full"], rng=np.random.default_rng(1))
    b = augment(frame, AUGMENTATIONS["full"], rng=np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_values_stay_in_range(frame) -> None:
    """Every effect clips; an overflow would wrap and produce impossible colours."""
    for seed in range(6):
        out = augment(frame, AUGMENTATIONS["full"], rng=np.random.default_rng(seed))
        assert out.min() >= 0 and out.max() <= 255


# --- no geometry beyond the mirror ------------------------------------------


def test_no_rotation_or_warp_is_ever_applied(frame) -> None:
    """Cameras are bolted in place. A rotated pitch is not a harder example, it is an
    impossible one, and training on it spends capacity on a case that never arrives."""
    marker = np.zeros((240, 400, 3), np.uint8)
    marker[10:20, 10:20] = 255  # a corner block; a rotation or warp would move it
    no_flip = AugmentConfig(brightness=0.2, contrast=0.2, gamma=0.3, noise=4, flip=0.0, p=1.0)
    out = augment(marker, no_flip, rng=np.random.default_rng(3))
    ys, xs = np.where(out[:, :, 0] > 200)
    assert xs.min() < 25 and ys.min() < 25  # still in the same corner


def test_flip_mirrors_horizontally_only(frame) -> None:
    """The one geometric change, and it is deliberate: it breaks memorisation of this
    pitch's layout without changing whether people are playing."""
    marker = np.zeros((60, 100, 3), np.uint8)
    marker[:, :10] = 255
    out = augment(marker, AugmentConfig(flip=1.0), rng=np.random.default_rng(0))
    assert out[:, -10:].mean() > 200  # moved to the right edge
    assert out[:10, :].shape == marker[:10, :].shape  # unchanged vertically


# --- individual effects -----------------------------------------------------


def test_colour_preset_changes_colour_without_destroying_it(frame) -> None:
    """The point of jitter over grayscale: the channels still differ afterwards.

    p is forced to 1 so this measures the effect, not the coin flip that gates it."""
    cfg = replace(AUGMENTATIONS["colour"], p=1.0)
    out = augment(frame, cfg, rng=np.random.default_rng(4))
    assert diff(out, frame) > 1.0
    assert float(np.mean(np.abs(out[:, :, 1].astype(float) - out[:, :, 2].astype(float)))) > 5


def test_fog_is_heavier_at_the_top(frame) -> None:
    """Distance grows up the frame in a fixed pitch camera, so fog should too."""
    out = synthetic_fog(frame, 0.8, np.random.default_rng(0))
    top = diff(out[:60], frame[:60])
    bottom = diff(out[-60:], frame[-60:])
    assert top > bottom


def test_rain_adds_streaks_and_lowers_contrast(frame) -> None:
    out = synthetic_rain(frame, 0.8, np.random.default_rng(0))
    assert diff(out, frame) > 1.0
    assert out.std() < frame.std() * 1.35  # wet air flattens rather than sharpens


def test_rain_at_zero_strength_is_a_no_op(frame) -> None:
    assert diff(synthetic_rain(frame, 0.0, np.random.default_rng(0)), frame) == 0.0


def test_noise_raises_local_variation(frame) -> None:
    out = augment(frame, AugmentConfig(noise=12, p=1.0), rng=np.random.default_rng(0))
    assert cv2.Laplacian(out, cv2.CV_64F).var() > cv2.Laplacian(frame, cv2.CV_64F).var()


# --- presets ----------------------------------------------------------------


def test_every_preset_enables_something_except_none() -> None:
    for name, cfg in AUGMENTATIONS.items():
        assert (cfg.enabled() == []) == (name == "none"), name


def test_the_colour_preset_targets_the_shortcut_the_ablation_found() -> None:
    """Grayscale helped because turf hue encodes venue identity. Jitter targets the same
    signal without discarding it, which is the whole hypothesis."""
    cfg = AUGMENTATIONS["colour"]
    assert cfg.saturation > 0 and cfg.hue > 0
    assert cfg.rain == 0 and cfg.fog == 0  # colour only, so the effect is attributable


def test_p_gates_how_often_an_effect_fires(frame) -> None:
    """Each enabled effect is applied with probability p, so a single draw may change
    nothing - which is why effect tests force p to 1."""
    never = AugmentConfig(brightness=0.3, p=0.0)
    always = AugmentConfig(brightness=0.3, p=1.0)
    assert diff(augment(frame, never, rng=np.random.default_rng(0)), frame) == 0.0
    changed = sum(
        diff(augment(frame, always, rng=np.random.default_rng(s)), frame) > 0.5
        for s in range(8)
    )
    assert changed == 8


def test_rain_looks_the_same_at_any_resolution() -> None:
    """Augmentation runs *before* preprocessing's resize, so any effect measured in raw
    pixels silently changes meaning with frame size - streaks tuned at 224 become poles on
    a 1080p source. Rain is the only effect with a length scale, so it is the only one that
    can get this wrong; this pins it."""
    small = np.full((180, 320, 3), 90, np.uint8)
    large = np.full((1080, 1920, 3), 90, np.uint8)
    a = diff(synthetic_rain(small, 0.6, np.random.default_rng(0)), small)
    b = diff(synthetic_rain(large, 0.6, np.random.default_rng(0)), large)
    assert abs(a - b) < 0.35 * max(a, b), f"{a:.2f} vs {b:.2f} - rain scales with resolution"


def test_rain_streaks_are_a_small_fraction_of_frame_height() -> None:
    """A streak spanning a tenth of the frame is a white pole, not a raindrop."""
    blank = np.zeros((400, 400, 3), np.uint8)
    out = synthetic_rain(blank, 1.0, np.random.default_rng(1))
    # rain also flattens contrast, which lifts a black frame off zero; a streak is bright
    # relative to that new floor, not to zero
    floor = int(out[:, :, 0].min())
    lit = out[:, :, 0] > floor + 40
    assert lit.any(), "no streaks drawn at full strength"

    # the longest *contiguous* run down a column is one streak's extent; a plain sum would
    # measure how many streaks stacked in that column, which is density, not length
    def longest_run(column: np.ndarray) -> int:
        best = run = 0
        for on in column:
            run = run + 1 if on else 0
            best = max(best, run)
        return best

    # 0.25 is not arbitrary: the absolute-pixel version this replaced measured 0.29 here,
    # and the fraction-of-height version measures 0.17
    assert max(longest_run(lit[:, x]) for x in range(400)) < 400 * 0.25
    assert lit.mean() < 0.15, "streaks cover so much frame it is a whiteout, not rain"
