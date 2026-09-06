"""Guards on the searchable preprocessing space.

The failure this exists to prevent is specific and was hit for real: `roi=True` with no
polygon returns the frame untouched, so the search evaluated a no-op and would have
recorded "ROI masking does not help" - a false conclusion drawn from a switch that never
ran. A switch that cannot change the image must never enter a search."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitch_occupancy.vision.preprocess import (
    SWITCHES,
    PreprocessConfig,
    preprocess,
    roi_mask,
    undistort_fisheye,
)


@pytest.fixture
def frame() -> np.ndarray:
    """Structured, coloured, and off-centre, so geometric and colour switches both bite."""
    rng = np.random.default_rng(0)
    img = np.zeros((360, 640, 3), np.uint8)
    img[:, :, 1] = 120  # green turf
    img[:120, :] = (90, 70, 60)  # a distinct band at the top: sky/stands
    for x in range(0, 640, 80):
        img[:, x : x + 3] = (240, 240, 240)  # pitch lines
    img[200:240, 300:330] = (30, 30, 200)  # a "player"
    return np.clip(img.astype(int) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)


def diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(float) - b.astype(float))))


# --- the guard that matters -------------------------------------------------


@pytest.mark.parametrize(
    "switch,value",
    [(s, v) for s, values in SWITCHES.items() for v in values],
    ids=[f"{s}={v}" for s, values in SWITCHES.items() for v in values],
)
def test_no_searchable_switch_is_a_no_op(frame, switch, value) -> None:
    """Every setting in the search space must visibly change the frame.

    A no-op scores identically to the baseline and is then reported as 'this does not
    help', which is a claim about the technique rather than about the code path."""
    base = preprocess(frame, PreprocessConfig())
    got = preprocess(frame, PreprocessConfig(**{switch: value}))
    assert diff(base, got) > 0.5, f"{switch}={value} left the frame effectively unchanged"


def test_roi_is_excluded_from_the_search_space() -> None:
    """It masks with a polygon; none are drawn yet (WP3-T1), so it would search a no-op."""
    assert "roi" not in SWITCHES


def test_roi_without_a_polygon_is_a_no_op_by_design(frame) -> None:
    """Documenting the behaviour that made excluding it necessary."""
    assert diff(roi_mask(frame, None), frame) == 0.0


def test_roi_with_a_polygon_does_mask(frame) -> None:
    masked = roi_mask(frame, [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])
    assert masked[5, 5].tolist() == [0, 0, 0]
    assert diff(masked, frame) > 1.0


# --- individual switch behaviour -------------------------------------------


def test_every_switch_still_yields_the_model_input_size(frame) -> None:
    for switch, values in SWITCHES.items():
        for value in values:
            out = preprocess(frame, PreprocessConfig(**{switch: value}))
            assert out.shape == (224, 224, 3), f"{switch}={value}"


def test_top_crop_removes_the_upper_band_not_the_pitch(frame) -> None:
    """Sky and stands sit above the horizon; the pitch does not."""
    out = preprocess(frame, PreprocessConfig(top_crop=0.35, letterbox=False))
    base = preprocess(frame, PreprocessConfig(letterbox=False))
    assert diff(out, base) > 1.0


def test_saturation_zero_is_grayscale(frame) -> None:
    out = preprocess(frame, PreprocessConfig(saturation=0.0, letterbox=False))
    assert np.allclose(out[:, :, 0], out[:, :, 2], atol=1)


def test_partial_saturation_sits_between_colour_and_grey(frame) -> None:
    """The dial exists because full desaturation broke when combined with cropping."""
    full = preprocess(frame, PreprocessConfig(letterbox=False))
    half = preprocess(frame, PreprocessConfig(saturation=0.5, letterbox=False))
    grey = preprocess(frame, PreprocessConfig(saturation=0.0, letterbox=False))
    assert diff(full, half) < diff(full, grey)


def test_gamma_below_one_brightens(frame) -> None:
    darker = preprocess(frame, PreprocessConfig(gamma=1.4, letterbox=False))
    brighter = preprocess(frame, PreprocessConfig(gamma=0.7, letterbox=False))
    assert brighter.mean() > darker.mean()


def test_standardisation_removes_a_brightness_offset(frame) -> None:
    """Two exposures of one scene should converge - that is the day/night attack."""
    dim = np.clip(frame.astype(int) - 55, 0, 255).astype(np.uint8)
    cfg = PreprocessConfig(per_image_standardise=True, letterbox=False)
    off = PreprocessConfig(letterbox=False)
    assert diff(preprocess(frame, cfg), preprocess(dim, cfg)) < diff(
        preprocess(frame, off), preprocess(dim, off)
    )


def test_undistort_moves_pixels_and_zero_is_identity(frame) -> None:
    assert diff(undistort_fisheye(frame, 0.0), frame) == 0.0
    assert diff(undistort_fisheye(frame, 0.3), frame) > 1.0


def test_sharpen_raises_edge_energy(frame) -> None:
    base = preprocess(frame, PreprocessConfig())
    sharp = preprocess(frame, PreprocessConfig(sharpen=0.6))
    assert cv2.Laplacian(sharp, cv2.CV_64F).var() > cv2.Laplacian(base, cv2.CV_64F).var()


def test_denoise_lowers_noise_without_flattening_the_frame(frame) -> None:
    base = preprocess(frame, PreprocessConfig())
    smooth = preprocess(frame, PreprocessConfig(denoise=25.0))
    assert cv2.Laplacian(smooth, cv2.CV_64F).var() < cv2.Laplacian(base, cv2.CV_64F).var()
    assert smooth.std() > base.std() * 0.5  # still an image, not a grey field


# --- config identity --------------------------------------------------------


def test_describe_lists_only_what_differs_from_the_default() -> None:
    assert PreprocessConfig().describe() == "baseline"
    d = PreprocessConfig(saturation=0.5, top_crop=0.2).describe()
    assert "saturation=0.5" in d and "top_crop=0.2" in d
    assert "gamma" not in d


def test_config_dict_covers_every_searchable_switch() -> None:
    """The feature cache is keyed on this dict; a missing switch would let two different
    preprocessings collide on one cache entry."""
    keys = set(PreprocessConfig().as_dict())
    assert set(SWITCHES) <= keys
