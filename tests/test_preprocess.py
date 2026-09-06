"""Preprocessing is the one code path shared by experiments and the live pipeline.
When the two drift apart the symptom is a model that scores well and fails in
production, which is precisely what happened in the pilot."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitch_occupancy.vision.preprocess import (
    PreprocessConfig,
    apply_clahe,
    letterbox,
    preprocess,
    rms_contrast,
    roi_mask,
)


@pytest.fixture
def wide() -> np.ndarray:
    """A wide frame with a bright square, so distortion and cropping are visible."""
    img = np.full((180, 320, 3), 60, np.uint8)
    img[70:110, 140:180] = 240
    return img


# --- letterbox --------------------------------------------------------------


def test_letterbox_is_square_and_preserves_aspect(wide) -> None:
    out = letterbox(wide, 224)
    assert out.shape == (224, 224, 3)
    # a 320x180 frame scales to 224x126, leaving grey bands top and bottom
    assert (out[0] == 114).all()
    assert (out[-1] == 114).all()


def test_letterbox_does_not_squash(wide) -> None:
    """The bright square must stay square, which a naive resize would not preserve."""
    out = letterbox(wide, 224)
    ys, xs = np.where(out[:, :, 0] > 200)
    assert abs((ys.max() - ys.min()) - (xs.max() - xs.min())) <= 2


def test_naive_resize_does_squash(wide) -> None:
    """Contrast case: this is what letterbox exists to avoid.

    Forcing 320x180 into a square compresses the width (320 -> 224) and stretches the
    height (180 -> 224), so the square renders noticeably taller than it is wide.
    """
    out = preprocess(wide, PreprocessConfig(letterbox=False))
    ys, xs = np.where(out[:, :, 0] > 200)
    assert (ys.max() - ys.min()) > (xs.max() - xs.min()) + 5


def test_letterbox_handles_a_square_input() -> None:
    assert letterbox(np.zeros((100, 100, 3), np.uint8), 64).shape == (64, 64, 3)


# --- roi --------------------------------------------------------------------


def test_roi_blacks_out_everything_outside_the_polygon() -> None:
    img = np.full((100, 100, 3), 255, np.uint8)
    out = roi_mask(img, [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    assert out[50, 50].tolist() == [255, 255, 255]  # inside
    assert out[5, 5].tolist() == [0, 0, 0]  # outside


def test_roi_without_a_polygon_is_a_no_op() -> None:
    img = np.full((20, 20, 3), 7, np.uint8)
    assert np.array_equal(roi_mask(img, None), img)


# --- photometric ------------------------------------------------------------


def test_clahe_raises_contrast_on_a_flat_frame() -> None:
    rng = np.random.default_rng(0)
    dim = (rng.normal(40, 6, (120, 120, 3))).clip(0, 255).astype(np.uint8)
    assert rms_contrast(apply_clahe(dim)) > rms_contrast(dim)


def test_auto_clahe_engages_only_below_the_threshold() -> None:
    rng = np.random.default_rng(1)
    flat = (rng.normal(40, 5, (120, 120, 3))).clip(0, 255).astype(np.uint8)
    varied = (rng.normal(128, 70, (120, 120, 3))).clip(0, 255).astype(np.uint8)
    cfg = PreprocessConfig(clahe="auto", letterbox=False)
    off = PreprocessConfig(clahe="off", letterbox=False)
    assert not np.array_equal(preprocess(flat, cfg), preprocess(flat, off))
    assert np.array_equal(preprocess(varied, cfg), preprocess(varied, off))


# --- diagnostics ------------------------------------------------------------


def test_grayscale_removes_colour_but_keeps_three_channels() -> None:
    """`grayscale` became `saturation=0.0` - a dial, because full desaturation helped
    alone but broke when combined with cropping."""
    img = np.zeros((60, 60, 3), np.uint8)
    img[:, :, 2] = 200  # pure red
    out = preprocess(img, PreprocessConfig(saturation=0.0, letterbox=False))
    assert out.shape[2] == 3
    assert out[:, :, 0].std() == 0
    assert np.allclose(out[:, :, 0], out[:, :, 2])  # channels identical == no colour left


def test_blur_destroys_fine_detail(wide) -> None:
    sharp = preprocess(wide, PreprocessConfig(letterbox=False))
    blurred = preprocess(wide, PreprocessConfig(blur_sigma=8.0, letterbox=False))
    assert cv2.Laplacian(blurred, cv2.CV_64F).var() < cv2.Laplacian(sharp, cv2.CV_64F).var()


def test_centre_crop_discards_the_border() -> None:
    img = np.zeros((100, 100, 3), np.uint8)
    img[0:10, :] = 255  # a bright band only at the top edge
    out = preprocess(img, PreprocessConfig(centre_crop=0.5, letterbox=False))
    assert out.max() == 0


def test_centre_crop_of_one_is_a_no_op(wide) -> None:
    assert np.array_equal(
        preprocess(wide, PreprocessConfig(centre_crop=1.0)),
        preprocess(wide, PreprocessConfig()),
    )


# --- contract ---------------------------------------------------------------


def test_output_is_always_the_requested_size(wide) -> None:
    for cfg in (
        PreprocessConfig(),
        PreprocessConfig(letterbox=False),
        PreprocessConfig(size=384),
        PreprocessConfig(saturation=0.0, blur_sigma=3.0, centre_crop=0.7),
    ):
        assert preprocess(wide, cfg).shape == (cfg.size, cfg.size, 3)


def test_preprocess_is_deterministic(wide) -> None:
    cfg = PreprocessConfig(clahe="on", blur_sigma=2.0)
    assert np.array_equal(preprocess(wide, cfg), preprocess(wide, cfg))


def test_config_dict_captures_every_switch_that_changes_output() -> None:
    """The cache is keyed on this dict; a switch missing from it would let two
    different preprocessings share one cache entry."""
    keys = set(PreprocessConfig().as_dict())
    assert {"roi", "letterbox", "clahe", "saturation", "blur_sigma", "centre_crop", "size"} <= keys
