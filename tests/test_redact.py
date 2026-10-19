"""Redaction: people pixelated, then the whole frame blurred (`vision/redact.py`)."""

from __future__ import annotations

import numpy as np

from pitch_occupancy.vision.redact import pixelate_boxes, redact_frame


def test_pixelation_changes_the_pixels_it_covers_and_nothing_else() -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    out = pixelate_boxes(image, [(0, 0, 30, 30)], blocks=3)
    assert not np.array_equal(out[:30, :30], image[:30, :30])
    assert np.array_equal(out[30:, 30:], image[30:, 30:])


def test_pixelation_actually_destroys_detail() -> None:
    """A pixelation that merely resamples would leave a face recognisable. Variance inside the
    covered region must collapse."""
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    out = pixelate_boxes(image, [(0, 0, 60, 60)], blocks=2)
    assert out.std() < image.std() / 2


def test_a_frame_with_no_boxes_still_gets_the_floor_of_blur() -> None:
    rng = np.random.default_rng(6)
    image = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    out = redact_frame(image, [])
    assert out.shape == image.shape and out.std() < image.std()
