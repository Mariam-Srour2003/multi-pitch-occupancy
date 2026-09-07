"""Frame quality and camera health.

The acceptance criterion for WP3-T5 is "known-bad frames flagged", and this dataset does
not contain any: 1,692 frames, none clipped beyond 1.5%, and every frame the filter flagged
turned out to be readable when rendered. So the filter is proved against **deliberately
degraded** frames, and the fact that the real data has no positives is recorded rather than
papered over by tuning until something is flagged.

Most of these tests exist because an earlier version was wrong in a way that looked right.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitch_occupancy.vision.quality import (
    CLIP_FRACTION, MATERIAL_BLUR_DROP, assess, build_baseline, measure,
)


def pitch(seed: int = 0, brightness: int = 130) -> np.ndarray:
    """A frame with real structure: turf, lines, and a figure worth keeping visible."""
    rng = np.random.default_rng(seed)
    img = np.zeros((360, 640, 3), np.uint8)
    img[:, :, 1] = brightness
    for x in range(0, 640, 80):
        img[:, x : x + 3] = (235, 235, 235)
    img[180:230, 300:330] = (40, 40, 200)
    return np.clip(img.astype(int) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def baseline_of(frames: list[np.ndarray], name: str = "cam"):
    return build_baseline(name, [measure(f) for f in frames])


@pytest.fixture
def healthy() -> list[np.ndarray]:
    return [pitch(s) for s in range(40)]


# --- measurement ------------------------------------------------------------


def test_blur_is_measured_at_a_fixed_width_not_a_fixed_factor() -> None:
    """Laplacian variance scales with resolution, so measuring cameras at their native sizes
    would compare a 1080p frame against a 720p one and call the difference sharpness."""
    small = pitch(0)
    large = cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    a, b = measure(small), measure(large)
    assert abs(a.blur - b.blur) < 0.35 * max(a.blur, b.blur)


def test_a_blurred_frame_measures_less_detail() -> None:
    assert measure(cv2.GaussianBlur(pitch(0), (0, 0), 4)).blur < measure(pitch(0)).blur


def test_a_flat_frame_measures_less_contrast() -> None:
    flat = np.full((360, 640, 3), 120, np.uint8)
    assert measure(flat).rms_contrast < measure(pitch(0)).rms_contrast


# --- the absolute check -----------------------------------------------------


def test_a_blown_out_frame_is_bad_with_no_baseline_at_all() -> None:
    """Exposure failure is absolute: a pixel at 255 has lost its detail whichever camera
    produced it, so this is the one judgement that needs no history."""
    verdict = assess(measure(np.full((360, 640, 3), 253, np.uint8)))
    assert verdict.quality == "bad"
    assert "blown out" in verdict.reasons[0]


def test_a_crushed_frame_is_bad_with_no_baseline_at_all() -> None:
    verdict = assess(measure(np.zeros((360, 640, 3), np.uint8)))
    assert verdict.quality == "bad"
    assert "crushed black" in verdict.reasons[0]


def test_a_normal_night_frame_is_not_an_exposure_failure() -> None:
    """Night footage is legitimately dark. The clipping threshold sits far above anything
    this dataset produces - its worst frame clips 1.5%, against a limit of 10%."""
    assert measure(pitch(0, brightness=40)).exposure_failure() is None
    assert CLIP_FRACTION > 0.05


# --- the relative checks, and the mistakes they encode -----------------------


def test_a_genuinely_degraded_frame_is_flagged(healthy) -> None:
    """The acceptance criterion, on a frame degraded on purpose since the dataset has none."""
    verdict = assess(measure(cv2.GaussianBlur(pitch(99), (0, 0), 6)), baseline_of(healthy))
    assert verdict.quality == "bad"
    # heavy blur costs contrast as well as detail, so both reasons fire; the point is that
    # the operator is told the lens may be at fault, not which reason came first
    assert any("lens may be dirty" in r for r in verdict.reasons), verdict.reasons


def test_a_normal_frame_from_the_same_camera_is_not_flagged(healthy) -> None:
    assert assess(measure(pitch(99)), baseline_of(healthy)).quality == "good"


def test_being_merely_unusual_is_not_enough_to_be_called_bad(healthy) -> None:
    """The mistake this replaced. venue_01's camera is so consistent that its MAD is tiny,
    so a 3-MAD rule alone put the floor within 9% of the median and flagged four frames that
    were indistinguishable from normal when rendered side by side. Statistically unusual is
    not the same as unusable, so a material drop is required as well."""
    baseline = baseline_of(healthy)
    slightly_soft = cv2.GaussianBlur(pitch(99), (0, 0), 0.4)
    q = measure(slightly_soft)
    assert q.blur < baseline.blur_floor, "should be a statistical outlier for this test to mean anything"
    assert q.blur > baseline.blur_median * MATERIAL_BLUR_DROP
    assert assess(q, baseline).quality == "good"


def test_a_soft_camera_is_not_bad_just_for_being_soft() -> None:
    """The whole design in one test. A global cutoff between these two cameras would flag
    every frame from the softer one - which is how a quality filter ends up measuring venue
    identity, the confound this project already found the hard way."""
    soft = [cv2.GaussianBlur(pitch(s), (0, 0), 3) for s in range(40)]
    sharp = [pitch(s) for s in range(40)]
    assert measure(soft[0]).blur < measure(sharp[0]).blur / 2  # a global rule would split them
    for frame in soft[:5]:
        assert assess(measure(frame), baseline_of(soft)).quality == "good"


def test_a_night_frame_is_not_degraded_for_being_darker_than_daylight() -> None:
    """Baselines are keyed on lighting for this reason. Judged against a daylight median, six
    perfectly readable floodlit frames were flagged for low contrast."""
    night = [pitch(s, brightness=45) for s in range(40)]
    for frame in night[:5]:
        assert assess(measure(frame), baseline_of(night)).quality == "good"


# --- refusing to guess ------------------------------------------------------


def test_a_thin_baseline_gives_unknown_rather_than_good() -> None:
    """A baseline from six frames describes those six frames. `unknown` is a different claim
    from `good`, and the honest one."""
    verdict = assess(measure(pitch(0)), baseline_of([pitch(s) for s in range(6)]))
    assert verdict.quality == "unknown"
    assert verdict.usable  # not having measured a camera is not evidence against it


def test_no_baseline_at_all_gives_unknown() -> None:
    assert assess(measure(pitch(0))).quality == "unknown"


def test_exposure_failure_beats_a_missing_baseline() -> None:
    """Absolute damage is still reportable when nothing is known about the camera."""
    assert assess(measure(np.full((360, 640, 3), 254, np.uint8))).quality == "bad"


def test_the_baseline_is_not_dragged_down_by_a_few_bad_frames() -> None:
    """Median and MAD, not mean and standard deviation: a camera with a dirty lens for a week
    would otherwise move its own definition of normal until it stopped reporting itself."""
    mostly_fine = [pitch(s) for s in range(36)]
    ruined = [cv2.GaussianBlur(pitch(s), (0, 0), 8) for s in range(90, 94)]
    contaminated = baseline_of(mostly_fine + ruined)
    clean = baseline_of(mostly_fine)
    assert abs(contaminated.blur_median - clean.blur_median) < 0.1 * clean.blur_median
    assert assess(measure(ruined[0]), contaminated).quality == "bad"
