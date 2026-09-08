"""Frame quality and camera health (WP3-T5).

Three jobs: flag frames too degraded to classify, notice when a camera is deteriorating,
and record the numbers so a `REVIEW` verdict can say *why* it hesitated (`slots/conditions.py`
consumes them).

## Every threshold here is relative to the camera's own baseline, and that is not a detail

The obvious design is a constant: Laplacian variance below *x* is blurry, RMS contrast below
*y* is unreadable. Measured across the 1,692 labelled frames, that design fails badly:

| camera group | median RMS contrast | median blur (Laplacian variance) |
|---|---|---|
| `venue_01`, night | 20.9 | 122 |
| `venue_01`, day | 17.3 | 274 |
| `clipvenue_i_outdoor_trees`, day | 47.1 | 2398 |
| `clipvenue_h_teal_pitch`, day | 39.5 | 1989 |

Blur spans **88 to 2755, a thirty-fold range**, and what it tracks is *which camera took the
frame* - resolution, optics, compression, distance to the pitch - not whether that frame is
usable. Any global cutoff placed between those groups flags **every frame from `venue_01`**,
which is 1,296 of 1,692 frames, 77% of the dataset, and the only venue with full-length
recordings. A "quality filter" that discards the primary venue and keeps the clip venues has
not measured quality. It has measured venue identity and called it quality - which is exactly
the confound this project already found the hard way with day-versus-night, arriving a second
time wearing a different hat.

So a camera is compared only against itself. `venue_01` is soft and low-contrast *for
`venue_01`* only when it drops below what `venue_01` normally does. Median and MAD rather
than mean and standard deviation, so a handful of genuinely bad frames cannot drag the
baseline down to meet them.

## Exposure clipping is the one absolute check, and it is honest to keep

Blown highlights and crushed blacks are absolute: a pixel at 255 has lost its information
regardless of which camera produced it. **This dataset contains none** - the worst frame
clips 1.5% of pixels dark and 1.2% bright, far below any sensible threshold - so the check
has no real positives here and is exercised by synthetic frames in the tests. That is worth
stating plainly rather than reporting a filter that has never fired as if it were validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

__all__ = [
    "FrameQuality", "CameraBaseline", "measure", "build_baseline", "assess",
    "CLIP_LOW", "CLIP_HIGH", "CLIP_FRACTION", "ROBUST_K",
]

#: Absolute exposure limits. A pixel this dark or this bright carries no detail whichever
#: camera produced it, so these do not need a per-camera baseline.
CLIP_LOW, CLIP_HIGH = 8, 247

#: Fraction of the frame that must be clipped before it counts. Set well above the worst
#: frame in this dataset (1.5%), so it fires on genuine exposure failure rather than on the
#: dark corners every night frame has.
CLIP_FRACTION = 0.10

#: How many robust deviations below a camera's own median counts as unusual. 3 is the
#: conventional outlier distance; with MAD scaled to sigma it corresponds to roughly the
#: 0.1st percentile of a normal camera's own distribution.
ROBUST_K = 3.0

#: A frame must **also** be this much worse than its camera's median before it is called bad.
#:
#: Statistically unusual is not the same as unusable, and on a consistent camera the two come
#: apart completely. `venue_01` is so stable that its MAD is tiny, so the 3-MAD rule alone put
#: the floor within 9% of the median and flagged four frames at blur 221-235 against a median
#: of 253. Rendered side by side with a typical frame, they are indistinguishable - same
#: pitch, same light, equally readable. The rule had found the bottom of a tight distribution
#: and called it damage.
#:
#: Requiring a *material* drop as well as an unusual one is what separates "this camera is
#: having a slightly worse minute" from "something is on the lens". A frame that has lost
#: half its detail, or a third of its contrast, has genuinely lost something.
MATERIAL_BLUR_DROP = 0.50
MATERIAL_CONTRAST_DROP = 0.65

#: MAD -> standard-deviation scale for a normal distribution.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class FrameQuality:
    """What one frame looks like, before any judgement is applied."""

    mean_intensity: float
    rms_contrast: float
    blur: float
    """Variance of the Laplacian. Higher is sharper. **Only comparable within one camera.**"""
    clipped_low: float
    clipped_high: float

    @property
    def clipped(self) -> float:
        return self.clipped_low + self.clipped_high

    def exposure_failure(self) -> str | None:
        """The one judgement that needs no baseline."""
        if self.clipped_low >= CLIP_FRACTION:
            return f"{self.clipped_low:.0%} of the frame is crushed black"
        if self.clipped_high >= CLIP_FRACTION:
            return f"{self.clipped_high:.0%} of the frame is blown out"
        return None


@dataclass(frozen=True, slots=True)
class CameraBaseline:
    """What normal looks like *for one camera*, built from its own frames.

    Median and MAD, not mean and standard deviation: a camera with a dirty lens for a week
    would otherwise move its own definition of normal far enough to stop reporting itself.
    """

    camera_id: str
    n_frames: int
    contrast_median: float
    contrast_mad: float
    blur_median: float
    blur_mad: float

    def floor(self, median_value: float, mad: float) -> float:
        return median_value - ROBUST_K * (mad * _MAD_TO_SIGMA)

    @property
    def contrast_floor(self) -> float:
        return self.floor(self.contrast_median, self.contrast_mad)

    @property
    def blur_floor(self) -> float:
        return self.floor(self.blur_median, self.blur_mad)

    @property
    def trustworthy(self) -> bool:
        """A baseline from a handful of frames describes those frames, not the camera.

        Below this, `assess` reports the numbers and declines to judge - which is the right
        answer, and a different answer from "this frame is fine".
        """
        return self.n_frames >= 20


@dataclass(frozen=True, slots=True)
class QualityVerdict:
    quality: str  # good | bad | unknown
    reasons: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """`unknown` is usable. Not having measured a camera is not evidence against it."""
        return self.quality != "bad"


def measure(image_bgr: np.ndarray, *, work_width: int = 320) -> FrameQuality:
    """Measure one frame.

    Downscaled first, and to a *fixed* width rather than a fixed factor. Laplacian variance
    depends strongly on resolution, so measuring cameras at their native sizes would compare
    a 1080p frame against a 720p one and call the difference sharpness.
    """
    h, w = image_bgr.shape[:2]
    if w != work_width:
        scale = work_width / w
        image_bgr = cv2.resize(
            image_bgr, (work_width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return FrameQuality(
        mean_intensity=float(grey.mean()),
        rms_contrast=float(grey.std()),
        blur=float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        clipped_low=float((grey <= CLIP_LOW).mean()),
        clipped_high=float((grey >= CLIP_HIGH).mean()),
    )


def _mad(values: list[float], centre: float) -> float:
    return median([abs(v - centre) for v in values]) if values else 0.0


def build_baseline(camera_id: str, qualities: list[FrameQuality]) -> CameraBaseline:
    """What this camera normally produces."""
    contrasts = [q.rms_contrast for q in qualities]
    blurs = [q.blur for q in qualities]
    c_med = median(contrasts) if contrasts else 0.0
    b_med = median(blurs) if blurs else 0.0
    return CameraBaseline(
        camera_id=camera_id,
        n_frames=len(qualities),
        contrast_median=c_med,
        contrast_mad=_mad(contrasts, c_med),
        blur_median=b_med,
        blur_mad=_mad(blurs, b_med),
    )


def assess(quality: FrameQuality, baseline: CameraBaseline | None = None) -> QualityVerdict:
    """Judge one frame, against its own camera wherever possible.

    Exposure failure is absolute and applies with or without a baseline. Everything else
    needs one, and without a trustworthy baseline the answer is `unknown` rather than
    `good` - the distinction matters, because `good` is a claim and `unknown` is not.
    """
    reasons: list[str] = []
    if (failure := quality.exposure_failure()) is not None:
        reasons.append(failure)

    if baseline is None or not baseline.trustworthy:
        if reasons:
            return QualityVerdict("bad", tuple(reasons))
        return QualityVerdict("unknown", ("no baseline for this camera yet",))

    # both conditions, deliberately: unusual for this camera *and* materially worse. Either
    # alone produces false positives - the statistical test flags the bottom of a tight
    # distribution, and a bare percentage flags every camera that is simply soft.
    unusual_contrast = baseline.contrast_mad > 0 and quality.rms_contrast < baseline.contrast_floor
    material_contrast = quality.rms_contrast < baseline.contrast_median * MATERIAL_CONTRAST_DROP
    if unusual_contrast and material_contrast:
        reasons.append(
            f"contrast {quality.rms_contrast:.1f} is far below this camera's usual "
            f"{baseline.contrast_median:.1f}"
        )

    unusual_blur = baseline.blur_mad > 0 and quality.blur < baseline.blur_floor
    material_blur = quality.blur < baseline.blur_median * MATERIAL_BLUR_DROP
    if unusual_blur and material_blur:
        reasons.append(
            f"detail {quality.blur:.0f} is far below this camera's usual "
            f"{baseline.blur_median:.0f}; the lens may be dirty or wet"
        )
    return QualityVerdict("bad" if reasons else "good", tuple(reasons))
