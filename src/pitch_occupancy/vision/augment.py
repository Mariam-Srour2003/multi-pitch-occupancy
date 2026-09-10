"""Train-time augmentation (WP3-T6).

Augmentation is the complement to preprocessing, and the input ablation is what makes the
distinction sharp. Removing information helps in exactly one place and stops: **grayscale**
gains recall *and* takes false-play from 0.230 to 0.021 at 0.979 empty accuracy, which is a
real improvement on both axes. A centre crop appeared to gain more, and did not — 0.998
recall with **empty accuracy 0.000**, a variant that never identifies an empty pitch on an
axis whose folds contain none to catch it (corrected 2026-09-11 when the ablation gained its
false-play control). The two together fall below the untouched baseline and call 98.8% of
empty pitches a match. There is a floor, and only one removal is above it.

Augmentation does not have that floor, because **nothing is discarded at inference time**.
Where grayscale threw colour away permanently, colour jitter leaves the pixels intact and
teaches the model not to lean on them - the same shortcut targeted, without crossing the
information floor. That is the hypothesis this module exists to test.

**No rotations, warps or perspective changes.** The cameras are bolted to a post and see one
view forever; a rotated pitch is not a harder example, it is an impossible one, and training
on it spends capacity on a case that will never arrive. Everything here varies illumination,
weather and colour - the things that genuinely differ between one recording and the next.

The one geometric exception is a horizontal flip, which is deliberate: it produces a mirror
of a scene the camera never sees, which is exactly why it is useful. Flipping cannot change
whether people are playing, but it does break memorisation of *this particular pitch's*
layout - the failure mode the whole cross-venue evaluation exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["AugmentConfig", "augment", "synthetic_rain", "synthetic_fog", "AUGMENTATIONS"]


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Ranges sampled per image. All default to off so a run must opt in."""

    brightness: float = 0.0
    """+/- fraction of full scale. Floodlights vary between venues and across a night."""

    contrast: float = 0.0
    """+/- fraction. Haze, glare and lens dirt all flatten contrast."""

    saturation: float = 0.0
    """+/- fraction. Turf hue is a venue cue that does not transfer - jitter it so the
    model cannot use it, without discarding it as grayscale does."""

    hue: float = 0.0
    """+/- degrees (OpenCV hue is 0-179). Small values only; a large shift turns grass
    into a colour no pitch has, which is noise rather than a harder example."""

    gamma: float = 0.0
    """+/- exponent offset. Night footage crushes shadows; varying gamma covers both."""

    noise: float = 0.0
    """Gaussian sigma in 8-bit levels. Night sensors are noisy."""

    fog: float = 0.0
    """Max haze strength, 0-1. The night footage already contains fog; this covers more."""

    rain: float = 0.0
    """Max streak density, 0-1. Nothing in the dataset is wet - see IDEAS #1 - so this is
    the only rain the model can currently learn from, and it is synthetic. A model trained
    on it has *not* been shown to work in real rain."""

    flip: float = 0.0
    """Probability of a horizontal mirror. Breaks pitch-layout memorisation."""

    p: float = 0.5
    """Probability each enabled effect is applied to a given image."""

    seed: int | None = None

    def enabled(self) -> list[str]:
        return [
            k for k in ("brightness", "contrast", "saturation", "hue", "gamma",
                        "noise", "fog", "rain", "flip")
            if getattr(self, k) > 0
        ]


#: Presets. `colour` is the one the ablation motivates directly.
AUGMENTATIONS: dict[str, AugmentConfig] = {
    "none": AugmentConfig(),
    "colour": AugmentConfig(brightness=0.2, contrast=0.2, saturation=0.35, hue=6),
    "weather": AugmentConfig(fog=0.4, rain=0.4, noise=6),
    "light": AugmentConfig(brightness=0.25, gamma=0.35, noise=5),
    "full": AugmentConfig(
        brightness=0.2, contrast=0.2, saturation=0.35, hue=6,
        gamma=0.3, noise=5, fog=0.3, rain=0.3, flip=0.5,
    ),
}


def synthetic_fog(image_bgr: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Depth-free haze: blend toward a bright grey, heavier toward the top of the frame.

    Real fog thickens with distance, and in a fixed pitch camera distance increases up the
    frame, so a vertical gradient approximates it far better than a uniform veil.
    """
    h, w = image_bgr.shape[:2]
    veil = float(np.clip(strength, 0, 1)) * rng.uniform(0.5, 1.0)
    gradient = np.linspace(1.0, 0.35, h, dtype=np.float32)[:, None, None]
    grey = np.full_like(image_bgr, 205, dtype=np.float32)
    out = image_bgr.astype(np.float32) * (1 - veil * gradient) + grey * (veil * gradient)
    return np.clip(out, 0, 255).astype(np.uint8)


def synthetic_rain(
    image_bgr: np.ndarray, strength: float, rng: np.random.Generator
) -> np.ndarray:
    """Oriented motion-blurred streaks, plus the contrast drop wet air causes.

    Drawn rather than sampled from real footage because no real footage is wet. Streaks
    share one angle per image, since rain in a single frame falls in one direction.

    **Every dimension is a fraction of frame height, never a pixel count.** Augmentation
    runs before preprocessing's resize, so absolute sizes do not survive: streaks tuned to
    look right on a 1080p frame become invisible hairlines at 224, and streaks tuned at 224
    become white poles on the source frame. The visual grid caught exactly that.
    """
    h, w = image_bgr.shape[:2]
    s = float(np.clip(strength, 0, 1))
    # density per unit of frame *area*, so it reads the same at any resolution
    n = int(s * 280 * (w / h))
    if n <= 0:
        return image_bgr

    layer = np.zeros((h, w), np.uint8)
    angle = rng.uniform(-25, 25)
    length = max(2, int(rng.uniform(0.02, 0.05) * h))  # a raindrop's 1/30 s smear
    thickness = max(1, round(h * 0.004))
    dx, dy = int(length * np.sin(np.radians(angle))), length
    for _ in range(n):
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        cv2.line(layer, (x, y), (x + dx, y + dy), 255, thickness)

    k = max(3, length // 3 | 1)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    kernel = cv2.warpAffine(
        kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle - 90, 1.0), (k, k)
    )
    layer = cv2.filter2D(layer, -1, kernel)

    out = image_bgr.astype(np.float32)
    out = out * (1 - 0.18 * s) + 128 * (0.18 * s)  # wet air flattens contrast
    out += layer[:, :, None].astype(np.float32) * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def _jitter_hsv(img: np.ndarray, sat: float, hue: float, rng) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if sat:
        hsv[:, :, 1] *= 1 + rng.uniform(-sat, sat)
    if hue:
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-hue, hue)) % 180
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment(
    image_bgr: np.ndarray,
    config: AugmentConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply one random augmentation draw. Returns a new array; the input is untouched.

    **Training only.** Applying this at inference would make predictions non-deterministic
    for the same frame, which an audit trail cannot survive.
    """
    cfg = config or AugmentConfig()
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    out = image_bgr.copy()

    def maybe() -> bool:
        return rng.random() < cfg.p

    if cfg.flip and rng.random() < cfg.flip:
        out = cv2.flip(out, 1)
    if cfg.brightness and maybe():
        out = np.clip(out.astype(np.float32) + rng.uniform(-cfg.brightness, cfg.brightness) * 255,
                      0, 255).astype(np.uint8)
    if cfg.contrast and maybe():
        f = 1 + rng.uniform(-cfg.contrast, cfg.contrast)
        out = np.clip((out.astype(np.float32) - 128) * f + 128, 0, 255).astype(np.uint8)
    if (cfg.saturation or cfg.hue) and maybe():
        out = _jitter_hsv(out, cfg.saturation, cfg.hue, rng)
    if cfg.gamma and maybe():
        g = 1 + rng.uniform(-cfg.gamma, cfg.gamma)
        lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    if cfg.fog and maybe():
        out = synthetic_fog(out, cfg.fog, rng)
    if cfg.rain and maybe():
        out = synthetic_rain(out, cfg.rain, rng)
    if cfg.noise and maybe():
        out = np.clip(out.astype(np.float32) + rng.normal(0, cfg.noise, out.shape),
                      0, 255).astype(np.uint8)
    return out
