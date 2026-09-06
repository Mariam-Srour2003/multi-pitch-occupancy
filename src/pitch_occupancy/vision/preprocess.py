"""Frame preprocessing (WP3).

**One entry point, used by both the experiments and the live pipeline.** The pilot's most
expensive bug came from a feature-extraction path that differed between the two, so
:func:`preprocess` is deliberately the only way in; anything that needs a variant passes
options rather than reimplementing a stage.

Stages, in order:

1. **ROI mask** - black out everything outside the pitch polygon. Not cosmetic: the
   footage shows neighbouring pitches, and an unmasked classifier will happily report play
   on the field next door.
2. **Letterbox** - aspect-preserving resize with grey padding, rather than a squashing
   thumbnail. The cameras are heavily fisheye and distortion is already the hard part.
3. **Photometric variants** - CLAHE for the low-contrast night and fog frames.

The degradation options (``grayscale``, ``blur``, ``centre_crop``) exist for the input
ablation: feeding a model inputs with specific information removed is how you find out
what it was using. They are diagnostics, never production settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["PreprocessConfig", "preprocess", "letterbox", "apply_clahe", "rms_contrast", "roi_mask"]


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Every switch that changes what the model sees. Hashed into the feature cache."""

    size: int = 224
    roi: bool = False
    letterbox: bool = True
    clahe: str = "off"  # off | on | auto
    clahe_contrast_below: float = 40.0  # RMS below which "auto" engages
    # --- diagnostics only ---
    grayscale: bool = False
    blur_sigma: float = 0.0
    centre_crop: float = 1.0  # 1.0 = whole frame; 0.6 = central 60%

    def as_dict(self) -> dict[str, object]:
        return {
            "size": self.size, "roi": self.roi, "letterbox": self.letterbox,
            "clahe": self.clahe, "grayscale": self.grayscale,
            "blur_sigma": self.blur_sigma, "centre_crop": self.centre_crop,
        }


def rms_contrast(image_bgr: np.ndarray) -> float:
    """Standard deviation of grayscale intensity - the low-light/fog gate."""
    return float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).std())


def roi_mask(image_bgr: np.ndarray, polygon: list[list[float]] | None) -> np.ndarray:
    """Black out everything outside ``polygon`` (normalised 0-1 coordinates)."""
    if not polygon:
        return image_bgr
    h, w = image_bgr.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in polygon], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def letterbox(image_bgr: np.ndarray, size: int, *, pad: int = 114) -> np.ndarray:
    """Aspect-preserving resize onto a square canvas with grey padding."""
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), pad, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def apply_clahe(image_bgr: np.ndarray, *, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE on the L channel in LAB, so colour is not shifted."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _centre_crop(image_bgr: np.ndarray, frac: float) -> np.ndarray:
    if frac >= 1.0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    top, left = (h - ch) // 2, (w - cw) // 2
    return image_bgr[top : top + ch, left : left + cw]


def preprocess(
    image_bgr: np.ndarray,
    config: PreprocessConfig | None = None,
    *,
    polygon: list[list[float]] | None = None,
) -> np.ndarray:
    """The single preprocessing path. Returns a ``size x size`` BGR frame."""
    cfg = config or PreprocessConfig()
    img = image_bgr

    if cfg.roi:
        img = roi_mask(img, polygon)
    img = _centre_crop(img, cfg.centre_crop)

    if cfg.clahe == "on" or (cfg.clahe == "auto" and rms_contrast(img) < cfg.clahe_contrast_below):
        img = apply_clahe(img)

    if cfg.grayscale:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    out = letterbox(img, cfg.size) if cfg.letterbox else cv2.resize(
        img, (cfg.size, cfg.size), interpolation=cv2.INTER_AREA
    )

    # Blur last, at output resolution. `blur_sigma` is therefore in *model-input* pixels,
    # which is both the meaningful unit ("how much detail does the model lose") and orders
    # of magnitude cheaper than convolving a large kernel across a 1080p frame first.
    if cfg.blur_sigma > 0:
        k = max(3, int(cfg.blur_sigma * 6) | 1)  # odd kernel spanning +/-3 sigma
        out = cv2.GaussianBlur(out, (k, k), cfg.blur_sigma)
    return out
