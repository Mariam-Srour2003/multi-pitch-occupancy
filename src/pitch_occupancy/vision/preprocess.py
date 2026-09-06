"""Frame preprocessing (WP3).

**One entry point, used by both the experiments and the live pipeline.** The pilot's most
expensive bug came from a feature-extraction path that differed between the two, so
:func:`preprocess` is deliberately the only way in; anything that needs a variant passes
options rather than reimplementing a stage.

Every switch here is a *hypothesis about what does not transfer between venues*, and the
input ablation showed those hypotheses are not independent: grayscale helped alone
(+0.022), a centre crop helped alone (+0.038), and both together scored *below* the
untouched baseline (-0.061). Removing information has a floor. That is why the switches
are searchable (`experiments/preprocess_search.py`) rather than chosen by argument.

Order matters and is fixed: geometry (undistort, ROI, crop) -> photometric (normalise,
CLAHE, gamma, saturation) -> resize -> post-resize effects (sharpen, blur). Blur and
sharpen come last so their radii are in *model-input* pixels, which is both the meaningful
unit and far cheaper than convolving a large kernel across a 1080p frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

__all__ = [
    "PreprocessConfig",
    "preprocess",
    "letterbox",
    "apply_clahe",
    "rms_contrast",
    "roi_mask",
    "undistort_fisheye",
    "SWITCHES",
]


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Every switch that changes what the model sees. Hashed into the feature cache."""

    size: int = 224

    # --- geometry ---
    undistort: float = 0.0
    """Barrel-distortion correction strength, 0 = off. The footage is heavily fisheye and
    the lens differs between venues, so geometry is itself a venue cue; straightening it
    should make a pitch look more like a pitch and less like *that* pitch."""

    roi: bool = False
    """Mask to the pitch polygon. The measured stand-in (blind centre crop) was the single
    best variant tested, so this is expected to help - the border carries stands, sky and
    adjacent pitches, none of which transfer."""

    centre_crop: float = 1.0
    """Keep the central fraction. 1.0 = whole frame."""

    top_crop: float = 0.0
    """Discard this fraction from the top only. More targeted than a symmetric centre crop:
    sky, stands and adjacent pitches sit above the horizon, while the pitch does not."""

    letterbox: bool = True
    """Aspect-preserving resize with grey padding. Worth ~0.03 recall over a squashing
    resize on this fisheye footage."""

    # --- photometric ---
    per_image_standardise: bool = False
    """Z-score the image. Removes the global brightness/contrast offset that separates a
    daylight morning from a floodlit night - i.e. attacks the day/night confound directly."""

    clahe: str = "off"  # off | on | auto
    clahe_contrast_below: float = 40.0

    gamma: float = 1.0
    """<1 brightens, >1 darkens. Night frames are dark; a lift may expose players."""

    saturation: float = 1.0
    """Scale colour saturation. 0.0 == grayscale. Full grayscale helped alone but broke
    when combined with cropping, so this is a dial rather than a switch - partial
    desaturation may sit above the information floor where full desaturation does not."""

    denoise: float = 0.0
    """Bilateral filter strength. Night footage is noisy; smoothing noise without losing
    edges may help where a plain blur hurts."""

    # --- post-resize ---
    sharpen: float = 0.0
    """Unsharp-mask amount, counteracting fisheye softness at the frame edge."""

    blur_sigma: float = 0.0
    """Gaussian blur in model-input pixels. Primarily a diagnostic: it removes the detail
    that people are made of, so a score that survives it was never about people."""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def describe(self) -> str:
        """Only the non-default switches, for a compact label."""
        base = PreprocessConfig()
        on = [
            f"{k}={v}"
            for k, v in self.as_dict().items()
            if v != getattr(base, k) and k != "clahe_contrast_below"
        ]
        return " ".join(on) if on else "baseline"


#: Search space: switch name -> values to try. Consumed by the preprocessing search.
#:
#: ``roi`` is deliberately absent. Masking needs a per-camera polygon, none have been drawn
#: yet (WP3-T1), and :func:`roi_mask` with no polygon returns the frame untouched - so
#: searching it would evaluate a no-op and record "ROI does not help", which is false. It
#: joins the search the day polygons exist.
SWITCHES: dict[str, list] = {
    "undistort": [0.15, 0.30],
    "centre_crop": [0.5, 0.7],
    "top_crop": [0.2, 0.35],
    "letterbox": [False],
    "per_image_standardise": [True],
    "clahe": ["on", "auto"],
    "gamma": [0.7, 1.4],
    "saturation": [0.0, 0.5],
    "denoise": [25.0],  # weaker values barely alter the frame - see the no-op test
    "sharpen": [0.6],
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


def undistort_fisheye(image_bgr: np.ndarray, strength: float) -> np.ndarray:
    """Approximate barrel-distortion correction.

    No per-camera calibration exists, so this uses a single radial coefficient rather than
    a true intrinsic matrix. It is a normalisation, not a rectification: the aim is to make
    two venues' geometry more similar, not to recover metric straight lines.
    """
    if strength <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    f = max(w, h)
    camera = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
    dist = np.array([-strength, 0.0, 0.0, 0.0], np.float32)
    return cv2.undistort(image_bgr, camera, dist)


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


def _apply_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image_bgr, lut)


def _apply_saturation(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image_bgr
    if scale <= 0.0:
        return cv2.cvtColor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= scale
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def _standardise(image_bgr: np.ndarray) -> np.ndarray:
    """Z-score to zero mean / unit variance, then rescale to 8-bit."""
    f = image_bgr.astype(np.float32)
    std = f.std()
    if std < 1e-6:
        return image_bgr
    z = (f - f.mean()) / std
    return np.clip(z * 48.0 + 128.0, 0, 255).astype(np.uint8)


def _centre_crop(image_bgr: np.ndarray, frac: float) -> np.ndarray:
    if frac >= 1.0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
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

    # --- geometry ---
    if cfg.undistort > 0:
        img = undistort_fisheye(img, cfg.undistort)
    if cfg.roi:
        img = roi_mask(img, polygon)
    if cfg.top_crop > 0:
        img = img[int(img.shape[0] * cfg.top_crop) :, :]
    img = _centre_crop(img, cfg.centre_crop)

    # --- photometric ---
    if cfg.per_image_standardise:
        img = _standardise(img)
    if cfg.clahe == "on" or (cfg.clahe == "auto" and rms_contrast(img) < cfg.clahe_contrast_below):
        img = apply_clahe(img)
    if cfg.gamma != 1.0:
        img = _apply_gamma(img, cfg.gamma)
    if cfg.saturation != 1.0:
        img = _apply_saturation(img, cfg.saturation)
    if cfg.denoise > 0:
        # diameter scales with strength: a fixed small neighbourhood leaves the frame
        # essentially untouched, which the no-op test caught
        d = max(5, int(cfg.denoise / 2) | 1)
        img = cv2.bilateralFilter(img, d, cfg.denoise * 4, cfg.denoise * 4)

    # --- resize, then output-resolution effects ---
    out = letterbox(img, cfg.size) if cfg.letterbox else cv2.resize(
        img, (cfg.size, cfg.size), interpolation=cv2.INTER_AREA
    )
    if cfg.sharpen > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1 + cfg.sharpen, blurred, -cfg.sharpen, 0)
    if cfg.blur_sigma > 0:
        k = max(3, int(cfg.blur_sigma * 6) | 1)
        out = cv2.GaussianBlur(out, (k, k), cfg.blur_sigma)
    return out
