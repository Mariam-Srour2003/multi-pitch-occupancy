"""Deliberately weak image features, for the baseline floor (H2, WP4-T10).

Before claiming that frozen transformer backbones are needed, the thesis has to show that
something trivial does not already do the job. On the current data a rule reading only the
clock scores 98.4%, so this is not a formality.

Three floors, in increasing order of effort:

* **majority class** — predicts the most common training label; the true zero point.
* **mean intensity** — one number per frame (how bright is it), fed to logistic
  regression. This is essentially "is it day or night", learned rather than hand-written.
* **colour histogram** — a coarse RGB histogram. Captures turf colour, floodlight cast and
  overall composition, but nothing about whether people are present.

None can represent "are there players on the pitch". If one of them matches a deep probe,
the benchmark is measuring scene recognition, and that is the finding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["mean_intensity", "colour_histogram", "build_cheap_features", "CHEAP_FEATURES"]


def mean_intensity(image_bgr: np.ndarray) -> np.ndarray:
    """One feature: mean grayscale intensity."""
    import cv2

    return np.array([cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).mean()], dtype=np.float32)


def colour_histogram(image_bgr: np.ndarray, *, bins: int = 16) -> np.ndarray:
    """Normalised per-channel RGB histogram, ``3 * bins`` features."""
    import cv2

    chans = []
    for c in range(3):
        h = cv2.calcHist([image_bgr], [c], None, [bins], [0, 256]).ravel()
        chans.append(h / max(h.sum(), 1.0))
    return np.concatenate(chans).astype(np.float32)


def _both(image_bgr: np.ndarray) -> np.ndarray:
    return np.concatenate([mean_intensity(image_bgr), colour_histogram(image_bgr)])


CHEAP_FEATURES = {
    "intensity": (mean_intensity, "mean grayscale intensity (1-d)"),
    "histogram": (colour_histogram, "RGB colour histogram, 16 bins/channel (48-d)"),
    "intensity+histogram": (_both, "both of the above (49-d)"),
}


def build_cheap_features(
    files: list[str], dataset_dir: Path, kind: str = "histogram"
) -> np.ndarray:
    """Compute a cheap feature matrix for ``files`` (paths relative to ``dataset_dir``)."""
    import cv2

    if kind not in CHEAP_FEATURES:
        raise KeyError(f"unknown feature {kind!r}; known: {', '.join(CHEAP_FEATURES)}")
    fn, _ = CHEAP_FEATURES[kind]
    out = []
    for f in files:
        img = cv2.imread(str(dataset_dir / f))
        if img is None:
            raise OSError(f"cannot read {dataset_dir / f}")
        out.append(fn(img))
    return np.stack(out)
