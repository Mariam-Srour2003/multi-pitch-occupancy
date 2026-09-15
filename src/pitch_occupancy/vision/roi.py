"""Per-camera pitch boundaries: say what is inside the pitch, and ignore the rest (WP3-T1).

`vision/__init__.py` has listed this module since the project began and it has never existed,
which is why `PreprocessConfig.roi` was a switch nothing could turn on: `roi_mask` with no
polygon returns the frame untouched, so the preprocessing search left `roi` out rather than
record "ROI does not help" about a no-op. This is the missing half.

**The problem it solves is not clutter, it is a neighbour.** Five-a-side pitches are built in
rows, and a camera watching pitch 2 sees pitches 1 and 3 down the sides of its frame. A match
on the next pitch over puts real players, moving realistically, into frames where *this* pitch
is empty - so the model is right about what it sees and wrong about what was asked. No amount
of training fixes that, because the evidence genuinely is there; the question was under-
specified. A boundary specifies it.

**A polygon is per camera, not per pitch.** The two cameras on one pitch see it from opposite
ends and their neighbours fall on opposite sides of the frame, so one shared outline would be
wrong for at least one of them.

**Normalised coordinates, so a resolution change does not silently invalidate the boundary.**
Points are fractions of width and height in 0-1. A camera swapped for a 4K one keeps its
outline; a polygon in pixels would have quietly masked the wrong region.

**Filling with black is the default and is not obviously right.** It is what `roi_mask` has
always done and what the preprocessing search would have evaluated, so it stays the default -
but a large black region is a thing no pretraining set contains, and the backbone's response
to it is itself a distribution shift. `blur` keeps the frame's statistics plausible while
destroying the detail a person is made of, and `mean` is the gentlest edge. Which is right is
an empirical question this module deliberately does not answer: it offers the three and the
editor shows what each does to the prediction, on the user's own frames.

Stored in `configs/roi.json`, keyed by camera. Unlike `configs/cameras.json` - which holds
RTSP credentials and is gitignored for that reason - a boundary is geometry, carries nothing
secret, and belongs in the repository: it is part of what a site *is*, and re-deriving it by
hand after a clone is exactly the sort of setup nobody repeats identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "Polygon", "STORE", "FILLS", "DEFAULT_FILL",
    "load_all", "get", "save", "remove", "validate", "coverage", "apply",
]

#: A polygon is a list of ``[x, y]`` pairs, each a fraction of the frame in 0-1.
Polygon = list[list[float]]

STORE = Path(__file__).resolve().parents[3] / "configs" / "roi.json"

#: How the outside of the boundary is filled. See the module docstring for why the default
#: is the least defensible of the three and is the default anyway.
FILLS = ("black", "blur", "mean")
DEFAULT_FILL = "black"

#: Below this, the boundary is almost certainly a mistake - three points clicked by accident,
#: or an outline drawn round a corner flag. Refused rather than saved, because a polygon that
#: masks 99% of the pitch produces confident nonsense rather than an obvious failure.
MIN_COVERAGE = 0.02


def validate(polygon: Polygon) -> Polygon:
    """Return the polygon as floats, or raise ``ValueError`` saying what is wrong with it.

    Validation lives here rather than in the API layer because the file can be hand-edited
    and the live worker reads it directly - a boundary that was fine when it was saved and
    is malformed when it is read would fail at the worst moment, on a camera, at night.
    """
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        raise ValueError("a boundary needs at least 3 points")
    clean: Polygon = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"each point must be [x, y], got {point!r}")
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"points are fractions of the frame, so both must be in 0-1; got [{x}, {y}]"
            )
        clean.append([x, y])
    area = coverage(clean)
    if area < MIN_COVERAGE:
        raise ValueError(
            f"this boundary keeps {area:.1%} of the frame, which is almost certainly a "
            f"mis-click - it would leave the model nothing to look at"
        )
    return clean


def coverage(polygon: Polygon) -> float:
    """Fraction of the frame the boundary keeps, by the shoelace formula.

    Reported to the editor beside the outline: "you are about to discard 73% of what this
    camera sees" is the sentence that catches a boundary drawn round the wrong pitch, and it
    is available before anything is saved or any model is run.
    """
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def load_all() -> dict[str, Polygon]:
    """Every saved boundary, keyed by camera. A missing or malformed store yields none.

    Malformed rather than raising, and deliberately: this is read on the live path, and a
    boundary file someone broke while editing should degrade to "no boundary" - which is the
    behaviour the system had before boundaries existed - rather than stop a night's capture.
    The editor validates on save, which is where a person is present to be told.
    """
    if not STORE.exists():
        return {}
    try:
        raw = json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Polygon] = {}
    for key, value in raw.items():
        if key.startswith("_"):  # comment keys, as in configs/cameras.example.json
            continue
        try:
            out[str(key)] = validate(value)
        except (ValueError, TypeError):
            continue
    return out


def get(camera: str) -> Polygon | None:
    """The boundary for one camera, or None. None means *no boundary*, never an empty one."""
    return load_all().get(camera)


def save(camera: str, polygon: Polygon) -> Polygon:
    """Validate and store one camera's boundary, preserving every other entry."""
    clean = validate(polygon)
    if not camera or not camera.strip():
        raise ValueError("a boundary needs a camera to belong to")
    current = load_all()
    current[camera.strip()] = clean
    _write(current)
    return clean


def remove(camera: str) -> bool:
    """Forget one camera's boundary. Returns whether there was one to forget."""
    current = load_all()
    if camera not in current:
        return False
    del current[camera]
    _write(current)
    return True


def _write(polygons: dict[str, Polygon]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": [
            "Pitch boundaries, one per camera, drawn in the editor at /roi.",
            "Points are [x, y] as fractions of the frame (0-1), so a resolution change does",
            "not invalidate them. Everything outside the outline is filled before the frame",
            "reaches the model - see src/pitch_occupancy/vision/roi.py.",
            "Committed on purpose: a boundary is geometry, not a credential, and re-drawing",
            "one by hand after a clone is setup nobody repeats identically.",
        ],
        **{k: polygons[k] for k in sorted(polygons)},
    }
    STORE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply(image_bgr: np.ndarray, polygon: Polygon | None, *,
          fill: str = DEFAULT_FILL) -> np.ndarray:
    """Suppress everything outside ``polygon``. No polygon returns the frame untouched.

    That last sentence is the reason `roi` stayed out of the preprocessing search: a no-op
    that looks like a switch will be measured as "makes no difference", and the conclusion
    will be recorded about ROI masking rather than about the absence of a polygon.

    ``fill`` is one of :data:`FILLS`:

    * ``black`` - zeroes the outside. What `roi_mask` has always done, and the hardest edge.
    * ``blur`` - a heavy Gaussian outside the outline. Destroys the detail a person is made
      of, which is what the neighbouring pitch has to lose, while leaving the frame's
      statistics closer to something the backbone has seen.
    * ``mean`` - flat fill with the mean colour *inside* the outline, the gentlest edge.
    """
    import cv2
    import numpy as np

    if not polygon:
        return image_bgr
    if fill not in FILLS:
        raise ValueError(f"unknown fill {fill!r}; known: {', '.join(FILLS)}")

    h, w = image_bgr.shape[:2]
    points = np.array([[int(round(x * w)), int(round(y * h))] for x, y in polygon],
                      dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [points], 255)

    if fill == "black":
        return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    if fill == "blur":
        # Kernel scaled to the frame so the same boundary behaves the same way on a 1080p
        # and a 4K camera. Odd, because OpenCV requires it.
        k = max(21, (min(h, w) // 12) | 1)
        outside = cv2.GaussianBlur(image_bgr, (k, k), 0)
    else:
        inside_mean = cv2.mean(image_bgr, mask=mask)[:3]
        outside = np.full_like(image_bgr, np.array(inside_mean, dtype=np.uint8))

    return np.where(mask[:, :, None].astype(bool), image_bgr, outside)
