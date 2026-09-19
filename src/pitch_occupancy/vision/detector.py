"""One detector behind one registry (A36, WP9-T2).

The detector-first path counts what stands on the pitch, so the detector is the model, and a
model needs the two things `vision/backbones.py` gives the frozen backbones: a registry that
names the candidates, and a loader that every caller shares. Until now the only detector in
the repository was the string ``"yolov8n.pt"``, passed around by five callers as a default
argument, with the thread-local cache that makes it affordable living in `vision/explain.py`
beside the heatmap code.

**What a registry entry says.** The key an experiment or a setting names it by, the weights
file ultralytics loads it from, whether it produces instance masks, and the long edge it runs
at. Nothing else - the person confidence, the ball confidence and the tiling are *rule*
parameters (`configs/rules.json`, WP9-T5), because they are what gets tuned, and a detector
key must mean the same model whatever they are set to.

**Detections carry a mask when a segmentation model produced one, and None otherwise.** The
overlay draws the mask; the count never reads it - a person is placed by the foot of their
box (`vision/people.py`), so the two model families count identically and differ only in what
they can show. That is deliberate: the selection in WP9-T2 is on counting, and masks are for
the reader.

**Tiling.** A far-side player on a fisheye 1080p frame is a few pixels tall at the detector's
input size, and A16 measured the difference between 640 and 1280 as the difference between
"finds nobody" and a usable count. `tiles=2` runs the frame as four overlapping quarters plus
the whole, and merges by class-wise non-maximum suppression. It costs roughly five forward
passes and is a candidate, not a default: `experiments/detector_audit.py` measures whether it
buys anything on the hand counts.

**Failure is None, never an empty list.** A missing weights file, an import error or a
prediction that raises all return ``None`` - "not checked" - and the rule engine turns that
into UNCERTAIN. ``[]`` means the frame was examined and nothing was found. Collapsing the two
is how a broken install comes to report every pitch empty (`vision/explain.py`).

**Weights are not in the repository** (``*.pt`` is gitignored). ``pitch fetch-weights``
downloads the registered ones deliberately; nothing here downloads as a side effect of a
prediction, because a worker that reaches for the network at 3 am on its first frame is a
worker that fails at 3 am.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PERSON", "SPORTS_BALL", "VEHICLES", "DEFAULT_DETECTOR", "DetectorSpec", "DETECTORS",
    "Detection", "Detector", "weights_path", "fetch_weights",
]

#: COCO class ids the rules read. `sports ball` is 32; the vehicles are what a groundskeeping
#: machine or a car parked on the surface would be detected as (A36 row 5, best effort).
PERSON = 0
SPORTS_BALL = 32
VEHICLES = (1, 2, 3, 5, 7)  # bicycle, car, motorcycle, bus, truck

#: Where weights live: the repository root, beside the `yolov8n.pt` every published count
#: was measured with. Gitignored there. A bare filename would make ultralytics download into
#: whatever the working directory happens to be.
WEIGHTS_DIR = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    key: str
    weights: str
    segment: bool
    imgsz: int = 1280
    note: str = ""


DETECTORS: dict[str, DetectorSpec] = {
    "yolov8n": DetectorSpec(
        "yolov8n", "yolov8n.pt", False,
        note="the gate's detector since A16; every published count was measured with it"),
    "yolo11n": DetectorSpec(
        "yolo11n", "yolo11n.pt", False,
        note="the same size class, one generation on; the like-for-like candidate"),
    "yolov8s": DetectorSpec(
        "yolov8s", "yolov8s.pt", False,
        note="roughly 3x the compute of n; the question is far-side recall"),
    "yolo11s": DetectorSpec(
        "yolo11s", "yolo11s.pt", False,
        note="as yolov8s, one generation on"),
    "yolov8n-seg": DetectorSpec(
        "yolov8n-seg", "yolov8n-seg.pt", True,
        note="instance masks for the overlay; counts by box like the rest"),
    "yolo11n-seg": DetectorSpec(
        "yolo11n-seg", "yolo11n-seg.pt", True,
        note="instance masks, one generation on"),
    "yolo11s-seg": DetectorSpec(
        "yolo11s-seg", "yolo11s-seg.pt", True,
        note="instance masks at the s size; the ceiling of what the cycle affords"),
}

#: What runs until WP9-T2's audit says otherwise: the detector the published counts rest on.
DEFAULT_DETECTOR = "yolov8n"


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in frame pixels; ``mask`` is a full-frame boolean array or None."""

    cls: int
    box: tuple[int, int, int, int]
    conf: float
    mask: np.ndarray | None = None

    @property
    def foot(self) -> tuple[int, int]:
        """Where the object stands: the bottom centre of its box (`vision/people.py`)."""
        x1, _y1, x2, y2 = self.box
        return (x1 + x2) // 2, y2

    @property
    def centre(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def weights_path(spec: DetectorSpec) -> Path:
    return WEIGHTS_DIR / spec.weights


#: One model per weights file per thread, built on first use. Thread-local for the reason
#: `vision/explain.py` gives: `evaluation/latency.py` measures concurrency on several threads
#: and an ultralytics model is not documented as safe to predict on from more than one.
_MODELS = threading.local()


def _load_model(spec: DetectorSpec, *, allow_download: bool = False):
    """The ultralytics model for ``spec`` on this thread, or None if it cannot be had.

    Without ``allow_download`` a missing weights file is a None, not a network request -
    see the module docstring. `fetch_weights` is the one caller that passes True.
    """
    cache = getattr(_MODELS, "models", None)
    if cache is None:
        cache = _MODELS.models = {}
    if spec.weights in cache:
        return cache[spec.weights]
    path = weights_path(spec)
    if not path.exists() and not allow_download:
        return None
    try:
        from ultralytics import YOLO
    except ImportError:  # pragma: no cover - dependency is pinned
        return None
    try:
        # A missing file is passed as a bare name so ultralytics fetches it into WEIGHTS_DIR
        # (its download lands in the working directory, hence the chdir below).
        if path.exists():
            model = YOLO(str(path))
        else:
            import os

            previous = Path.cwd()
            os.chdir(WEIGHTS_DIR)
            try:
                model = YOLO(spec.weights)
            finally:
                os.chdir(previous)
    except Exception:  # noqa: BLE001 - a missing or corrupt weights file must fail closed
        return None
    cache[spec.weights] = model
    return model


class Detector:
    """A registered detector: ``detect`` returns detections or None, never raises."""

    def __init__(self, spec: DetectorSpec) -> None:
        self.spec = spec

    @classmethod
    def load(cls, key: str = DEFAULT_DETECTOR) -> Detector:
        """A detector for a registry key. Raises on an unknown key; a missing weights file is
        found out on the first `detect`, which returns None."""
        if key not in DETECTORS:
            raise KeyError(f"unknown detector {key!r}; known: {', '.join(sorted(DETECTORS))}")
        return cls(DETECTORS[key])

    @property
    def available(self) -> bool:
        return _load_model(self.spec) is not None

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        confidence: float = 0.1,
        classes: Sequence[int] = (PERSON, SPORTS_BALL),
        imgsz: int | None = None,
        tiles: int = 1,
        masks: bool = False,
    ) -> list[Detection] | None:
        """Every object of ``classes`` at or above ``confidence``, or None if not checked.

        ``confidence`` is the floor for the *pass*; callers apply per-class thresholds on top
        (a ball is held to a lower one than a person, `vision/people.py`). ``masks`` asks a
        segmentation model for its masks; a detection model ignores it. ``tiles > 1`` runs
        the frame as ``tiles x tiles`` overlapping crops plus the whole and merges them.
        """
        model = _load_model(self.spec)
        if model is None:
            return None
        size = imgsz or self.spec.imgsz
        want_masks = masks and self.spec.segment
        if tiles <= 1:
            return self._predict(model, image_bgr, confidence, classes, size, want_masks)
        return self._tiled(model, image_bgr, confidence, classes, size, want_masks, tiles)

    # --- internals ------------------------------------------------------------------

    def _predict(self, model, image_bgr, confidence, classes, imgsz, want_masks,
                 *, offset: tuple[int, int] = (0, 0),
                 frame_shape: tuple[int, int] | None = None) -> list[Detection] | None:
        try:
            results = model.predict(image_bgr, verbose=False, conf=confidence,
                                    classes=list(classes), imgsz=imgsz)
        except Exception:  # noqa: BLE001 - fail closed
            return None
        height, width = frame_shape or image_bgr.shape[:2]
        ox, oy = offset
        out: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            raw = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            polygons = None
            if want_masks and getattr(result, "masks", None) is not None:
                polygons = list(result.masks.xy)
            for i, ((x1, y1, x2, y2), c, p) in enumerate(zip(raw, cls, conf, strict=True)):
                x1, y1 = max(int(x1) + ox, 0), max(int(y1) + oy, 0)
                x2, y2 = min(int(x2) + ox, width), min(int(y2) + oy, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                mask = None
                if polygons is not None and i < len(polygons) and len(polygons[i]):
                    mask = _rasterise(polygons[i], (height, width), offset)
                out.append(Detection(int(c), (x1, y1, x2, y2), float(p), mask))
        return out

    def _tiled(self, model, image_bgr, confidence, classes, imgsz, want_masks, tiles):
        height, width = image_bgr.shape[:2]
        found = self._predict(model, image_bgr, confidence, classes, imgsz, want_masks)
        if found is None:
            return None
        overlap = 0.1
        tile_h, tile_w = height / tiles, width / tiles
        for row in range(tiles):
            for col in range(tiles):
                y1 = max(0, int((row - overlap) * tile_h))
                y2 = min(height, int((row + 1 + overlap) * tile_h))
                x1 = max(0, int((col - overlap) * tile_w))
                x2 = min(width, int((col + 1 + overlap) * tile_w))
                part = self._predict(model, image_bgr[y1:y2, x1:x2], confidence, classes,
                                     imgsz, want_masks, offset=(x1, y1),
                                     frame_shape=(height, width))
                if part is None:
                    return None
                found.extend(part)
        return merge(found)


def merge(detections: Iterable[Detection], *, iou_threshold: float = 0.5) -> list[Detection]:
    """Class-wise non-maximum suppression: the highest-confidence box of each overlapping
    group survives. Pure, so the tile merge is testable without a model."""
    survivors: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.conf, reverse=True):
        if all(d.cls != det.cls or _iou(d.box, det.box) < iou_threshold for d in survivors):
            survivors.append(det)
    return survivors


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _rasterise(polygon, shape: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, np.uint8)
    pts = np.asarray(polygon, dtype=np.float32)
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def fetch_weights(keys: Sequence[str] | None = None) -> list[tuple[str, Path, bool]]:
    """Download the weights for ``keys`` (default: every registered detector) into
    `WEIGHTS_DIR`. Returns ``(key, path, present_afterwards)`` per key.

    The one deliberate network access in this module. Run as ``pitch fetch-weights``.
    """
    out: list[tuple[str, Path, bool]] = []
    for key in keys or sorted(DETECTORS):
        spec = DETECTORS[key]
        _load_model(spec, allow_download=True)
        out.append((key, weights_path(spec), weights_path(spec).exists()))
    return out
