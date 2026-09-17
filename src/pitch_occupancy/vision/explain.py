"""Where the model is looking (WP4-T5's XAI half).

Two independent views of the same frame, because one of them can be wrong in a way the other
cannot:

* :func:`class_evidence_map` - **an exact decomposition, not a saliency heuristic.** Every
  probe in this project is a logistic regression on *mean-pooled* frozen features, and that
  composition is linear all the way down, so a class score is exactly the mean over spatial
  positions of a per-position contribution. There is no gradient to approximate and no
  smoothing to tune: the map sums to the score. Grad-CAM exists because a head is usually
  non-linear over a feature map; here it is not, so the approximation is unnecessary and the
  honest thing is to say what the score is actually made of.
* :func:`attention_rollout` - what the transformer attended to, independent of any classifier.
  Available for ViT and DINOv2 and not for ConvNeXtV2, which has no attention.

They answer different questions. The first says *which regions the probe's decision came
from*; the second says *which regions the backbone propagated information from*. Agreement
between them is weak evidence the model is reading the pitch; disagreement is worth a look.

**Faces are pixelated before anything is written to disk.** The heatmap is computed on the
original frame, so the explanation is faithful; it is then drawn over a redacted copy of the
same frame, which has identical geometry, so the overlay still lands where it should. Doing it
the other way round - explaining a pixelated frame - would produce an honest picture of a
model looking at pixelation. See :func:`redact_people`.

The redaction is not optional and not a flag. Frame images written by this module go into git
history, where they are permanent, and the data-release decision (WP1-T5) governs what may be
published from this dataset. Redacting by default costs nothing here and cannot be undone
later if it is skipped.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ExplainedFrame",
    "detect_objects",
    "detect_people",
    "evidence_on_people",
    "class_evidence_map",
    "evidence_outside",
    "probe_weights",
    "spatial_features",
    "attention_rollout",
    "redact_people",
    "pixelate_boxes",
    "overlay_heatmap",
    "SUPPORTS_ATTENTION",
]

#: Backbones with attention to roll out. ConvNeXtV2 is a convnet and has none, which is a
#: property of the architecture rather than a gap to fill.
SUPPORTS_ATTENTION: tuple[str, ...] = ("vit", "dinov2")


@dataclass(frozen=True, slots=True)
class ExplainedFrame:
    """One frame's explanation, with the numbers that let a reader check it.

    ``evidence`` is the per-position contribution map in *score units*, before any
    normalisation for display. ``score_from_map`` is its mean plus the constant term, and
    ``score_direct`` is what the probe itself returns. The two are printed together by
    `make_xai_figures.py` precisely so the "exact" claim in this module's docstring is
    checkable rather than asserted.
    """

    file: str
    backbone: str
    predicted: str
    explained_class: str
    evidence: np.ndarray
    score_from_map: float
    score_direct: float
    attention: np.ndarray | None = None

    @property
    def reconstruction_error(self) -> float:
        return abs(self.score_from_map - self.score_direct)


def probe_weights(probe, class_name: str) -> tuple[np.ndarray, float]:
    """The scaler-adjusted weight vector and constant term for one class.

    Returns ``(w / sigma, b - sum(w * mu / sigma))`` so a caller can dot the first against raw
    (unscaled) features and add the second. Unpicking the pipeline here rather than in the
    experiment keeps one place that knows a `LinearProbe` is a StandardScaler followed by a
    LogisticRegression - if that ever changes, this raises rather than silently explaining the
    wrong function.
    """
    model = getattr(probe, "_model", None)
    if model is None:
        raise RuntimeError("the probe was not fitted")
    try:
        scaler, logistic = model.named_steps["standardscaler"], model.named_steps[
            "logisticregression"
        ]
    except (AttributeError, KeyError) as exc:  # pragma: no cover - guards a refactor
        raise TypeError(
            "explain.py assumes a LinearProbe is StandardScaler -> LogisticRegression; the "
            "pipeline has changed, so the exact decomposition below no longer holds"
        ) from exc

    classes = [str(c) for c in logistic.classes_]
    if class_name not in classes:
        raise KeyError(f"{class_name!r} is not one of {', '.join(classes)}")
    index = classes.index(class_name)
    coef = logistic.coef_
    intercept = logistic.intercept_
    # Binary logistic regression stores one row meaning "the second class".
    if coef.shape[0] == 1:
        sign = 1.0 if index == 1 else -1.0
        w, b = sign * coef[0], sign * float(intercept[0])
    else:
        w, b = coef[index], float(intercept[index])

    sigma = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    adjusted = w / sigma
    return adjusted, b - float(np.sum(adjusted * scaler.mean_))


def spatial_features(model, processor, spec, image_bgr: np.ndarray, *,
                     processor_geometry: bool = True) -> tuple[np.ndarray, tuple[int, int]]:
    """Pre-pooling features as ``(n_positions, dim)``, with the grid shape they came from.

    The pooling that follows must match `backbones.embed_batch` exactly - a plain mean over
    positions - or the decomposition explains a different model from the one that produced the
    cached features. For transformers that mean includes the CLS token, so CLS is returned as
    position 0 and the caller drops it from the *picture* while keeping it in the arithmetic.
    """
    import torch
    from PIL import Image

    kwargs: dict[str, object] = {}
    if not processor_geometry:
        kwargs["do_resize"] = False
        if getattr(processor, "do_center_crop", False):
            kwargs["do_center_crop"] = False
    pil = Image.fromarray(image_bgr[:, :, ::-1])
    inputs = processor(images=[pil], return_tensors="pt", **kwargs)
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state

    if spec.kind == "convnet":
        _, channels, height, width = hidden.shape
        flat = hidden[0].reshape(channels, height * width).T
        return flat.cpu().numpy().astype(np.float64), (height, width)

    tokens = hidden[0].cpu().numpy().astype(np.float64)  # (1 + N, D), CLS first
    side = int(round((tokens.shape[0] - 1) ** 0.5))
    if side * side != tokens.shape[0] - 1:  # pragma: no cover - non-square patch grid
        raise ValueError(f"{tokens.shape[0] - 1} patch tokens is not a square grid")
    return tokens, (side, side)


def class_evidence_map(
    features: np.ndarray, weights: np.ndarray, constant: float, grid: tuple[int, int],
    *, drop_first: bool, cell_weights: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """The exact per-position contribution map, and the score it reconstructs.

    ``score = mean_p (weights . features_p) + constant``, so the map below is not an
    attribution *method* with choices in it - it is the summands of that mean.

    ``drop_first`` removes the CLS token from the returned picture. It stays in the score,
    because the model pooled it; dropping it from the arithmetic too would make the
    reconstruction check pass on a different quantity than the one the probe used.

    ``cell_weights`` is a ``grid``-shaped 0-1 array from `roi.grid_weights`, and it must be
    the *same* array `backbones.embed_batch` pooled with or this decomposes a model nobody
    ran. Positions outside the boundary get weight zero, so they contribute exactly zero to
    the score and appear as exactly zero on the map - which is the point: the picture can no
    longer show the model reading a pitch it was told to ignore, because it no longer is.

    The rescaling by ``size / total`` keeps the map in the units the unweighted path uses,
    where the *mean* over positions plus ``constant`` is the score. An all-inside boundary
    therefore reproduces the unweighted map element for element rather than approximately.
    """
    contributions = features @ weights
    if cell_weights is None:
        score = float(contributions.mean()) + constant
        spatial = contributions[1:] if drop_first else contributions
        return spatial.reshape(grid), score

    flat = np.asarray(cell_weights, dtype=np.float64).reshape(-1)
    if drop_first:  # CLS is not a position on the pitch and keeps full weight, as in pooling
        flat = np.concatenate([[1.0], flat])
    if flat.shape[0] != contributions.shape[0]:
        raise ValueError(
            f"{flat.shape[0]} cell weights for {contributions.shape[0]} positions - the "
            f"boundary was mapped onto a different grid than the features came from"
        )
    total = float(flat.sum())
    if total <= 1e-12:  # pragma: no cover - `roi.validate` refuses a boundary this small
        raise ValueError("this boundary covers no patch position at all")
    adjusted = contributions * flat * (flat.shape[0] / total)
    score = float(adjusted.mean()) + constant
    spatial = adjusted[1:] if drop_first else adjusted
    return spatial.reshape(grid), score


def evidence_outside(evidence: np.ndarray, cell_weights: np.ndarray | None) -> float | None:
    """Share of the positive evidence that fell outside the boundary. ``None`` without one.

    Reported rather than asserted. Under ROI pooling this is zero by construction, and a
    number on the page that a reader watches stay at 0.0% is worth more than a docstring
    promising it - especially here, where the complaint being answered is precisely that the
    evidence map appeared to show the model reading the neighbouring pitch.
    """
    if cell_weights is None:
        return None
    positive = np.clip(np.asarray(evidence, dtype=np.float64), 0.0, None)
    total = float(positive.sum())
    if total <= 0.0:
        return 0.0
    outside = np.asarray(cell_weights, dtype=np.float64).reshape(positive.shape) <= 0.0
    return float(positive[outside].sum()) / total


def load_for_attention(key: str):
    """Load a backbone that will actually return its attention weights.

    Transformers now defaults to SDPA, whose fused kernel does not materialise the attention
    matrix, so ``output_attentions=True`` is silently ignored and the model returns none. The
    first run of this module hit exactly that and produced no attention figure at all - a
    missing figure, not an error. Eager attention is slower and is only used here.
    """
    from transformers import AutoImageProcessor, AutoModel

    from pitch_occupancy.vision.backbones import BACKBONES

    if key not in SUPPORTS_ATTENTION:
        raise KeyError(f"{key!r} has no attention; known: {', '.join(SUPPORTS_ATTENTION)}")
    spec = BACKBONES[key]
    model = AutoModel.from_pretrained(spec.hf_id, attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, AutoImageProcessor.from_pretrained(spec.hf_id), spec


def attention_rollout(model, processor, image_bgr: np.ndarray, *,
                      discard_ratio: float = 0.0) -> np.ndarray | None:
    """Attention rollout over all layers, as a patch grid. ``None`` if the model has none.

    The standard formulation: average heads, add the identity to account for the residual
    stream, renormalise, and multiply the layers together. The CLS row of the product is how
    much each patch reached the classification token.

    Load the model with :func:`load_for_attention`. A model built the ordinary way uses SDPA
    and returns no attentions at all, and this returns ``None`` for that case as well as for
    a convnet - so the caller must distinguish "has no attention" from "was loaded wrongly",
    which is what :data:`SUPPORTS_ATTENTION` is for.
    """
    import torch
    from PIL import Image

    pil = Image.fromarray(image_bgr[:, :, ::-1])
    inputs = processor(images=[pil], return_tensors="pt")
    with torch.inference_mode():
        out = model(**inputs, output_attentions=True)
    attentions = getattr(out, "attentions", None)
    if not attentions:
        return None

    result = None
    for layer in attentions:
        a = layer[0].mean(dim=0)  # heads -> (T, T)
        if discard_ratio:
            flat = a.flatten()
            cutoff = int(flat.numel() * discard_ratio)
            if cutoff:
                threshold = flat.sort().values[cutoff]
                a = torch.where(a < threshold, torch.zeros_like(a), a)
        a = a + torch.eye(a.shape[0])  # the residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result

    mass = result[0, 1:].cpu().numpy().astype(np.float64)
    side = int(round(len(mass) ** 0.5))
    if side * side != len(mass):  # pragma: no cover
        return None
    return mass.reshape(side, side)


#: One detector per thread, built on first use.
#:
#: `detect_objects` used to call ``YOLO(model_name)`` on every frame. Measured on eight 1080p
#: frames at imgsz=1280: **292 ms per frame constructing, 152 ms reusing** - the construction
#: was costing more than the inference. At one frame per camera per minute across 20 cameras
#: that is 2.8 s a cycle spent loading the same weights twenty times.
#:
#: Thread-local rather than a plain module global because `evaluation/latency.py` runs the
#: pipeline on several threads to measure concurrency, and an ultralytics model is not
#: documented as safe to predict on from more than one. One model per thread costs one extra
#: load per thread and removes the question.
_DETECTORS = threading.local()


def _detector(model_name: str):
    """The loaded detector for this thread, or ``None`` if ultralytics is unavailable."""
    cache = getattr(_DETECTORS, "models", None)
    if cache is None:
        cache = _DETECTORS.models = {}
    if model_name not in cache:
        try:
            from ultralytics import YOLO
        except ImportError:  # pragma: no cover - dependency is pinned
            return None
        try:
            cache[model_name] = YOLO(model_name)
        except Exception:  # noqa: BLE001 - a missing weights file must fail closed
            return None
    return cache[model_name]


def detect_objects(image_bgr: np.ndarray, *, confidence: float = 0.25,
                   model_name: str = "yolov8n.pt", imgsz: int | None = None,
                   classes: Sequence[int] = (0,),
                   ) -> list[tuple[int, tuple[int, int, int, int], float]] | None:
    """``(class_id, box, confidence)`` per detection, or ``None`` if unavailable.

    The general form of :func:`detect_people`, added so a caller that wants people *and* the
    ball gets both from one forward pass. Two calls would double the second-per-frame the
    detector costs, and the ball measurement (A17) is only affordable because it is free -
    the detector was already running for the person count.
    """
    detector = _detector(model_name)
    if detector is None:
        return None

    try:
        kwargs = {"verbose": False, "conf": confidence, "classes": list(classes)}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        results = detector.predict(image_bgr, **kwargs)
    except Exception:  # noqa: BLE001 - any failure here must fail closed, not open
        return None

    height, width = image_bgr.shape[:2]
    out = []
    for result in results:
        boxes = result.boxes
        raw = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(raw, cls, conf, strict=True):
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), width), min(int(y2), height)
            if x2 > x1 and y2 > y1:
                out.append((int(c), (x1, y1, x2, y2), float(p)))
    return out


def detect_people(image_bgr: np.ndarray, *, confidence: float = 0.25,
                  model_name: str = "yolov8n.pt",
                  imgsz: int | None = None) -> list[tuple[int, int, int, int]] | None:
    """Person boxes as ``(x1, y1, x2, y2)``, or ``None`` if the detector is unavailable.

    ``None`` and ``[]`` mean different things and callers must treat them differently: the
    first is "not checked", the second is "checked, found none". Collapsing them is how a
    redaction step comes to report success on a frame it never examined.

    ``imgsz`` is the long edge the detector runs at. It was not exposed, so every caller got
    the default 640 - which downscales a 1080p CCTV frame until a player at the far end is a
    few pixels across, and is most of why this function was recorded as finding nobody on
    frames with people in them. `vision/people.py` passes 1280 and the counts there separate
    the classes.
    """
    found = detect_objects(image_bgr, confidence=confidence, model_name=model_name,
                           imgsz=imgsz, classes=(0,))
    if found is None:
        return None
    return [box for _cls, box, _conf in found]


def redact_people(image_bgr: np.ndarray, *, blocks: int = 16, confidence: float = 0.25,
                  model_name: str = "yolov8n.pt") -> tuple[np.ndarray, int]:
    """Pixelate every detected person. Returns the redacted frame and how many were found.

    **Zero detections does not mean zero people**, and the caller is told the count so it can
    say so rather than implying a frame was checked and found empty. A detector that misses a
    distant player produces an unredacted face; that is why `make_xai_figures.py` also applies
    a floor of blur to the whole frame rather than trusting this alone.

    Reports ``-1`` if the detector could not be loaded, so a missing model weight cannot
    silently produce unredacted output that looks redacted.
    """
    boxes = detect_people(image_bgr, confidence=confidence, model_name=model_name)
    if boxes is None:
        return image_bgr.copy(), -1
    return pixelate_boxes(image_bgr, boxes, blocks=blocks), len(boxes)


def pixelate_boxes(image_bgr: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
                   *, blocks: int = 16) -> np.ndarray:
    """Pixelate the given regions of a copy of the frame.

    Split out from :func:`redact_people` so a caller that has already run the detector can
    redact with *those* boxes rather than detecting a second time. Two detector runs on one
    frame can disagree, and a redaction that disagrees with the measurement taken beside it
    is a figure whose caption is about different pixels from the ones it shows.
    """
    out = image_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        out[y1:y2, x1:x2] = _pixelate(out[y1:y2, x1:x2], blocks)
    return out


def evidence_on_people(
    evidence: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> tuple[float, float]:
    """How much of the positive evidence falls on people, against how much area they occupy.

    Returns ``(evidence_fraction, area_fraction)``. Their ratio is the number that matters: a
    model reading *players* concentrates evidence far above the area they cover, and a ratio
    near 1 means the evidence is spread as though the people were not there.

    Only positive contributions are counted. The map is signed - positions can argue against
    the class - and mixing the two would let a strongly negative region cancel a strongly
    positive one and report the frame as uninformative when it is the opposite.
    """
    import cv2

    height, width = shape
    heat = cv2.resize(evidence.astype(np.float32), (width, height),
                      interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0.0, None)
    total = float(heat.sum())
    if total <= 0.0:
        return 0.0, 0.0

    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return float(heat[mask].sum()) / total, float(mask.sum()) / mask.size


def _pixelate(patch: np.ndarray, blocks: int) -> np.ndarray:
    import cv2

    h, w = patch.shape[:2]
    small = cv2.resize(patch, (max(1, min(blocks, w)), max(1, min(blocks, h))),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay_heatmap(image_bgr: np.ndarray, heat: np.ndarray, *, alpha: float = 0.45,
                    polygon: Sequence[Sequence[float]] | None = None) -> np.ndarray:
    """Draw a heatmap over a frame, normalised to its own range.

    Normalisation is per-map and deliberately so: these maps are in score units and their
    absolute scale differs between backbones, so a shared colour scale would compare
    magnitudes that are not comparable. The consequence is that colour shows *where*, never
    *how much*, and the figure caption has to say so.

    ``polygon`` confines the colour to the pitch boundary and draws the outline on top.

    **The clip is not cosmetic and it is not hiding anything.** A boundary that is pooled
    with, rather than merely filled, gives every outside position a contribution of exactly
    zero - but the colour map sends *zero* to whatever colour sits between the frame's
    minimum and maximum, so a frame with negative evidence anywhere inside paints the
    excluded region mid-scale, and the excluded region then reads as "somewhere the model
    looked". It is the unclipped picture that misleads. Normalisation is still taken over the
    whole map, so the colours inside are the same colours they would have been.

    The outline is drawn last, over the colour, because the one thing a reader has to be able
    to locate is where the model's attention was allowed to fall.
    """
    import cv2

    span = float(heat.max() - heat.min())
    normalised = (heat - heat.min()) / span if span > 1e-12 else np.zeros_like(heat)
    resized = cv2.resize(normalised.astype(np.float32),
                         (image_bgr.shape[1], image_bgr.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    coloured = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(coloured, alpha, image_bgr, 1 - alpha, 0)

    if polygon is None:
        return blended

    from pitch_occupancy.vision import roi

    height, width = image_bgr.shape[:2]
    points = np.array([[int(round(x * width)), int(round(y * height))] for x, y in polygon],
                      dtype=np.int32)
    inside = np.zeros((height, width), np.uint8)
    cv2.fillPoly(inside, [points], 255)
    blended = np.where(inside[:, :, None].astype(bool), blended, image_bgr)
    return roi.outline(blended, [list(pt) for pt in polygon])
