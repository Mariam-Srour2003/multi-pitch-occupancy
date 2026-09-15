"""Frozen feature extractors (WP4).

Backbones are never fine-tuned. Each produces one pooled vector per image, a small head
is trained on those vectors, and the expensive part happens once into the feature cache.

**Mean pooling, always.** Hugging Face exposes ``pooler_output`` on several of these
models; on a plain ViT that head is *randomly initialised* and never trained. Reading it
cost the pilot a false 38% score and a silent production failure that looked like a model
problem for hours. Every extractor here pools explicitly:

* transformers  ``last_hidden_state.mean(dim=1)``   - over patch tokens
* convnets      ``last_hidden_state.mean(dim=(2,3))`` - global average pool

Every cached feature file carries ``pooling="mean"``; anything loading features asserts it.
If the pooling ever changes, bump :data:`POOLING_STAMP` so stale caches and heads are
rejected loudly instead of silently mixing conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Backbone", "BACKBONES", "POOLING_STAMP", "load_backbone", "embed_batch"]

POOLING_STAMP = "mean"


@dataclass(frozen=True, slots=True)
class Backbone:
    key: str  # short name used on the command line and in filenames
    hf_id: str
    kind: str  # "transformer" | "convnet" - decides the pooling axes
    note: str


BACKBONES: dict[str, Backbone] = {
    "convnextv2": Backbone(
        "convnextv2",
        "facebook/convnextv2-tiny-22k-224",
        "convnet",
        "production lead - smallest and fastest",
    ),
    "vit": Backbone(
        "vit",
        "google/vit-base-patch16-224",
        "transformer",
        "supervised ImageNet-21k->1k; accuracy reference",
    ),
    "dinov2": Backbone(
        "dinov2",
        "facebook/dinov2-base",
        "transformer",
        "self-supervised; robustness reference",
    ),
}


def load_backbone(key: str):
    """Return ``(model, processor, spec)`` with the model frozen and in eval mode."""
    from transformers import AutoImageProcessor, AutoModel

    if key not in BACKBONES:
        raise KeyError(f"unknown backbone {key!r}; known: {', '.join(sorted(BACKBONES))}")
    spec = BACKBONES[key]
    processor = AutoImageProcessor.from_pretrained(spec.hf_id)
    model = AutoModel.from_pretrained(spec.hf_id)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor, spec


#: What each model's Hugging Face processor does to an image *after* `preprocess.py` has
#: already produced a 224x224 letterboxed frame (WP3-T3 audit, 2026-09-07).
#:
#: **Two of the three re-crop the frame, and they are the two every experiment uses.** Both
#: upscale to 256 and centre-crop back to 224, cutting a 16-pixel border off every side and
#: discarding **23.4% of the frame area** - precisely where the letterbox padding of WP3-T2
#: lives. So the aspect-preserving resize is partly undone, and a centre crop nobody asked
#: for is applied on top of whatever the preprocessing search chose.
#:
#: They express it differently, which is why it hid. DINOv2 says so plainly, with
#: ``do_center_crop`` and ``crop_size``. ConvNeXtV2 says ``shortest_edge: 224`` and looks
#: like a no-op until you notice ``crop_pct: 0.875``: it resizes to 224/0.875 = 256 first,
#: then crops. Reading `size` alone gives the wrong answer for it.
#:
#: Measured on real frames, embeddings with and without that step have a cosine similarity
#: of 0.961 for DINOv2 and **0.886** for ConvNeXtV2 - differences far beyond numerical noise.
#: ViT is the only model that sees exactly what preprocessing produced.
PROCESSOR_GEOMETRY = {
    "convnextv2": "crop_pct 0.875: resize to 256, then centre-crop 224 - discards 23.4%",
    "vit": "resize to exactly 224x224 - a true no-op on a 224x224 input",
    "dinov2": "resize shortest edge 256, then centre-crop 224 - discards 23.4%",
}

#: Backbones whose processor leaves an already-224 frame untouched.
GEOMETRY_IS_A_NO_OP = ("vit",)


@torch.inference_mode()
def embed_batch(
    model, processor, spec: Backbone, images: list, *, processor_geometry: bool = True,
    roi_polygon: list[list[float]] | None = None,
) -> np.ndarray:
    """Embed a batch of PIL images into pooled feature vectors.

    Never touches ``pooler_output`` - see the module docstring.

    ``processor_geometry=False`` turns off the processor's own resize and centre crop, keeping
    only its rescale and per-model normalisation, so the frame the model sees is exactly the
    frame ``preprocess.py`` produced. That is the coherent choice for this project, where
    preprocessing is a searched variable and must therefore be what the model actually gets -
    see :data:`PROCESSOR_GEOMETRY`.

    It is **not** the default, for two reasons worth stating rather than assuming. The
    256-then-crop transform is DINOv2's canonical inference recipe, so departing from it is a
    decision that needs measuring, not a bug fix to be applied quietly. And flipping it
    silently would invalidate every cached DINOv2 feature and every result built on one. The
    flag is part of the cache fingerprint, so the two conventions can never mix.

    ``roi_polygon`` pools **only over the positions inside a pitch boundary** (WP3-T1). Without
    it, `roi.apply` blacks out the neighbouring pitch and the mean above averages the black in
    anyway - the fill reaches the probe and lights up the evidence map, which is the complaint
    a boundary was drawn to answer. With it, those positions leave the average, so what is
    outside the outline stops contributing rather than contributing a dark rectangle. Read
    :func:`_roi_weights` for what this does and does not achieve.

    **This is an inference-time choice and it is not what the cache was built with.** The
    feature cache pools over the whole frame, so a probe fitted on it and then fed an
    ROI-pooled vector is being asked about a slightly different distribution. That is the price
    of confining the model without re-extracting every feature, it is why :data:`POOLING_STAMP`
    is unchanged - no cached file is produced this way - and it is a difference that has to be
    measured on the development split rather than assumed small.
    """
    kwargs: dict[str, object] = {}
    if not processor_geometry:
        kwargs["do_resize"] = False
        if getattr(processor, "do_center_crop", False):
            kwargs["do_center_crop"] = False
    inputs = processor(images=images, return_tensors="pt", **kwargs)
    hidden = model(**inputs).last_hidden_state
    grid = _grid_of(spec, hidden)
    weights = _roi_weights(roi_polygon, images, grid, processor, processor_geometry)

    if spec.kind == "convnet":
        if weights is None:
            pooled = hidden.mean(dim=(2, 3))  # N,C,H,W -> N,C
        else:
            w = weights.reshape(weights.shape[0], 1, *grid)  # N,1,H,W
            pooled = (hidden * w).sum(dim=(2, 3)) / w.sum(dim=(2, 3)).clamp_min(_EPS)
    elif weights is None:
        pooled = hidden.mean(dim=1)  # N,T,D -> N,D
    else:
        # CLS keeps weight 1, so an all-inside boundary reproduces the plain mean exactly.
        # It is not a position on the pitch and cannot be masked out of the picture; see
        # the caveat in :func:`_roi_weights`.
        w = torch.cat([torch.ones(weights.shape[0], 1), weights], dim=1).unsqueeze(-1)
        pooled = (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(_EPS)
    return pooled.cpu().numpy().astype(np.float32)


#: Guards the degenerate boundary that rounds to no cells at all. `roi.validate` refuses a
#: polygon under 2% of the frame, which on ConvNeXtV2's 7x7 grid is still about one cell, so
#: this should be unreachable - it is here because dividing by it would produce confident
#: nonsense rather than an error.
_EPS = 1e-6


def _grid_of(spec: Backbone, hidden) -> tuple[int, int]:
    """The spatial grid the backbone's last hidden state is laid out on."""
    if spec.kind == "convnet":
        return int(hidden.shape[2]), int(hidden.shape[3])
    side = int(round((hidden.shape[1] - 1) ** 0.5))
    if side * side != hidden.shape[1] - 1:  # pragma: no cover - non-square patch grid
        raise ValueError(f"{hidden.shape[1] - 1} patch tokens is not a square grid")
    return side, side


def _roi_weights(polygon, images, grid, processor, processor_geometry):
    """Per-image, per-position pooling weights for a boundary, as ``(N, H*W)``. None if none.

    **This is what makes the fill stop mattering.** Without it a boundary changes the pixels
    outside the pitch and nothing else: the backbone still emits a token for every position
    and the mean above still averages them in, so black fill is not absence - it is a large,
    uniform, out-of-distribution region that the probe scores and the evidence map lights up.
    Weighting the pool by coverage removes those positions from the average, which is the
    difference between hiding the neighbouring pitch and not looking at it.

    **The honest caveat: this confines pooling, not attention.** A transformer's patch tokens
    attend to each other, so an inside token has already seen the fill by the time it is
    pooled, and a convnet's receptive field bleeds across the outline the same way. Excluding
    the outside positions removes their *direct* contribution - the dominant one, and the one
    the evidence map draws - and cannot remove the indirect one. CLS is the clearest case:
    it is a global summary of the whole frame and it keeps full weight, because dropping it
    would change the pooling for reasons that have nothing to do with the boundary. Filling
    with ``blur`` or ``mean`` rather than ``black`` is what limits the indirect path, which is
    the empirical question `roi.FILLS` exists to let someone answer.
    """
    if not polygon:
        return None
    from pitch_occupancy.vision import roi

    rows = []
    for image in images:
        size = getattr(image, "size", None) or (image.shape[1], image.shape[0])
        cells = roi.grid_weights(polygon, size, grid, processor,
                                 processor_geometry=processor_geometry)
        rows.append(cells.reshape(-1))
    return torch.from_numpy(np.stack(rows).astype(np.float32))
