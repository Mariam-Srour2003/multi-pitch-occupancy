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
    model, processor, spec: Backbone, images: list, *, processor_geometry: bool = True
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
    """
    kwargs: dict[str, object] = {}
    if not processor_geometry:
        kwargs["do_resize"] = False
        if getattr(processor, "do_center_crop", False):
            kwargs["do_center_crop"] = False
    inputs = processor(images=images, return_tensors="pt", **kwargs)
    hidden = model(**inputs).last_hidden_state
    if spec.kind == "convnet":
        pooled = hidden.mean(dim=(2, 3))  # N,C,H,W -> N,C
    else:
        pooled = hidden.mean(dim=1)  # N,T,D -> N,D
    return pooled.cpu().numpy().astype(np.float32)
