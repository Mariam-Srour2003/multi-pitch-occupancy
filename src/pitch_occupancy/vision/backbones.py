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


@torch.inference_mode()
def embed_batch(model, processor, spec: Backbone, images: list) -> np.ndarray:
    """Embed a batch of PIL images into pooled feature vectors.

    Never touches ``pooler_output`` - see the module docstring.
    """
    inputs = processor(images=images, return_tensors="pt")
    hidden = model(**inputs).last_hidden_state
    if spec.kind == "convnet":
        pooled = hidden.mean(dim=(2, 3))  # N,C,H,W -> N,C
    else:
        pooled = hidden.mean(dim=1)  # N,T,D -> N,D
    return pooled.cpu().numpy().astype(np.float32)
