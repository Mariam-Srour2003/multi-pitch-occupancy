"""Embed every frame once per backbone and reuse the vectors forever (WP0-T5).

Embedding 1,692 frames on CPU takes minutes; training a logistic-regression head on the
cached vectors takes under a second. Every downstream experiment - label-efficiency
curves, split comparisons, ablations, calibration - is therefore cheap, but only if the
cache is trustworthy. Two things make it trustworthy:

1. **A preprocessing fingerprint.** The cache records the backbone id, the pooling
   convention and a hash of the preprocessing settings. Loading with different settings
   is refused rather than silently returning vectors built under the old ones.
2. **Frames are addressed by path.** Features are returned in the order the caller's
   manifest rows specify, so a cache built before new frames were added still serves the
   rows it knows and reports exactly which are missing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.vision.backbones import BACKBONES, POOLING_STAMP

__all__ = [
    "CachedFeatures", "preprocessing_hash", "cache_path", "build_cache", "load_cache",
    "features_for",
]


@dataclass(frozen=True, slots=True)
class CachedFeatures:
    backbone: str
    pooling: str
    preproc_hash: str
    files: np.ndarray  # (n,) of str
    features: np.ndarray  # (n, d) float32

    def __len__(self) -> int:
        return len(self.files)

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def index(self) -> dict[str, int]:
        return {str(f): i for i, f in enumerate(self.files)}


def preprocessing_hash(**settings: object) -> str:
    """Stable fingerprint of the preprocessing that produced a set of features."""
    payload = "|".join(f"{k}={settings[k]!r}" for k in sorted(settings))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path(
    backbone: str, cache_dir: Path, *, processor_geometry: bool = True
) -> Path:
    """Where a cache lives. The non-default geometry convention gets its own file.

    Both conventions have to be able to exist at once, because settling which one to use
    (WP3-T3) means comparing them - and one filename for two conventions would have meant
    the second build silently overwriting the first.
    """
    suffix = "" if processor_geometry else "_nogeom"
    return cache_dir / f"{backbone}{suffix}.npz"


def build_cache(
    rows: list[ManifestRow],
    backbone: str,
    dataset_dir: Path,
    cache_dir: Path,
    *,
    batch_size: int = 16,
    preproc: dict[str, object] | None = None,
    progress: bool = True,
    processor_geometry: bool = True,
    preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CachedFeatures:
    """Embed every frame in ``rows`` and write the cache to disk.

    ``processor_geometry=False`` disables the HF processor's own resize and centre crop,
    keeping only its rescale and normalisation, so the model sees exactly what
    ``preprocess.py`` produced. WP3-T3 found that ConvNeXtV2 and DINOv2 otherwise resize to
    256 and crop back to 224 *after* preprocessing has run, discarding 23.4% of the frame.

    **This parameter did not exist**, so the alternative convention could not be cached at
    all - and the flag was not in the fingerprint either, though `thesis/protocol.md` and
    `EXPERIMENT_LOG.md` both stated that it was. Two conventions would have collided on one
    cache key *and* one filename.

    ``preprocess_fn`` is **required** when ``processor_geometry=False``, and the reason is
    the finding that came out of trying to use the flag:

    **This function does not apply ``preprocess.py``.** It opens raw frames and hands them
    to the HF processor, whose resize is what makes them model-sized. ``preproc`` is a dict
    of *labels for the fingerprint* - it has never driven a transform. So the letterbox of
    WP3-T2, ROI masking, CLAHE and the rest of the ten searched switches are **not in the
    path that produced any cached feature the headline experiments read**; the search and
    ablation caches are separate because those scripts call ``preprocess`` themselves.

    Turning the processor's geometry off therefore leaves nothing to resize the frame, and
    the model rejects a 1080x1920 input outright. Rather than crash deep inside
    transformers, this asks for the preprocessing step explicitly - which also makes the
    dependency legible: the alternative convention is only meaningful *with* a
    preprocessing path, and the default one currently has none.
    """
    from PIL import Image

    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    if not processor_geometry and preprocess_fn is None:
        raise ValueError(
            "processor_geometry=False needs preprocess_fn. build_cache feeds *raw* frames "
            "to the processor, whose resize is the only thing making them model-sized - "
            "preprocess.py is not in this path. With the processor's geometry off and no "
            "preprocessing, the model receives a full-resolution frame and refuses it. "
            "Pass the preprocessing callable that produces a model-sized frame, and note "
            "in the run that this cache is not comparable with the default ones."
        )

    preproc = preproc or {"roi": False, "resize": "processor_default"}
    model, processor, spec = load_backbone(backbone)

    files = [r.file for r in rows]
    chunks: list[np.ndarray] = []
    for start in range(0, len(files), batch_size):
        batch = files[start : start + batch_size]
        images = [Image.open(dataset_dir / f).convert("RGB") for f in batch]
        if preprocess_fn is not None:
            images = [
                Image.fromarray(preprocess_fn(np.asarray(im)[:, :, ::-1])[:, :, ::-1])
                for im in images
            ]
        chunks.append(
            embed_batch(
                model, processor, spec, images, processor_geometry=processor_geometry
            )
        )
        if progress:
            done = min(start + batch_size, len(files))
            print(f"  {backbone}: {done}/{len(files)}", end="\r", flush=True)
    if progress:
        print()

    feats = np.concatenate(chunks) if chunks else np.zeros((0, 0), np.float32)
    cached = CachedFeatures(
        backbone=backbone,
        pooling=POOLING_STAMP,
        preproc_hash=preprocessing_hash(
            backbone=BACKBONES[backbone].hf_id,
            processor_geometry=processor_geometry,
            **preproc,
        ),
        files=np.array(files, dtype=object),
        features=feats,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(backbone, cache_dir, processor_geometry=processor_geometry),
        files=cached.files,
        features=cached.features,
        backbone=cached.backbone,
        pooling=cached.pooling,
        preproc_hash=cached.preproc_hash,
    )
    return cached


def load_cache(
    backbone: str,
    cache_dir: Path,
    *,
    expect_preproc_hash: str | None = None,
    processor_geometry: bool = True,
) -> CachedFeatures:
    """Load a cache, refusing anything built under a different convention."""
    path = cache_path(backbone, cache_dir, processor_geometry=processor_geometry)
    if not path.exists():
        raise FileNotFoundError(f"no feature cache for {backbone!r} at {path}")
    z = np.load(path, allow_pickle=True)
    pooling = str(z["pooling"])
    if pooling != POOLING_STAMP:
        raise ValueError(
            f"{path.name} was built with pooling={pooling!r}, expected "
            f"{POOLING_STAMP!r}. Rebuild the cache and retrain every head that used it - "
            f"mixing pooling conventions produces plausible, wrong numbers."
        )
    got = str(z["preproc_hash"])
    if expect_preproc_hash is not None and got != expect_preproc_hash:
        raise ValueError(
            f"{path.name} was built under preprocessing {got}, but {expect_preproc_hash} "
            f"was requested. Rebuild rather than compare across preprocessing."
        )
    return CachedFeatures(
        backbone=str(z["backbone"]),
        pooling=pooling,
        preproc_hash=got,
        files=z["files"],
        features=z["features"],
    )


def features_for(
    cached: CachedFeatures, rows: list[ManifestRow]
) -> tuple[np.ndarray, list[ManifestRow]]:
    """Features aligned to ``rows``, plus the rows actually covered.

    Missing frames are dropped and reported by the caller rather than silently
    zero-filled: a head trained on zero vectors would still converge to something.
    """
    idx = cached.index()
    keep = [r for r in rows if r.file in idx]
    if not keep:
        return np.zeros((0, cached.dim), np.float32), []
    return cached.features[[idx[r.file] for r in keep]], keep
