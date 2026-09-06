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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.vision.backbones import BACKBONES, POOLING_STAMP

__all__ = ["CachedFeatures", "preprocessing_hash", "build_cache", "load_cache", "features_for"]


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


def cache_path(backbone: str, cache_dir: Path) -> Path:
    return cache_dir / f"{backbone}.npz"


def build_cache(
    rows: list[ManifestRow],
    backbone: str,
    dataset_dir: Path,
    cache_dir: Path,
    *,
    batch_size: int = 16,
    preproc: dict[str, object] | None = None,
    progress: bool = True,
) -> CachedFeatures:
    """Embed every frame in ``rows`` and write the cache to disk."""
    from PIL import Image

    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    preproc = preproc or {"roi": False, "resize": "processor_default"}
    model, processor, spec = load_backbone(backbone)

    files = [r.file for r in rows]
    chunks: list[np.ndarray] = []
    for start in range(0, len(files), batch_size):
        batch = files[start : start + batch_size]
        images = [Image.open(dataset_dir / f).convert("RGB") for f in batch]
        chunks.append(embed_batch(model, processor, spec, images))
        if progress:
            done = min(start + batch_size, len(files))
            print(f"  {backbone}: {done}/{len(files)}", end="\r", flush=True)
    if progress:
        print()

    feats = np.concatenate(chunks) if chunks else np.zeros((0, 0), np.float32)
    cached = CachedFeatures(
        backbone=backbone,
        pooling=POOLING_STAMP,
        preproc_hash=preprocessing_hash(backbone=BACKBONES[backbone].hf_id, **preproc),
        files=np.array(files, dtype=object),
        features=feats,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(backbone, cache_dir),
        files=cached.files,
        features=cached.features,
        backbone=cached.backbone,
        pooling=cached.pooling,
        preproc_hash=cached.preproc_hash,
    )
    return cached


def load_cache(
    backbone: str, cache_dir: Path, *, expect_preproc_hash: str | None = None
) -> CachedFeatures:
    """Load a cache, refusing anything built under a different convention."""
    path = cache_path(backbone, cache_dir)
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
