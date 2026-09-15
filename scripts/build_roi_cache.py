"""Build a feature cache pooled inside each camera's derived pitch boundary.

`build_cache` gained `roi_for` so that training features can be produced the way
`classifier.classify_batch` produces them at serve time - pooling only over positions inside
the boundary. This uses the boundaries derived by `scripts/derive_roi.py`, which exist for
every camera in the corpus rather than for the two that `configs/roi.json` names.

    uv run python scripts/build_roi_cache.py --backbone dinov2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pitch_occupancy.data.feature_cache import build_cache, cache_path
from pitch_occupancy.data.manifest import read_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--roi", type=Path, default=ROOT / "configs" / "roi_derived.json")
    args = ap.parse_args()

    polys = {k: v for k, v in json.loads(args.roi.read_text(encoding="utf-8")).items()
             if not k.startswith("_")}
    rows = read_manifest(DATASET / "manifest.csv")

    missing = sorted({r.camera for r in rows} - set(polys))
    if missing:
        print(f"{len(missing)} camera(s) without a boundary; they pool unbounded: "
              f"{missing[:5]}")

    out = cache_path(args.backbone, CACHE, roi_pooled=True)
    print(f"{len(rows)} frames | {len(polys)} boundaries -> {out.name}")
    build_cache(rows, args.backbone, DATASET, CACHE,
                roi_for=lambda r: polys.get(r.camera))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
