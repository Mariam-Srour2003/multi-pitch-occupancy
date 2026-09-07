"""WP3-T3 option (a): does the processor's geometry actually cost anything? (RQ1, RQ3)

`thesis/protocol.md` says the convention question is settled by *"one cross-venue run per
model under each convention"*. Setting that up found something first: `build_cache` hands
**raw frames** to the HF processor and never applies `preprocess.py`, so the searched
switches - letterbox, ROI, CLAHE - are absent from every cached feature the headline
experiments read. Answering the convention question honestly would mean putting
`preprocess.py` into the main path, which changes the input to *every* published number.

That is a decision, not a run, so this is the cheap version that informs it: a
**self-contained probe** that touches no existing cache and compares three arms.

| arm | what the model sees |
|---|---|
| `raw+geom` | raw frame -> processor resizes (and, for ConvNeXtV2/DINOv2, crops 224 from 256). **The published baseline.** |
| `preproc+geom` | `preprocess.py` -> 224x224 letterboxed -> processor resizes to 256 and crops back. The double transform WP3-T3 found. |
| `preproc+nogeom` | `preprocess.py` -> 224x224 letterboxed -> processor only rescales and normalises. The coherent single path. |

Comparing `preproc+geom` against `preproc+nogeom` answers the convention question. Comparing
either against `raw+geom` says what adopting `preprocess.py` at all would cost or buy - which
is the more expensive decision and the one worth having a number for.

**Both axes, never recall alone.** Cross-venue folds are 100% ACTIVE_PLAY, so recall rises by
answering "playing" more often. The second column is false-play on held-out empty frames.

**Its caches live in `data/cache/geom_probe/` on purpose.** Under the default naming an
arm with `processor_geometry=True` would be written as `<backbone>.npz` and overwrite the
main cache every published number depends on.

    uv run python experiments/geometry_convention_probe.py

Roughly 8 minutes per backbone per new arm on this machine; the `raw+geom` arm is free
because it reuses the existing cache.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.feature_cache import build_cache, cache_path, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.heads import LinearProbe
from pitch_occupancy.vision.preprocess import PreprocessConfig, preprocess

PLAY, EMPTY = "C2_ACTIVE_PLAY", "C1_EMPTY"
SEED = 42
BACKBONES = ("convnextv2", "dinov2", "vit")
OUT = settings.results_dir / "geometry_convention_probe.csv"
PROBE_CACHE = settings.feature_cache_dir / "geom_probe"

#: The default config: aspect-preserving resize to 224 with grey padding (WP3-T2), nothing
#: photometric. Deliberately not a searched configuration - the question here is what the
#: *geometry* convention does, and adding switches would confound the two.
CFG = PreprocessConfig()

#: Labels recorded in the fingerprint so the probe's caches can never be mistaken for the
#: main ones, which were built from raw frames.
PROBE_PREPROC = {"roi": False, "resize": "preprocess.py letterbox 224"}


def _preprocess_bgr(bgr: np.ndarray) -> np.ndarray:
    return preprocess(bgr, CFG)


def features(backbone: str, arm: str, rows) -> tuple[list, np.ndarray]:
    """Features for one arm, building the probe caches on first use."""
    if arm == "raw+geom":
        cached = load_cache(backbone, settings.feature_cache_dir)
    else:
        geom = arm == "preproc+geom"
        path = cache_path(backbone, PROBE_CACHE, processor_geometry=geom)
        if not path.exists():
            print(f"  building {path.name} ({arm}) - this is the slow part", flush=True)
            build_cache(
                rows, backbone, settings.dataset_dir, PROBE_CACHE,
                preproc=PROBE_PREPROC,
                processor_geometry=geom,
                preprocess_fn=_preprocess_bgr,
                progress=True,
            )
        cached = load_cache(backbone, PROBE_CACHE, processor_geometry=geom)

    idx = cached.index()
    kept = [r for r in rows if r.file in idx]
    if len(kept) != len(rows):
        print(f"  ({len(rows) - len(kept)} row(s) uncached - dropped, not zero-filled)")
    return kept, cached.features[[idx[r.file] for r in kept]]


def cross_venue(rows, X) -> dict[str, float]:
    pos = {r.file: i for i, r in enumerate(rows)}
    out: dict[str, float] = {}
    for fold in leave_one_group_out(rows):
        play = [r for r in fold.test if r.class3 == PLAY]
        if not play or len({r.class3 for r in fold.train}) < 2:
            continue  # the venue_01 fold trains on a single class; 7 folds, not 8
        head = LinearProbe("probe", seed=SEED).fit(
            X[[pos[r.file] for r in fold.train]], fold.train
        )
        pred = head.predict(X[[pos[r.file] for r in play]], play)
        out[fold.name] = sum(p == PLAY for p in pred) / len(play)
    return out


def false_play(rows, X) -> tuple[float, int]:
    """Trained on venue_01 camera A, scored on camera B's empty frames - as in H3."""
    pos = {r.file: i for i, r in enumerate(rows)}
    cam = lambda r: PHYSICAL_CAMERA.get(r.camera, r.camera)  # noqa: E731
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if cam(r) == "camera_A"]
    empties = [r for r in venue if cam(r) == "camera_B" and r.class3 == EMPTY]
    if not empties or len({r.class3 for r in train}) < 2:
        return float("nan"), 0
    head = LinearProbe("probe", seed=SEED).fit(X[[pos[r.file] for r in train]], train)
    pred = head.predict(X[[pos[r.file] for r in empties]], empties)
    return float(np.mean([p == PLAY for p in pred])), len(empties)


def main() -> None:
    all_rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    print(f"development rows: {len(all_rows)}")
    print(f"probe caches in: {PROBE_CACHE}\n")

    arms = ("raw+geom", "preproc+geom", "preproc+nogeom")
    records: list[dict] = []

    for backbone in BACKBONES:
        print(f"=== {backbone} ===")
        for arm in arms:
            rows, X = features(backbone, arm, all_rows)
            folds = cross_venue(rows, X)
            mean = float(np.mean(list(folds.values())))
            worst = min(folds.values())
            fp, n_empty = false_play(rows, X)
            print(f"  {arm:<16} recall {mean:.4f}  worst {worst:.3f}  "
                  f"false-play {fp:.4f}  (n_empty={n_empty})", flush=True)
            for name, r in folds.items():
                records.append({
                    "backbone": backbone, "arm": arm, "held_out_venue": name,
                    "play_recall": f"{r:.4f}", "false_play_rate": "", "n_held_out_empty": "",
                })
            records.append({
                "backbone": backbone, "arm": arm, "held_out_venue": "MEAN_ACROSS_FOLDS",
                "play_recall": f"{mean:.4f}", "false_play_rate": f"{fp:.4f}",
                "n_held_out_empty": n_empty,
            })
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"wrote {OUT.name}")

    means = {
        (r["backbone"], r["arm"]): (float(r["play_recall"]), float(r["false_play_rate"]))
        for r in records if r["held_out_venue"] == "MEAN_ACROSS_FOLDS"
    }
    print("\n=== the convention question: preproc+geom vs preproc+nogeom ===")
    for b in BACKBONES:
        g, ng = means[(b, "preproc+geom")], means[(b, "preproc+nogeom")]
        print(f"  {b:<12} recall {ng[0] - g[0]:+.4f}   false-play {ng[1] - g[1]:+.4f}"
              f"   (negative false-play is better)")
    print("\n=== the bigger question: does preprocess.py help at all? ===")
    for b in BACKBONES:
        raw, best = means[(b, "raw+geom")], means[(b, "preproc+nogeom")]
        print(f"  {b:<12} recall {best[0] - raw[0]:+.4f}   false-play {best[1] - raw[1]:+.4f}"
              f"   (preproc+nogeom vs the published raw+geom)")

    with (settings.results_dir / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} | WP3-T3 geometry probe | "
            f"`python experiments/geometry_convention_probe.py` | `{OUT.name}` | "
            f"three arms x {len(BACKBONES)} backbones, both axes\n"
        )


if __name__ == "__main__":
    main()
