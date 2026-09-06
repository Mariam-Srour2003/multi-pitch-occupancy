"""What are the models actually looking at? (WP3-T8, WP4-T6, RQ3/RQ7)

H2 showed a clock rule matching two of three backbones inside the confounded venue; H3
showed the same rule collapsing across venues while the backbones held. Both are indirect.
This asks the question directly by removing information from the input and seeing what
survives:

| variant | what it removes | what it leaves |
|---|---|---|
| `full` | nothing | the baseline |
| `grayscale` | colour - turf hue, floodlight cast, kit colours | shape, texture, people |
| `blur8` | fine detail; people become smudges | scene layout, lighting, geometry |
| `blur16` | almost all object detail | the gross composition of the scene |
| `crop50` | the outer border - stands, sky, adjacent pitches | the central pitch area |

The logic is a dissociation. **If accuracy survives `blur16`, the model cannot be
recognising people** - at that scale nobody is visible - so it must be reading the scene.
If it collapses, the prediction depended on the objects in the frame.

`crop50` is the complement: a crude stand-in for the ROI masking of WP3-T1, which needs
hand-drawn polygons for all nine venues. If discarding the border hurts, the border was
carrying the signal.

Evaluated on the held-out *venues* (H3's protocol, ACTIVE_PLAY recall), because the
in-venue grouped split is too degenerate to read anything from.

    uv run python experiments/input_ablation.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.vision.backbones import embed_batch, load_backbone
from pitch_occupancy.vision.heads import LinearProbe
from pitch_occupancy.vision.preprocess import PreprocessConfig, preprocess

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
SEED = 42
PLAY = "C2_ACTIVE_PLAY"
BACKBONE = "dinov2"  # the strongest generaliser from H3

VARIANTS = {
    "full": PreprocessConfig(),
    "grayscale": PreprocessConfig(grayscale=True),
    "blur8": PreprocessConfig(blur_sigma=8.0),
    "blur16": PreprocessConfig(blur_sigma=16.0),
    "crop50": PreprocessConfig(centre_crop=0.5),
}


def embed_variant(rows, name: str, cfg: PreprocessConfig) -> np.ndarray:
    """Embed every frame under one preprocessing variant, cached to disk."""
    path = CACHE / f"ablate_{BACKBONE}_{name}.npz"
    files = [r.file for r in rows]
    if path.exists():
        z = np.load(path, allow_pickle=True)
        idx = {str(f): i for i, f in enumerate(z["files"])}
        if all(f in idx for f in files):
            return z["features"][[idx[f] for f in files]]

    model, processor, spec = load_backbone(BACKBONE)
    chunks = []
    for start in range(0, len(files), 24):
        batch = files[start : start + 24]
        images = []
        for f in batch:
            bgr = cv2.imread(str(DATASET / f))
            images.append(Image.fromarray(cv2.cvtColor(preprocess(bgr, cfg), cv2.COLOR_BGR2RGB)))
        chunks.append(embed_batch(model, processor, spec, images))
        print(f"    {name}: {min(start + 24, len(files))}/{len(files)}", end="\r", flush=True)
    print()
    feats = np.concatenate(chunks)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, files=np.array(files, dtype=object), features=feats)
    return feats


def main() -> None:
    rows = development_rows(read_manifest(DATASET / "manifest.csv"))
    pos = {r.file: i for i, r in enumerate(rows)}
    folds = [
        f for f in leave_one_group_out(rows, group_key="venue")
        if not f.name.endswith("venue_01")
    ]
    print(f"{len(rows)} rows | {len(folds)} held-out venue folds | backbone {BACKBONE}\n")

    records, summary = [], {}
    for name, cfg in VARIANTS.items():
        print(f"=== {name} ===")
        X = embed_variant(rows, name, cfg)
        recalls = []
        for fold in folds:
            venue = fold.name.split("__")[-1]
            probe = LinearProbe(name, seed=SEED).fit(
                X[[pos[r.file] for r in fold.train]], fold.train
            )
            pred = probe.predict(X[[pos[r.file] for r in fold.test]], fold.test)
            truth = [r.class3 for r in fold.test]
            play = [(p, t) for p, t in zip(pred, truth, strict=True) if t == PLAY]
            recall = sum(p == PLAY for p, _ in play) / len(play) if play else float("nan")
            recalls.append(recall)
            records.append(
                {"variant": name, "venue": venue, "n_test": len(fold.test),
                 "play_recall": round(recall, 4)}
            )
        arr = np.array(recalls)
        summary[name] = (float(arr.mean()), float(arr.min()))
        print(f"  mean play-recall {arr.mean():.3f}   worst {arr.min():.3f}\n")

    base = summary["full"][0]
    print("=== change from the unmodified input ===")
    for name, (mean, worst) in summary.items():
        if name == "full":
            continue
        delta = mean - base
        verdict = "signal survives" if delta > -0.05 else "signal lost"
        print(f"  {name:<10} {mean:.3f}  ({delta:+.3f})  {verdict}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "input_ablation.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "venue", "n_test", "play_recall"])
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")

    with (RESULTS / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} | input ablation | "
            f"`python experiments/input_ablation.py` | `{out.name}` | "
            f"{len(VARIANTS)} variants x {len(folds)} folds, {BACKBONE}\n"
        )


if __name__ == "__main__":
    main()
