"""Re-score every searched configuration with the repaired false-play control (WP3-T8).

The search's control was measuring the probe on its own training data, so it read 0.0000 for
all 52 evaluations and discriminated between none of them. `false_play_rate` in
`preprocess_search.py` now uses genuinely held-out empty frames instead.

Every configuration's embeddings are already cached by `(backbone, config hash)`, so the
corrected control can be computed for all of them without a single new embedding pass - about
a second each, against sixteen minutes to recompute.

The question this answers is whether the search's ranking survives. If the top configurations
also have the highest false-play rates, they were winning by making the probe readier to say
PLAY on test sets that are 100% ACTIVE_PLAY, and the ranking is an artifact.

    uv run python experiments/rescore_false_play.py
"""

from __future__ import annotations

import csv
import json

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows

from experiments.preprocess_search import CACHE, OUT_JSON, false_play_rate

OUT = settings.results_dir / "false_play_rescored.csv"


def main() -> None:
    state = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    pos = {r.file: i for i, r in enumerate(rows)}

    rescored = []
    for entry in state["evaluations"]:
        path = CACHE / f"{entry['model']}_{entry['hash']}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        index = {str(f): i for i, f in enumerate(data["files"])}
        if not all(r.file in index for r in rows):
            continue
        X = data["features"][[index[r.file] for r in rows]]
        rescored.append({
            "model": entry["model"],
            "describe": entry["describe"],
            "play_recall": entry["play_recall"],
            "worst_fold": entry["worst_fold"],
            "false_play_old": entry["false_play"],
            "false_play_fixed": false_play_rate(rows, X, pos),
        })

    if not rescored:
        raise SystemExit("no cached configurations found to re-score")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rescored[0]))
        w.writeheader()
        w.writerows(rescored)

    for model in sorted({r["model"] for r in rescored}):
        group = [r for r in rescored if r["model"] == model]
        fp = np.array([r["false_play_fixed"] for r in group], dtype=float)
        print(f"\n=== {model}: {len(group)} configurations re-scored ===")
        print(f"old control: every value {set(r['false_play_old'] for r in group)}")
        print(f"fixed control: min {np.nanmin(fp):.4f}  median {np.nanmedian(fp):.4f}  "
              f"max {np.nanmax(fp):.4f}")

        ranked = sorted(group, key=lambda r: -r["play_recall"])
        print(f"\n{'recall':>8} {'worst':>7} {'false-play':>11}  configuration")
        for r in ranked[:8]:
            print(f"{r['play_recall']:8.4f} {r['worst_fold']:7.3f} "
                  f"{r['false_play_fixed']:11.4f}  {r['describe'][:52]}")

        # the question that matters: does high recall come with high false play?
        recall = np.array([r["play_recall"] for r in group])
        ok = ~np.isnan(fp)
        if ok.sum() > 3 and np.std(fp[ok]) > 1e-9:
            corr = float(np.corrcoef(recall[ok], fp[ok])[0, 1])
            print(f"\ncorrelation(recall, false-play) = {corr:+.3f}")
            print("  a strong positive value would mean the search was buying recall by")
            print("  shifting the boundary toward PLAY, and the ranking is an artifact")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
