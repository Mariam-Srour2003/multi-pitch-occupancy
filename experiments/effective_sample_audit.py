"""How many independent observations are the reported tests actually based on? (WP4-T4b)

The false-play significance work found that 243 held-out empty frames amount to three to ten
*distinct scenes*, and that once counted properly not one of six pairwise comparisons
survived. That was not a quirk of one experiment. Every frame-level test in this project runs
on footage sampled every fifteen seconds from fixed cameras, and the near-duplicate audit
found 98.5% of frames have a direct near-duplicate.

So this audits the reported evaluation sets rather than assuming them independent.

**H3 is not in scope, and that is a point in its favour.** Its confidence intervals bootstrap
over the seven *venue folds*, not over frames — seven genuinely different facilities are seven
observations, and the interval is wide because of it. H1 and H2 are different: they use
`bootstrap_metric_ci` and frame-level McNemar over a 394-frame test set, so every
near-duplicate in that set is counted as an independent observation.

For each split protocol this reports the nominal test-set size, the number of distinct scenes
in it, and whether the published comparison survives being recomputed on one frame per scene.

    uv run python experiments/effective_sample_audit.py
"""

from __future__ import annotations

import csv

import cv2
import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import DEFAULT_THRESHOLD, dhash, distinct_subset
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, grouped_split, random_split
from pitch_occupancy.evaluation.stats import holm_bonferroni, mcnemar
from pitch_occupancy.vision.heads import ClockRule, LinearProbe

SEED = 42
OUT = settings.results_dir / "effective_sample_audit.csv"
CACHE = "dinov2.npz"
CHEAP = "cheap_histogram.npz"


def distinct_scenes(rows, *, threshold: int = DEFAULT_THRESHOLD) -> list[int]:
    """Indices of a greedy maximal set of pairwise-distinct frames.

    The selection rule now lives in :func:`pitch_occupancy.data.dedup.distinct_subset`,
    because `logit_average_baseline.py` and this script both need it and two greedy loops
    meant to agree are two that can drift apart. This wrapper keeps the I/O - reading the
    frames and hashing them - which is the part that is specific to a script.
    """
    hashes = {}
    for i, row in enumerate(rows):
        image = cv2.imread(str(settings.dataset_dir / row.file))
        if image is not None:
            hashes[i] = dhash(image)
    return distinct_subset(hashes, threshold=threshold)


def correctness(model: str, train, test, features, pos) -> np.ndarray:
    head = ClockRule() if model == "clock_rule" else LinearProbe(model, seed=SEED)
    head.fit(features[[pos[r.file] for r in train]], train)
    pred = head.predict(features[[pos[r.file] for r in test]], test)
    return np.array([p == r.class3 for p, r in zip(pred, test, strict=True)])


def published_accuracies() -> dict[tuple[str, str], float]:
    """The accuracies H1/H2 reported, keyed by (split prefix, model)."""
    path = settings.results_dir / "h1_h2_baseline_floor.csv"
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = "random" if row["split"].startswith("random") else "grouped"
        out[(key, row["model"])] = float(row["accuracy"])
    return out


def main() -> None:
    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    data = np.load(settings.feature_cache_dir / CACHE, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(data["files"])}
    rows = [r for r in rows if r.file in index]
    deep = data["features"][[index[r.file] for r in rows]]
    pos = {r.file: i for i, r in enumerate(rows)}

    cheap_path = settings.feature_cache_dir / CHEAP
    cheap = None
    if cheap_path.exists():
        cd = np.load(cheap_path, allow_pickle=True)
        ci = {str(f): i for i, f in enumerate(cd["files"])}
        if all(r.file in ci for r in rows):
            cheap = cd["features"][[ci[r.file] for r in rows]]

    records = []
    for name, split in (("random (leaky)", random_split(rows)),
                        ("grouped", grouped_split(rows))):
        test = list(split.test)
        keep = distinct_scenes(test)
        print(f"\n=== {name} split ===")
        print(f"  nominal test frames : {len(test)}")
        print(f"  distinct scenes     : {len(keep)}  "
              f"({len(keep) / len(test):.1%} of the nominal size)")

        models = {"dinov2": deep}
        if cheap is not None:
            models["colour_histogram"] = cheap
        models["clock_rule"] = np.zeros((len(rows), 1), dtype=np.float32)

        ok = {
            m: correctness(m if m != "colour_histogram" else "hist",
                           list(split.train), test, X, pos)
            for m, X in models.items()
        }
        # Reproduce before extending, the same rule the H3 harness follows: a number that
        # cannot re-derive the published one has no standing to qualify it.
        reference = published_accuracies()
        key = "random" if name.startswith("random") else "grouped"
        for m, c in ok.items():
            want = reference.get((key, {"colour_histogram": "cheap_histogram"}.get(m, m)))
            agree = want is None or abs(c.mean() - want) < 5e-4
            mark = "" if want is None else ("  == published" if agree else "  != PUBLISHED")
            print(f"    {m:18} accuracy {c.mean():.4f} on all frames, "
                  f"{c[keep].mean():.4f} on distinct scenes{mark}")
            if not agree:
                raise SystemExit(
                    f"{m} on the {key} split gives {c.mean():.4f} against a published "
                    f"{want:.4f}; fix that before reading anything below it"
                )

        pairs = [(a, b) for i, a in enumerate(ok) for b in list(ok)[i + 1:]]
        full = [mcnemar(ok[a], ok[b]).p_value for a, b in pairs]
        _, rej_full = holm_bonferroni(full)
        reduced = [mcnemar(ok[a][keep], ok[b][keep]).p_value for a, b in pairs]
        _, rej_red = holm_bonferroni(reduced)

        print(f"  significant comparisons: {sum(rej_full)}/{len(pairs)} on all frames, "
              f"{sum(rej_red)}/{len(pairs)} on distinct scenes")
        for (a, b), pf, rf, pr, rr in zip(pairs, full, rej_full, reduced, rej_red,
                                          strict=True):
            print(f"    {a + ' vs ' + b:34} p {pf:9.2e} -> {pr:9.2e}  "
                  f"{'yes' if rf else 'no':>4} -> {'yes' if rr else 'no':>4}")
            records.append({
                "split": name, "a": a, "b": b,
                "n_frames": len(test), "n_distinct": len(keep),
                "p_all": f"{pf:.3e}", "sig_all": rf,
                "p_distinct": f"{pr:.3e}", "sig_distinct": rr,
            })

    print("\nH3 is deliberately absent: its intervals bootstrap over seven venue folds,")
    print("not over frames, so seven facilities are seven observations and the width")
    print("already reflects that. Frame-level tests are the ones this audit is for.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
