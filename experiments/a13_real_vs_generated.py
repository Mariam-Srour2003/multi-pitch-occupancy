"""A13's withdrawal gate: can a probe tell a generated frame from a recorded one?

Declared in `thesis/preregistration.md` A13 **before the frames existed**, so the threshold
could not be chosen after seeing the answer:

> Train a probe on frozen features to classify real vs generated. If it reaches
> macro-F1 > 0.90, the generated frames are a distinguishable distribution rather than an
> augmentation of this one, and the augmentation is withdrawn.

The point is not that a separable set is useless - it is that a separable set is not what
A13 licensed. Generated frames were admitted as *more of the same distribution*, thin in
classes the corpus cannot cover. If a linear probe on frozen features can pick them out,
they are a second distribution wearing the corpus's labels, and a model trained on both
learns "was this drawn or recorded?" alongside - or instead of - "is this pitch in use?".

Run before any classifier is retrained on them.

    uv run python experiments/a13_real_vs_generated.py --backbone dinov2

The split is grouped by **venue**, so the probe is never scored on a venue it trained on.
A random split here would let it memorise each venue's background and report a number about
scenery. That is the same mistake this whole project exists to document, and it would be a
poor place to make it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pitch_occupancy.config import RESULTS_DIR
from pitch_occupancy.data.feature_cache import build_cache, load_cache
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.evaluation.stats import bootstrap_metric_ci

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = "synthetic"
WITHDRAW_ABOVE = 0.90  # pre-declared in A13


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/processed/manifest.csv")
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "a13_real_vs_generated.json")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    by_file = {r.file: r for r in rows}

    cached = load_cache(args.backbone, args.cache_dir)
    feats = {f: v for f, v in zip(cached.files, cached.features)}

    missing = [r for r in rows if r.file not in feats]
    if missing:
        print(f"embedding {len(missing)} frame(s) absent from the {args.backbone} cache ...")
        extra_dir = args.cache_dir / "a13_gate"
        extra_dir.mkdir(parents=True, exist_ok=True)
        extra = build_cache(missing, args.backbone, args.dataset_dir, extra_dir)
        feats.update(dict(zip(extra.files, extra.features)))

    files = [r.file for r in rows if r.file in feats]
    X = np.stack([feats[f] for f in files])
    # String labels, not 0/1: `bootstrap_metric_ci` casts to dtype=object, and sklearn
    # reads an object array of ints as target type "unknown" and refuses it.
    y = np.array(["generated" if by_file[f].source == SYNTHETIC else "recorded"
                  for f in files])
    venues = np.array([by_file[f].venue for f in files])

    n_gen = int((y == "generated").sum())
    n_real = int((y == "recorded").sum())
    print(f"\n{n_real} recorded, {n_gen} generated, {X.shape[1]}-d {args.backbone} features")
    if n_gen == 0:
        print("no generated frames - nothing to test")
        return 0

    # Leave-one-venue-out over venues that hold generated frames. A venue with none cannot
    # contribute a positive to the held-out side and would only inflate the denominator.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gen_venues = sorted({v for v, lab in zip(venues, y) if lab == "generated"})
    per_venue: dict[str, float] = {}
    all_true: list[int] = []
    all_pred: list[int] = []

    for held in gen_venues:
        te = venues == held
        tr = ~te
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            print(f"  {held}: skipped - one class only on a side")
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        f1 = _macro_f1(y[te], pred)
        per_venue[held] = f1
        all_true.extend(y[te].tolist())
        all_pred.extend(pred.tolist())
        print(f"  hold out {held:28s} n={te.sum():5d}  macro-F1 {f1:.4f}")

    pooled = _macro_f1(np.array(all_true), np.array(all_pred))
    ci = bootstrap_metric_ci(np.array(all_true), np.array(all_pred), _macro_f1, seed=0)
    lo, hi = ci.low, ci.high

    verdict = "WITHDRAW" if pooled > WITHDRAW_ABOVE else "RETAIN"
    print(f"\npooled macro-F1 = {pooled:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"pre-declared threshold = {WITHDRAW_ABOVE}")
    print(f"VERDICT: {verdict}")
    if verdict == "WITHDRAW":
        print(
            "\nThe generated frames are a distinguishable distribution. Under A13 the\n"
            "augmentation is withdrawn - not quietly dropped, but reported as a finding."
        )
    else:
        print(
            "\nThe probe cannot reliably separate them, which is the condition A13 set for\n"
            "keeping the augmentation. It is not evidence that they are useful - only that\n"
            "they are not trivially distinguishable."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "n_recorded": n_real,
                "n_generated": n_gen,
                "protocol": "leave-one-venue-out over venues holding generated frames",
                "per_venue_macro_f1": per_venue,
                "pooled_macro_f1": pooled,
                "ci95": [lo, hi],
                "threshold": WITHDRAW_ABOVE,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
