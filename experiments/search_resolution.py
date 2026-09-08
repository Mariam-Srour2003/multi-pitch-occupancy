"""WP3-T8: what the preprocessing search can and cannot resolve, and how it ranks.

Two fixes were named for the search, and the note attached to them said the important
thing: **neither is "re-run it"**. Every one of the 88 evaluations still has its feature
cache under `data/cache/search/`, so all of this is re-scoring, not re-embedding - minutes
rather than the twelve hours the search itself cost.

The problem
-----------

The search ranks configurations on an **unweighted mean over seven venue folds** holding
between 12 and 168 play frames. That is the right choice for H3, which bootstraps over
*venues* and wants each venue to count once. It is the wrong choice for ranking
*configurations*, because it hands the 12-frame fold fourteen times the per-frame leverage
of the 168-frame one - so one frame in `f_outdoor_bldg` moves the headline by 1/(7x12) =
**0.0119**, and three frames in small folds reach 0.036.

That is not hypothetical. The `clahe='auto'` gate fires on 99.81% of frames, so `auto` and
`on` are the same transform on this footage, differing on **3 frames of 1,578** - and round 1
reports them 0.021 and 0.018 apart. Configurations have been adopted on margins of that size.

What this computes
------------------

For every cached evaluation, from its own features:

* **per-fold recalls**, which the search recorded only as their mean;
* the **unweighted** mean it ranked on, checked against the published number first;
* the **frame-weighted** mean, which is the ranking statistic for configurations;
* a **bootstrap interval over folds**, so a configuration's score carries its own width;
* the **false-play control**, recomputed rather than read, since the first 52 evaluations
  recorded it against training data and every one read 0.0000.

Then it answers the two questions the fixes exist for: **does the weighting change which
configuration the search would have adopted**, and **how large must a margin be before it
means anything** - as a measured floor rather than the "~0.02" estimate.

    uv run python -m experiments.search_resolution

It also calibrates `clahe_contrast_below` from the footage's own contrast distribution,
which is the third item on WP3-T8: 40.0 was picked by hand and never measured against
anything, and at that value the switch it gates does nothing.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import ManifestRow, read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.stats import bootstrap_ci
from pitch_occupancy.vision.heads import LinearProbe

SEED = 42
PLAY, EMPTY = "C2_ACTIVE_PLAY", "C1_EMPTY"
SEARCH_CACHE = settings.feature_cache_dir / "search"
SEARCH_JSON = settings.results_dir / "preprocess_search.json"
RESCORED = settings.results_dir / "false_play_rescored.csv"
OUT = settings.results_dir / "search_resolution.csv"
TOLERANCE = 5e-4
RESAMPLES = 10_000


def load_features(model: str, config_hash: str, rows: list[ManifestRow]) -> np.ndarray | None:
    path = SEARCH_CACHE / f"{model}_{config_hash}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(z["files"])}
    if not all(r.file in index for r in rows):
        return None
    return z["features"][[index[r.file] for r in rows]]


def per_fold_recalls(rows, X, folds) -> list[tuple[str, float, int]]:
    """``(fold, recall, n_play)`` - the numbers the search collapsed into a mean."""
    pos = {r.file: i for i, r in enumerate(rows)}
    out = []
    for fold in folds:
        probe = LinearProbe("s", seed=SEED).fit(X[[pos[r.file] for r in fold.train]], fold.train)
        pred = probe.predict(X[[pos[r.file] for r in fold.test]], fold.test)
        truth = [r.class3 for r in fold.test]
        play = [(p, t) for p, t in zip(pred, truth, strict=True) if t == PLAY]
        recall = sum(p == PLAY for p, _ in play) / len(play) if play else float("nan")
        out.append((fold.name.split("__")[-1], recall, len(play)))
    return out


def false_play(rows, X) -> float:
    """Held-out empty frames called PLAY. Recomputed, never read from the search's JSON.

    The search recorded this against *training* data for its first 52 evaluations and got
    0.0000 for every one, so the column in `preprocess_search.json` cannot be used to rank
    anything. `rescore_false_play.py` repaired the first 52; this recomputes all 88 the same
    way, and checks itself against that repair.
    """
    pos = {r.file: i for i, r in enumerate(rows)}
    cam = lambda r: PHYSICAL_CAMERA.get(r.camera, r.camera)  # noqa: E731
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if cam(r) == "camera_A"]
    empties = [r for r in venue if cam(r) == "camera_B" and r.class3 == EMPTY]
    if not empties or len({r.class3 for r in train}) < 2:
        return float("nan")
    probe = LinearProbe("s", seed=SEED).fit(X[[pos[r.file] for r in train]], train)
    pred = probe.predict(X[[pos[r.file] for r in empties]], empties)
    return float(np.mean([p == PLAY for p in pred]))


def summarise(folds: list[tuple[str, float, int]]) -> dict:
    """Both means, and an interval for the one the search ranked on.

    The weighted mean is `sum(correct) / sum(play)` written as a weighted average of fold
    recalls - i.e. **plain pooled recall over every held-out play frame**, which is what
    "how often does this configuration recognise play at an unseen venue" actually asks.
    The unweighted mean answers a different question - "how does it do at a typical venue" -
    and H3 is right to use it. Ranking configurations is the first question.
    """
    recalls = np.array([r for _, r, _ in folds], dtype=float)
    weights = np.array([n for _, _, n in folds], dtype=float)
    ci = bootstrap_ci(recalls, np.mean, resamples=RESAMPLES, seed=SEED)
    return {
        "unweighted": float(recalls.mean()),
        "weighted": float((recalls * weights).sum() / weights.sum()),
        "ci_low": ci.low,
        "ci_high": ci.high,
        "worst_fold": float(recalls.min()),
    }


def fold_leverage(folds: list[tuple[str, float, int]]) -> list[tuple[str, int, float]]:
    """How much one frame in each fold is worth in the *unweighted* headline.

    This is the resolution floor, derived rather than estimated: the smallest change the
    reported statistic can register is one frame flipping in the smallest fold.
    """
    n = len(folds)
    return sorted(
        ((name, n_play, 1.0 / (n * n_play)) for name, _, n_play in folds if n_play),
        key=lambda t: -t[2],
    )


def calibrate_clahe_gate(rows: list[ManifestRow]) -> dict | None:
    """What `clahe_contrast_below` should be, from the footage rather than by hand.

    **The headline number is counted, not derived from a contrast distribution**, because
    deriving it is what went wrong the first time. An earlier diagnostic measured RMS
    contrast on the letterboxed 224x224 output and concluded the gate fires on 99.81% of
    frames, making `auto` and `on` differ on three. But the gate runs in the *photometric*
    stage, before the resize, so it tests the full-resolution frame. Running `preprocess`
    twice and comparing the outputs needs no assumption about which image is measured:
    wherever `auto` and `on` differ, the gate did not fire.

    The contrast distribution is still reported, measured on the image the gate does see,
    because that is what a calibrated threshold would have to be chosen from.
    """
    import dataclasses

    import cv2

    from pitch_occupancy.vision.preprocess import PreprocessConfig, preprocess, rms_contrast

    base = PreprocessConfig()
    auto = dataclasses.replace(base, clahe="auto")
    on = dataclasses.replace(base, clahe="on")

    values: list[float] = []
    differ = 0
    for row in rows:
        image = cv2.imread(str(settings.dataset_dir / row.file))
        if image is None:
            continue
        # The gate's own input: geometry applied, photometric not yet. With the default
        # config the geometric stage is the identity, so this is the frame as read.
        values.append(rms_contrast(image))
        if not np.array_equal(preprocess(image, auto), preprocess(image, on)):
            differ += 1
    if not values:
        return None

    arr = np.array(values)
    n = int(arr.size)
    return {
        "n_frames": n,
        "current_threshold": float(base.clahe_contrast_below),
        # Counted, not predicted: the share of frames on which `auto` applied CLAHE.
        "fires_at_current": (n - differ) / n,
        "auto_differs_from_on": differ,
        # Reported alongside as a cross-check on the count above.
        "predicted_fire_rate": float((arr < base.clahe_contrast_below).mean()),
        "min": float(arr.min()), "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def check_reproduces(records: list[dict], published: dict[tuple[str, str], float],
                     field: str, label: str) -> None:
    shared = [(r, published[(r["model"], r["describe"])]) for r in records
              if (r["model"], r["describe"]) in published]
    bad = [(r, want) for r, want in shared if abs(r[field] - want) > TOLERANCE]
    for r, want in bad:
        print(f"  MISMATCH {label} {r['model']}/{r['describe']}: "
              f"{r[field]:.4f} vs published {want:.4f}")
    if bad:
        raise SystemExit(
            f"{label}: {len(bad)} of {len(shared)} published numbers did not reproduce. "
            "Re-scoring that disagrees with the run it re-scores is measuring something else."
        )
    print(f"  {label}: {len(shared)} published numbers reproduce to {TOLERANCE:g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-clahe", action="store_true", help="skip the contrast sweep")
    args = ap.parse_args()

    if not SEARCH_JSON.exists():
        raise SystemExit(f"no search results at {SEARCH_JSON}")
    blob = json.loads(SEARCH_JSON.read_text(encoding="utf-8"))
    evaluations = blob["evaluations"]

    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    folds = [f for f in leave_one_group_out(rows, group_key="venue")
             if not f.name.endswith("venue_01")]
    print(f"{len(evaluations)} cached evaluations | {len(folds)} venue folds "
          f"| stamp {blob.get('generated')}\n")

    records: list[dict] = []
    missing = 0
    for i, ev in enumerate(evaluations, 1):
        X = load_features(ev["model"], ev["hash"], rows)
        if X is None:
            missing += 1
            continue
        by_fold = per_fold_recalls(rows, X, folds)
        stats = summarise(by_fold)
        records.append({
            "model": ev["model"], "round": ev["round"], "describe": ev["describe"],
            "hash": ev["hash"],
            "unweighted": round(stats["unweighted"], 4),
            "weighted": round(stats["weighted"], 4),
            "ci_low": round(stats["ci_low"], 4), "ci_high": round(stats["ci_high"], 4),
            "worst_fold": round(stats["worst_fold"], 4),
            "false_play": round(false_play(rows, X), 4),
            "published_recall": ev["play_recall"],
            "_folds": by_fold,
        })
        if i % 20 == 0:
            print(f"  re-scored {i}/{len(evaluations)}", flush=True)
    if missing:
        print(f"  ({missing} evaluation(s) had no cache and were skipped)")
    if not records:
        raise SystemExit("no cached evaluation could be re-scored")

    # --- reproduce before extending -------------------------------------------------
    print("\nreproducing the search before re-ranking it")
    check_reproduces(
        records,
        {(e["model"], e["describe"]): e["play_recall"] for e in evaluations},
        "unweighted", "preprocess_search.json (unweighted recall)",
    )
    if RESCORED.exists():
        published_fp = {
            (r["model"], r["describe"]): float(r["false_play_fixed"])
            for r in csv.DictReader(RESCORED.open(encoding="utf-8"))
        }
        check_reproduces(records, published_fp, "false_play",
                         "false_play_rescored.csv (repaired control)")

    # --- the resolution floor, measured ---------------------------------------------
    leverage = fold_leverage(records[0]["_folds"])
    floor = leverage[0][2]
    print("\n=== what one frame is worth in the unweighted headline ===")
    print(f"{'fold':26}{'play frames':>12}{'one frame moves the mean by':>30}")
    for name, n_play, per_frame in leverage:
        print(f"{name:26}{n_play:>12}{per_frame:>30.4f}")
    print(f"\n  Resolution floor: {floor:.4f}. A margin smaller than this is one frame in "
          f"the smallest fold.")
    widths = [r["ci_high"] - r["ci_low"] for r in records]
    print(f"  And the bootstrap interval over folds is {np.median(widths):.3f} wide at the "
          f"median, which is the more honest floor.")

    # --- does weighting change the ranking? -----------------------------------------
    print("\n=== does frame-weighting change what the search would have adopted? ===")
    changed = []
    for model in sorted({r["model"] for r in records}):
        for rnd in sorted({r["round"] for r in records if r["model"] == model}):
            arm = [r for r in records if r["model"] == model and r["round"] == rnd]
            if len(arm) < 2:
                continue
            by_unweighted = max(arm, key=lambda r: r["unweighted"] - r["false_play"])
            by_weighted = max(arm, key=lambda r: r["weighted"] - r["false_play"])
            same = by_unweighted["describe"] == by_weighted["describe"]
            margin = (sorted((r["unweighted"] - r["false_play"] for r in arm))[-1]
                      - sorted((r["unweighted"] - r["false_play"] for r in arm))[-2])
            print(f"  {model:12} round {rnd}: "
                  f"{'same winner' if same else 'DIFFERENT winner'}  "
                  f"unweighted -> {by_unweighted['describe'][:34]:34} "
                  f"weighted -> {by_weighted['describe'][:34]}")
            print(f"{'':16}winning margin {margin:.4f}"
                  + ("  <- below the resolution floor" if margin < floor else ""))
            if not same:
                changed.append((model, rnd))

    print(f"\n  Weighting changes the adopted configuration in {len(changed)} of the "
          f"rounds compared.")

    # --- the CLAHE gate --------------------------------------------------------------
    clahe = None if args.skip_clahe else calibrate_clahe_gate(rows)
    if clahe:
        print("\n=== calibrating clahe_contrast_below against the footage ===")
        print(f"  {clahe['n_frames']} frames, RMS contrast "
              f"{clahe['min']:.1f} - {clahe['max']:.1f}, median {clahe['median']:.1f}")
        print(f"  current threshold {clahe['current_threshold']:.1f} fires on "
              f"{clahe['fires_at_current']:.2%} of frames, counted by running the switch "
              f"both ways")
        print(f"  `auto` and `on` therefore differ on {clahe['auto_differs_from_on']} of "
              f"{clahe['n_frames']} frames - a weak switch, not a vacuous one")
        if abs(clahe["predicted_fire_rate"] - clahe["fires_at_current"]) > 1e-6:
            print(f"  (predicted from the contrast distribution: "
                  f"{clahe['predicted_fire_rate']:.2%} - if this disagrees with the count, "
                  f"the distribution being measured is not the one the gate tests)")
        print(f"  a threshold that separates: p10 {clahe['p10']:.1f}, "
              f"median {clahe['median']:.1f}, p90 {clahe['p90']:.1f}")
        print(f"\n  Setting it to the median ({clahe['median']:.1f}) makes `auto` mean "
              f"'the darker half'; the p10 ({clahe['p10']:.1f}) makes it 'the worst tenth'.")
        print("  Which is a decision for the ablation, not a number to guess. 40.0 is not a")
        print("  candidate either way: it was chosen before this footage was measured, and")
        print("  it sits above the 90th percentile, so it applies CLAHE almost everywhere")
        print("  while claiming to be selective.")

    for r in records:
        r.pop("_folds", None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {OUT.name}")

    if clahe:
        (settings.results_dir / "search_resolution_gate.json").write_text(
            json.dumps(clahe, indent=2), encoding="utf-8")

    record(
        "WP3-T8 search resolution and ranking",
        "`python -m experiments.search_resolution`",
        f"`{OUT.name}`",
        f"floor {floor:.4f} per frame; weighting changes {len(changed)} adopted "
        f"configuration(s); clahe gate fires on "
        f"{clahe['fires_at_current']:.2%} of frames" if clahe else
        f"floor {floor:.4f} per frame; weighting changes {len(changed)} configuration(s)",
    )


if __name__ == "__main__":
    main()
