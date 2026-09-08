"""WP4-T6 - what the models get wrong, and whether the errors are a set worth reading.

The task says "categorise every grouped-split misclassification by (condition, confusion
pair, camera)". Done on the grouped split alone that table is nearly empty, and the emptiness
is the first finding: **every grouped-split error comes from one slot, at night, and for two
of the three backbones every one of them is the same confusion** - an empty pitch called
active play. Nine errors from one scene is not a taxonomy, it is a single failure counted
nine times.

So the same categorisation runs across all four protocols from WP4-T1. The error *structure*
changes completely between them, and that contrast is what makes the table worth having:

* **random (leaky)** - few errors, and the question is whether they had a near-duplicate in
  training;
* **grouped** - the honest in-venue split, where the test side is 97.6% one class;
* **cross-venue** - held-out venues are 100% active play, so every error is a missed match;
* **temporal** - trained on a daylight morning and tested on a floodlit night, so the errors
  are what a total condition shift does.

Every error row carries the frame's venue, physical camera, lighting and slot, plus one
column the others cannot supply:

**`near_duplicate_in_train`** - does this frame have a near-duplicate on the training side?
On a leaky split that is how an error survives memorisation; on an honest one it should be
zero by construction, so a non-zero count is a defect in the split rather than a property of
the model. It is a check on the protocol, run through the error set.

    uv run python -m experiments.error_taxonomy

**The explainability half is not done here, and not because it was forgotten.** Attention
rollout and Grad-CAM would write frame images to `results/figs/xai/`, and WP1-T5 records
that nine images are already permanent in git history including five sheets of unblurred
players. Publishing more frames is the data-release decision, which is open and is not mine
to take. The overlays are worth doing once it is settled.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import DEFAULT_THRESHOLD, dhash, hamming
from pitch_occupancy.data.feature_cache import features_for, load_cache
from pitch_occupancy.data.manifest import ManifestRow, read_manifest
from pitch_occupancy.data.splits import (
    Split,
    development_rows,
    grouped_split,
    leave_one_group_out,
    random_split,
    temporal_split,
)
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision.backbones import BACKBONES
from pitch_occupancy.vision.heads import LinearProbe

SEED = 42
OUT = settings.results_dir / "error_taxonomy.csv"
TEMPORAL_CUTOFF = "2026-07-12"


def short(label: str) -> str:
    """`C1_EMPTY` -> `C1`. The confusion pair is the unit, not the prose name."""
    return label.split("_")[0]


def physical(row: ManifestRow) -> str:
    return PHYSICAL_CAMERA.get(row.camera, row.camera)


def protocols(rows: list[ManifestRow]) -> list[tuple[str, Split]]:
    out: list[tuple[str, Split]] = [
        ("random", random_split(rows, seed=SEED)),
        ("grouped_slot", grouped_split(rows, group_key="slot_id", seed=SEED)),
    ]
    for fold in leave_one_group_out(rows, group_key="venue"):
        if len({r.class3 for r in fold.train}) < 2:
            continue
        out.append(("lo_venue_out", fold))
    out.append(("temporal", temporal_split(rows, cutoff_date=TEMPORAL_CUTOFF)))
    return out


def hashes_for(rows: list[ManifestRow], cache: dict[str, int]) -> dict[str, int]:
    """Perceptual hashes, computed once per frame across every protocol."""
    import cv2

    for row in rows:
        if row.file in cache:
            continue
        image = cv2.imread(str(settings.dataset_dir / row.file))
        if image is not None:
            cache[row.file] = dhash(image)
    return cache


def has_near_duplicate(file: str, train_hashes: list[int], cache: dict[str, int]) -> bool | None:
    """Is this frame a near-duplicate of something the model trained on?

    On a leaky split this is how a frame can be got right by memorisation - and, read the
    other way, an error *despite* a near-duplicate in training is a genuinely hard frame.
    On an honest split the count should be zero by construction, which makes this a check on
    the protocol rather than on the model.
    """
    h = cache.get(file)
    if h is None:
        return None
    return any(hamming(h, t) <= DEFAULT_THRESHOLD for t in train_hashes)


def errors_for(split: Split, rows: list[ManifestRow], X: np.ndarray, model: str,
               cache: dict[str, int], check_duplicates: bool) -> list[dict]:
    pos = {r.file: i for i, r in enumerate(rows)}
    probe = LinearProbe(model, seed=SEED).fit(X[[pos[r.file] for r in split.train]], split.train)
    pred = probe.predict(X[[pos[r.file] for r in split.test]], split.test)

    train_hashes: list[int] = []
    if check_duplicates:
        hashes_for(list(split.train), cache)
        train_hashes = [cache[r.file] for r in split.train if r.file in cache]

    out = []
    for row, guess in zip(split.test, pred, strict=True):
        if guess == row.class3:
            continue
        if check_duplicates:
            hashes_for([row], cache)
        out.append({
            "model": model,
            "true": short(row.class3),
            "predicted": short(guess),
            "pair": f"{short(row.class3)}->{short(guess)}",
            "venue": row.venue,
            "camera": physical(row),
            "lighting": row.lighting,
            "slot_id": row.slot_id,
            "file": row.file,
            "near_duplicate_in_train": (
                has_near_duplicate(row.file, train_hashes, cache) if check_duplicates else ""
            ),
        })
    return out


def concentration(errors: list[dict], field: str) -> str:
    """How many distinct values the errors span, and the share taken by the commonest.

    An error set spread over one slot is a single failure counted many times, and a table
    that reports only counts hides that. This is the column that says whether the taxonomy
    has anything to categorise.
    """
    if not errors:
        return "—"
    counts = Counter(e[field] for e in errors)
    top, n = counts.most_common(1)[0]
    return f"{len(counts)} ({top} {n / len(errors):.0%})"


def finding(protocol: str, model: str, errors: list[dict], n_test: int) -> str:
    """One sentence per model per protocol, derived from the errors rather than written.

    The acceptance criterion asks for "a finding sentence per class". A sentence typed by
    hand beside a generated table is the arrangement that let a figure contradict its own
    source, so this is composed from the counts.
    """
    if not errors:
        return f"no errors on {n_test} test frames"
    pairs = Counter(e["pair"] for e in errors)
    pair, n = pairs.most_common(1)[0]
    slots = len({e["slot_id"] for e in errors})
    lightings = {e["lighting"] for e in errors}
    parts = [f"{len(errors)} errors of {n_test}"]
    parts.append(
        f"all {pair}" if len(pairs) == 1 else
        f"{pair} is {n / len(errors):.0%} of them across {len(pairs)} pairs"
    )
    parts.append(f"from {slots} slot{'s' if slots != 1 else ''}")
    if len(lightings) == 1:
        parts.append(f"all {next(iter(lightings))}")
    dupes = [e for e in errors if e["near_duplicate_in_train"] is True]
    if dupes:
        parts.append(f"{len(dupes) / len(errors):.0%} had a near-duplicate in training")
    return "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-duplicates", action="store_true",
                    help="skip the near-duplicate column, which reads every frame")
    args = ap.parse_args()

    rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    features: dict[str, np.ndarray] = {}
    for key in sorted(BACKBONES):
        try:
            X, kept = features_for(load_cache(key, settings.feature_cache_dir), rows)
        except FileNotFoundError:
            continue
        if len(kept) == len(rows):
            features[key] = X
    if not features:
        raise SystemExit("no feature cache; run `uv run pitch cache`")

    cache: dict[str, int] = {}
    records: list[dict] = []
    summary: list[dict] = []

    for protocol, split in protocols(rows):
        # The near-duplicate column is only informative where the protocol permits one; on
        # the honest splits it is a check that it does not, and on cross-venue folds the
        # answer is fixed by construction, so it is computed on the two in-venue protocols.
        check = not args.skip_duplicates and protocol in ("random", "grouped_slot")
        for model, X in features.items():
            errs = errors_for(split, rows, X, model, cache, check)
            for e in errs:
                records.append({"protocol": protocol, "replicate": split.name, **e})
            summary.append({
                "protocol": protocol, "replicate": split.name, "model": model,
                "n_test": len(split.test), "n_errors": len(errs),
                "error_rate": round(len(errs) / len(split.test), 4) if split.test else "",
                "distinct_pairs": concentration(errs, "pair"),
                "distinct_slots": concentration(errs, "slot_id"),
                "distinct_cameras": concentration(errs, "camera"),
                "finding": finding(protocol, model, errs, len(split.test)),
            })

    print(f"{len(records)} misclassifications across "
          f"{len({(s['protocol'], s['replicate']) for s in summary})} splits\n")

    for protocol in ("random", "grouped_slot", "lo_venue_out", "temporal"):
        block = [s for s in summary if s["protocol"] == protocol]
        if not block:
            continue
        print(f"=== {protocol} ===")
        for model in sorted({s["model"] for s in block}):
            mine = [s for s in block if s["model"] == model]
            n_err = sum(s["n_errors"] for s in mine)
            n_test = sum(s["n_test"] for s in mine)
            errs = [r for r in records if r["protocol"] == protocol and r["model"] == model]
            print(f"  {model:11} {n_err:>4} of {n_test:>5}  "
                  f"pairs {concentration(errs, 'pair'):<18} "
                  f"slots {concentration(errs, 'slot_id'):<14} "
                  f"lighting {concentration(errs, 'lighting')}")
        print()

    # --- what the concentration means -------------------------------------------------
    print("=== is there anything to categorise? ===")
    for protocol in ("random", "grouped_slot", "lo_venue_out", "temporal"):
        errs = [r for r in records if r["protocol"] == protocol]
        if not errs:
            continue
        slots = len({e["slot_id"] for e in errs})
        pairs = len({e["pair"] for e in errs})
        verdict = ("a single failure counted many times" if slots <= 1 else
                   "one condition" if len({e["lighting"] for e in errs}) == 1 and slots < 4 else
                   "a spread worth categorising")
        print(f"  {protocol:14} {len(errs):>4} errors from {slots:>2} slot(s), "
              f"{pairs} confusion pair(s) - {verdict}")

    # --- the near-duplicate check on the protocols that permit one --------------------
    checked = [r for r in records if r["near_duplicate_in_train"] != ""]
    if checked:
        print("\n=== did the error have a near-duplicate on the training side? ===")
        print("  On the leaky split this is what memorisation looks like; on the honest one")
        print("  it should be zero by construction, so a non-zero count is a split defect.")
        for protocol in sorted({r["protocol"] for r in checked}):
            block = [r for r in checked if r["protocol"] == protocol]
            n = sum(1 for r in block if r["near_duplicate_in_train"] is True)
            print(f"  {protocol:14} {n:>4} of {len(block):>4} errors had one "
                  f"({n / len(block):.0%})")

    print("\n=== one sentence per model per protocol ===")
    for s in summary:
        if s["protocol"] == "lo_venue_out" and not s["n_errors"]:
            continue
        label = s["replicate"].split("__")[-1] if "__" in s["replicate"] else s["protocol"]
        print(f"  {s['model']:11} {label:26} {s['finding']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    summary_path = settings.results_dir / "error_taxonomy_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {OUT.name} and {summary_path.name}")

    print("\nThe explainability half (attention rollout, Grad-CAM) is not done: it writes")
    print("frame images, and publishing more frames is WP1-T5's open data-release decision.")

    record(
        "WP4-T6 error taxonomy",
        "`python -m experiments.error_taxonomy`",
        f"`{OUT.name}`",
        f"{len(records)} misclassifications across four protocols; "
        f"the grouped split's errors are one slot",
    )


if __name__ == "__main__":
    main()
