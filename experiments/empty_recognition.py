"""What the false-play control actually measures, and whether C3's six frames are to blame.

Two questions, one experiment, because the answer to the first turned out to be the answer
to the second.

**Where this started.** `fusion_head_ablation.py` added a column asking what each model
answers *instead* of ACTIVE_PLAY on the 243 held-out empty frames, and found that DINOv2 -
whose false-play rate of 0.309 is quoted across this repository as its headline strength -
gets **none of the 243 right**. It answers ACTIVE_PLAY 75 times and MAINTENANCE 168 times.
That entry blamed the training mix: 518 ACTIVE_PLAY, 251 EMPTY and **6** MAINTENANCE frames,
with `class_weight="balanced"` giving the six-frame class a weight of 43.

**That explanation was half right, and the half that was wrong is the more interesting half.**
Removing C3 from training entirely does stop the MAINTENANCE absorption - and leaves EMPTY
accuracy at exactly 0.0000, with all 243 frames now called ACTIVE_PLAY instead. The class
weighting decides *which* wrong answer the model gives. It has nothing to do with whether the
model can recognise an empty pitch.

**What decides it is whether the model has ever seen a labelled empty frame of that camera,
and the amount needed is about one.**

| training set | DINOv2 | ConvNeXtV2 | ViT |
|---|---|---|---|
| camera A only *(the published protocol)* | 0.0000 | 0.0082 | 0.1646 |
| camera A + camera B's ACTIVE_PLAY frames | 0.7325 | 0.1975 | 0.0041 |
| the above + **1** labelled empty frame of camera B | **0.9793** | **0.9793** | **0.9793** |
| + 25 labelled empty frames | 0.9817 | 0.9817 | 0.9817 |

So there are two findings, and the second is the useful one.

*First*, **the false-play control is a camera-transfer test**, not a specificity test. Every
false-play number this project has published is a cross-camera transfer number, and reading
them as "how often does the model mistake an empty pitch for a match" overstates what the
protocol asks. The numbers are correct; the caption is wrong.

The middle row shows how badly it conflates the two. There camera B appears in training in
**one class only**, which makes "camera B implies play" available as a shortcut, and the three
backbones respond in opposite directions: DINOv2 largely resists it and reaches 0.73,
ConvNeXtV2 partly takes it, and ViT takes it completely and *falls* from 0.165 to 0.004. On
that axis the model ranking is partly a ranking of shortcut resistance, which is not what the
column is captioned as measuring. It is the same confound this project keeps meeting, in a new
place.

*Second*, and this is the practical answer: **one labelled empty frame of a new camera is
worth more than 769 frames of a different one.** All three backbones jump to 0.9793 at k=1 and
stay flat to k=25 - the curve has no slope to climb. Adding a camera does not need a new
dataset; it needs a handful of labelled frames from that camera.

**Three caveats, all of which cut against the flattering reading.**

The 243 empty frames are **three distinct scenes**. That is *why* one frame suffices: a fixed
camera pointed at an empty pitch genuinely sees about three views, so the first labelled frame
already covers one of them. The flatness from k=1 to k=25 is the evidence, and it means the
k-shot rows are not a learning curve and must not be drawn as one.

Every k-shot row is therefore reported with a **count of near-duplicate pairs crossing the
train/test boundary**, which at k=1 is already 1,400. The single training frame is a near
copy of much of the test set. This is honest about the mechanism rather than presenting 0.979
as clean generalisation - though it is also the operational reality, because the deployed
system really would be scoring frames that look like the ones it was given.

The within-camera 1.0000 is the leaky upper bound, with 13,406 crossing pairs, and is included
as the ceiling the honest protocols are measured against rather than as a score.

    uv run python experiments/empty_recognition.py
"""

from __future__ import annotations

import csv
from collections import Counter

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import dhash, distinct_subset, hamming
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision.heads import LinearProbe

PLAY = "C2_ACTIVE_PLAY"
EMPTY = "C1_EMPTY"
MAINT = "C3_MAINTENANCE_NON_SPORTING"
SEED = 42
OUT = settings.results_dir / "empty_recognition.csv"
PUBLISHED = settings.results_dir / "h3_with_false_play.csv"
BACKBONES = ("dinov2", "convnextv2", "vit")

#: The four class configurations. `balanced` matches the project's default and the published
#: numbers; the unbalanced rows exist to show that the weighting changes *which* wrong answer
#: appears and not whether the answer is wrong.
CLASS_CONFIGS = (
    ("3class_balanced", True, True),
    ("3class_unbalanced", True, False),
    ("2class_balanced", False, True),
    ("2class_unbalanced", False, False),
)


def camera(row) -> str:
    return PHYSICAL_CAMERA.get(row.camera, row.camera)


def load(backbone: str):
    d = np.load(settings.feature_cache_dir / f"{backbone}.npz", allow_pickle=True)
    index = {str(f): i for i, f in enumerate(d["files"])}
    rows = [
        r for r in development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
        if r.file in index
    ]
    return rows, d["features"][[index[r.file] for r in rows]]


def fit_predict(X, pos, train_rows, test_rows, *, balanced=True, backbone="dinov2"):
    tr = [pos[r.file] for r in train_rows]
    te = [pos[r.file] for r in test_rows]
    probe = LinearProbe(backbone, balanced=balanced, seed=SEED).fit(X[tr], train_rows)
    return probe.predict(X[te], test_rows)


def scores(predicted) -> dict[str, float]:
    """Both readings of the same predictions, side by side, so neither can stand alone."""
    return {
        "empty_accuracy": float(np.mean([p == EMPTY for p in predicted])),
        "false_play_rate": float(np.mean([p == PLAY for p in predicted])),
        "maintenance_rate": float(np.mean([p == MAINT for p in predicted])),
    }


def hashes_for(rows) -> dict[str, int]:
    import cv2

    out = {}
    for r in rows:
        image = cv2.imread(str(settings.dataset_dir / r.file))
        if image is not None:
            out[r.file] = dhash(image)
    return out


def crossing_pairs(train_rows, test_rows, cache: dict[str, int], threshold: int = 6) -> int:
    """Near-duplicate pairs straddling the train/test boundary.

    The number that says whether a configuration's score is earned. A pair across the
    boundary is one frame being scored against a copy of itself, and the within-camera
    protocol below is built precisely so this is large - it is the leaky upper bound, and
    reporting it without this count would be reporting the leak as a result.
    """
    a = [r.file for r in train_rows if r.file in cache]
    b = [r.file for r in test_rows if r.file in cache]
    return sum(
        1 for x in a for y in b if hamming(cache[x], cache[y]) <= threshold
    )


def published_false_play() -> dict[str, float]:
    if not PUBLISHED.exists():
        return {}
    return {
        r["model"]: float(r["false_play_rate"])
        for r in csv.DictReader(PUBLISHED.open(encoding="utf-8"))
        if r["held_out_venue"] == "MEAN_ACROSS_FOLDS" and r.get("false_play_rate")
    }


def main() -> None:
    rows, _ = load(BACKBONES[0])
    venue = [r for r in rows if r.venue == "venue_01"]
    cam_a = [r for r in venue if camera(r) == "camera_A"]
    cam_b = [r for r in venue if camera(r) == "camera_B"]
    b_empty = [r for r in cam_b if r.class3 == EMPTY]
    b_play = [r for r in cam_b if r.class3 == PLAY]

    print("venue_01, class x camera:")
    table = Counter((camera(r), r.class3) for r in venue)
    for key in sorted(table):
        print(f"   {key[0]:<10} {key[1]:<32} {table[key]}")

    print(f"\nhashing {len(venue)} venue_01 frames for the leakage counts...")
    cache = hashes_for(venue)
    scene_ids = distinct_subset({r.file: cache[r.file] for r in b_empty if r.file in cache})
    print(f"  the {len(b_empty)} held-out empty frames are {len(scene_ids)} distinct scenes")

    results: list[dict] = []

    # --- axis 1: does C3 explain it? ------------------------------------------------
    print("\n=== axis 1: class configuration (train camera A -> camera B empties) ===")
    print(f"{'backbone':<12}{'configuration':<20}{'EMPTY acc':>11}{'false-play':>12}"
          f"{'maint':>8}   answers")
    for backbone in BACKBONES:
        rows_b, X = load(backbone)
        pos = {r.file: i for i, r in enumerate(rows_b)}
        for name, keep_c3, balanced in CLASS_CONFIGS:
            train = [r for r in cam_a if keep_c3 or r.class3 != MAINT]
            predicted = fit_predict(X, pos, train, b_empty, balanced=balanced,
                                    backbone=backbone)
            s = scores(predicted)
            results.append({"axis": "class_configuration", "backbone": backbone,
                            "configuration": name, "n_train": len(train),
                            "n_test": len(b_empty), "n_distinct_scenes": len(scene_ids),
                            "crossing_pairs": "", **s})
            answers = "  ".join(f"{k.split('_', 1)[1].lower()} {v}"
                                for k, v in sorted(Counter(predicted).items()))
            print(f"{backbone:<12}{name:<20}{s['empty_accuracy']:11.4f}"
                  f"{s['false_play_rate']:12.4f}{s['maintenance_rate']:8.3f}   {answers}")

    # --- axis 2: how much of the new camera does the model need? ---------------------
    # `A_plus_B_play` is not a clean "camera is represented" condition and must not be read
    # as one: camera B appears there in *one class only*, so "camera B implies play" is
    # available as a shortcut, and taking it scores 100% on B's play frames. The k-shot rows
    # below are the clean version - camera B appears in both classes - and the two together
    # separate "has seen the camera" from "has seen the camera's empty pitch".
    half = len(b_empty) // 2
    held_back = {f.file for f in b_empty[:half]}
    rng = np.random.default_rng(SEED)
    shots = [0, 1, 3, 5, 10, 25]
    order = list(rng.permutation(len(b_empty)))

    protocols = [
        ("A_only__published", cam_a, b_empty,
         "the published protocol: camera B is entirely unseen"),
        ("A_plus_B_play__SHORTCUT", cam_a + b_play, b_empty,
         "camera B in ONE class only, so 'camera B implies play' is available as a shortcut"),
    ]
    for k in shots:
        if k == 0:
            continue
        chosen = {b_empty[i].file for i in order[:k]}
        train = cam_a + b_play + [r for r in b_empty if r.file in chosen]
        test = [r for r in b_empty if r.file not in chosen]
        protocols.append((
            f"A_plus_B_play_plus_{k:02d}_empties", train, test,
            f"{k} labelled empty frame(s) of camera B; the camera is now in both classes",
        ))
    protocols.append((
        "within_B__LEAKY", [r for r in cam_b if r.file not in held_back], b_empty[:half],
        "within camera B; near-duplicate leakage by construction - an upper bound, not a score",
    ))

    print("\n=== axis 2: how much of camera B does the model need to see? ===")
    print(f"{'backbone':<12}{'protocol':<34}{'n_test':>7}{'EMPTY acc':>11}"
          f"{'false-play':>12}{'leak pairs':>12}")
    for backbone in BACKBONES:
        rows_b, X = load(backbone)
        pos = {r.file: i for i, r in enumerate(rows_b)}
        for name, train, test, _note in protocols:
            predicted = fit_predict(X, pos, train, test, backbone=backbone)
            s = scores(predicted)
            leak = crossing_pairs(train, test, cache)
            subset = distinct_subset({r.file: cache[r.file] for r in test if r.file in cache})
            results.append({"axis": "camera_representation", "backbone": backbone,
                            "configuration": name, "n_train": len(train),
                            "n_test": len(test), "n_distinct_scenes": len(subset),
                            "crossing_pairs": leak, **s})
            print(f"{backbone:<12}{name:<34}{len(test):>7}{s['empty_accuracy']:11.4f}"
                  f"{s['false_play_rate']:12.4f}{leak:12,}")
        print()
    for name, _, _, note in protocols:
        print(f"  {name:<34} {note}")
    print("\n  Read the k-shot rows with their leak column. The 243 empty frames are 3 distinct")
    print("  scenes, so even k=1 puts a near-duplicate of much of the test set into training;")
    print("  that is a fact about this camera's view, not a trick, and it is why the practical")
    print("  answer below is stated as 'a handful of frames' rather than as a learning curve.")

    # --- reproduction check ----------------------------------------------------------
    print("\n=== reproduction check against the published false-play table ===")
    reference = published_false_play()
    baseline = {
        r["backbone"]: r["false_play_rate"] for r in results
        if r["axis"] == "class_configuration" and r["configuration"] == "3class_balanced"
    }
    for backbone in BACKBONES:
        want = reference.get(backbone)
        got = baseline[backbone]
        agree = want is None or abs(got - want) < 0.005
        shown = "-" if want is None else f"{want:.4f}"
        print(f"  {backbone:<12} {got:.4f}  published {shown}  "
              f"{'ok' if agree else 'MISMATCH'}")

    # --- the finding, stated in numbers ----------------------------------------------
    print("\n=== what this changes ===")
    for backbone in BACKBONES:
        got = {r["configuration"]: r for r in results
               if r["axis"] == "camera_representation" and r["backbone"] == backbone}
        published_acc = got["A_only__published"]["empty_accuracy"]
        shortcut = got["A_plus_B_play__SHORTCUT"]["empty_accuracy"]
        one_shot = got["A_plus_B_play_plus_01_empties"]["empty_accuracy"]
        print(f"  {backbone:<12} EMPTY accuracy {published_acc:.4f} (published protocol) -> "
              f"{shortcut:.4f} (camera in one class) -> {one_shot:.4f} (one labelled frame)")
    print("\n  1. The false-play control is a camera-transfer test. Its numbers stand; the")
    print("     reading 'the model mistakes an empty pitch for a match' does not.")
    print("  2. Removing C3 does not help - it changes which wrong answer appears.")
    print("  3. One labelled empty frame of a camera is worth more than 769 frames of")
    print("     another one. Adding a camera needs labels from it, not a bigger dataset.")
    print("     Read that with the leak column and the 3-distinct-scene count above.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["axis", "backbone", "configuration", "n_train", "n_test", "n_distinct_scenes",
              "crossing_pairs", "empty_accuracy", "false_play_rate", "maintenance_rate"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"\nwrote {OUT.name}")

    lift = [r for r in results if r["axis"] == "camera_representation"]
    a = next(r for r in lift if r["backbone"] == "dinov2"
             and r["configuration"] == "A_only__published")["empty_accuracy"]
    b = next(r for r in lift if r["backbone"] == "dinov2"
             and r["configuration"] == "A_plus_B_play_plus_01_empties")["empty_accuracy"]
    record(
        "WP4-T13 what the false-play control measures",
        "`python experiments/empty_recognition.py`",
        f"`{OUT.name}`",
        f"DINOv2 EMPTY accuracy {a:.4f} -> {b:.4f} with ONE labelled empty frame of the "
        f"held-out camera; removing C3 changes nothing - the control is a camera-transfer test",
    )


if __name__ == "__main__":
    main()
