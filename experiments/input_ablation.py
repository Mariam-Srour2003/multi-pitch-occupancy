"""What are the models actually looking at? (WP3-T8, WP4-T6, RQ3/RQ7)

H2 showed a clock rule matching two of three backbones inside the confounded venue; H3
showed the same rule collapsing across venues while the backbones held. Both are indirect.
This asks the question directly by removing information from the input and seeing what
survives:

| variant | what it removes | what it leaves |
|---|---|---|
| `full` | nothing | the baseline |
| `grayscale` | colour - turf hue, floodlight cast, kit colours | shape, texture, people |
| `blur4` | fine detail; people become smudges | scene layout, lighting, geometry |
| `blur8` | almost all object detail | the gross composition of the scene |
| `crop50` | the outer border - stands, sky, adjacent pitches | the central pitch area |
| `gray+crop50` | both of the above | central pitch structure only |

The logic is a dissociation. **If accuracy survives `blur8`, the model cannot be
recognising people** - at that scale nobody is visible - so it must be reading the scene.
If it collapses, the prediction depended on the objects in the frame.

`crop50` is the complement: a crude stand-in for the ROI masking of WP3-T1, which needs
hand-drawn polygons for all nine venues. If discarding the border hurts, the border was
carrying the signal.

Evaluated on the held-out *venues* (H3's protocol), **with the false-play control beside
every row**. Recall alone is not readable on this protocol: the held-out venues contain no
empty pitch, so a variant that answers ACTIVE_PLAY more often earns recall for doing
something worse, and a constant predictor scores 1.000. The control trains on `venue_01`'s
camera A and scores camera B's 243 empty frames — the only held-out empty pitch this dataset
has — and reports `empty_accuracy` beside the rate, because the complement of a false-play
rate is not correctness.

What the control changed (2026-09-11)
-------------------------------------

| variant | recall | false-play | empty accuracy |
|---|---|---|---|
| `full` | 0.960 | 0.230 | 0.770 |
| **`grayscale`** | **0.982** | **0.021** | **0.979** |
| `blur4` | 0.929 | 1.000 | 0.000 |
| `blur8` | 0.840 | 0.926 | 0.074 |
| `crop50` | **0.998** | 0.313 | **0.000** |
| `gray+crop50` | 0.899 | 0.988 | 0.012 |

**`crop50` was the best row in this table and it never identifies an empty pitch.** 0.998
recall with `empty_accuracy` 0.000: its gain is the artefact the axis cannot see, because the
folds hold nothing it could get wrong. Read on recall alone it says *"the border was
redundant"*; read with the control it says the crop moved the model toward PLAY.

**Grayscale is the finding, and it was previously indistinguishable from the artefact.** It
is the only removal that improves *both* axes — recall +0.023 and false-play 0.230 → **0.021**
at 0.979 empty accuracy. Turf hue is a venue cue that does not transfer, and discarding it
helps the model recognise an empty pitch rather than helping it say PLAY.

So the older reading — *"removal has a floor: grayscale helped, the crop helped more, both
together fell below baseline"* — was two-thirds artefact. One removal helps. The second only
appeared to. The combination is worse than either and calls 98.8% of empty pitches a match.

    uv run python experiments/input_ablation.py
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pitch_occupancy.data.feature_cache import preprocessing_hash
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision.backbones import BACKBONES, POOLING_STAMP, embed_batch, load_backbone
from pitch_occupancy.vision.heads import LinearProbe
from pitch_occupancy.vision.preprocess import PreprocessConfig, preprocess

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
SEED = 42
PLAY = "C2_ACTIVE_PLAY"
EMPTY = "C1_EMPTY"
BACKBONE = "dinov2"  # the strongest generaliser from H3

# `grayscale=True` became `saturation=0.0` when the switch was generalised into a dial
# (see PreprocessConfig.saturation). This script kept the old keyword and so raised
# TypeError on import of the module-level dict - it could not run at all, while its
# numbers stayed cited in five places and its caches sat on disk from the run before the
# rename. The variant *names* are deliberately unchanged: they key the cache files those
# numbers came from, so renaming them would orphan the results instead of restoring them.
VARIANTS = {
    "full": PreprocessConfig(),
    "grayscale": PreprocessConfig(saturation=0.0),
    "blur4": PreprocessConfig(blur_sigma=4.0),
    "blur8": PreprocessConfig(blur_sigma=8.0),
    "crop50": PreprocessConfig(centre_crop=0.5),
    # the two individually-helpful removals together: does the benefit compound, or was
    # colour and border carrying the same redundant venue signal?
    "gray+crop50": PreprocessConfig(saturation=0.0, centre_crop=0.5),
}


def embed_variant(rows, name: str, cfg: PreprocessConfig) -> np.ndarray:
    """Embed every frame under one preprocessing variant, cached to disk."""
    path = CACHE / f"ablate_{BACKBONE}_{name}.npz"
    files = [r.file for r in rows]
    # These caches are keyed by *variant name* only. That is fine while the name and the
    # config agree, and silently wrong the moment they drift - which is exactly what
    # happened here: `grayscale=True` was renamed to `saturation=0.0` and this file kept
    # the old keyword, so the config expression changed while the cache name did not.
    # feature_cache.py already solved this for the main caches with a pooling stamp and a
    # preprocessing fingerprint; the same two fields are written here now.
    # `asdict`, not `vars`: `PreprocessConfig` is a slotted dataclass and has no
    # `__dict__`, so `vars()` raises TypeError and this script could not run at all.
    # That is the **second** time it has been un-runnable while its numbers stayed
    # cited - the first was the `grayscale=True` -> `saturation=0.0` rename recorded
    # above. Both times `reproduce_all --check` reported the stage "done", because the
    # CSV it produced years-of-commits ago still existed.
    want = preprocessing_hash(
        backbone=BACKBONES[BACKBONE].hf_id, variant=name, **asdict(cfg)
    )
    if path.exists():
        z = np.load(path, allow_pickle=True)
        idx = {str(f): i for i, f in enumerate(z["files"])}
        if all(f in idx for f in files):
            if "preproc_hash" not in z.files:
                # Written before stamping existed. Accepted, but never silently: the whole
                # point of the stamp is that "I cannot check this" and "I checked this"
                # must not look the same in a log.
                print(
                    f"    WARNING: {path.name} predates preprocessing stamps - reusing it "
                    f"on the strength of its filename alone. Delete it to re-embed under "
                    f"the current config ({want})."
                )
            elif str(z["preproc_hash"]) != want:
                raise SystemExit(
                    f"{path.name} was embedded under preprocessing {str(z['preproc_hash'])} "
                    f"but this run asks for {want}. Mixing preprocessing conventions in one "
                    f"comparison produces plausible, wrong numbers. Delete the file to "
                    f"re-embed."
                )
            elif str(z["pooling"]) != POOLING_STAMP:
                raise SystemExit(
                    f"{path.name} was built with pooling={str(z['pooling'])!r}, expected "
                    f"{POOLING_STAMP!r} - see SUPER_PLAN.md gotcha 2.1."
                )
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
    np.savez_compressed(
        path,
        files=np.array(files, dtype=object),
        features=feats,
        pooling=POOLING_STAMP,
        preproc_hash=want,
    )
    return feats


def false_play_for(rows, X, variant: str) -> tuple[float, float, int]:
    """The control this ablation was reported without: what each variant answers on empty
    pitches it has never seen.

    Returns (false-play rate, EMPTY accuracy, n). The held-out venues contain **no empty
    pitch** - every one of them is 100% active play - so `play_recall` on that protocol is
    earned by answering PLAY more often, and a variant that pushes the model toward PLAY
    scores better for doing something worse. This project has established that twice
    already: a constant predictor takes 1.000 cross-venue (`baseline_floor`), and WP4-T13
    found the false-play rate itself is not a specificity measure. The ablation was reported
    on the same axis without the same control.

    The control is the camera-transfer one WP4-T13 settled on: train on `venue_01`'s camera A
    and score the empty frames of camera B, which is the only held-out empty pitch this
    dataset has. `empty_accuracy` is reported beside the rate, because the complement of the
    rate is not correctness - the third class absorbs the difference.
    """
    pos = {r.file: i for i, r in enumerate(rows)}
    cam = lambda r: PHYSICAL_CAMERA.get(r.camera, r.camera)  # noqa: E731
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if cam(r) == "camera_A"]
    empties = [r for r in venue if cam(r) == "camera_B" and r.class3 == EMPTY]
    if not empties or len({r.class3 for r in train}) < 2:
        return float("nan"), float("nan"), 0
    probe = LinearProbe(f"{variant}_control", seed=SEED).fit(
        X[[pos[r.file] for r in train]], train
    )
    pred = probe.predict(X[[pos[r.file] for r in empties]], empties)
    false_play = float(np.mean([p == PLAY for p in pred]))
    empty_acc = float(np.mean([p == EMPTY for p in pred]))
    return false_play, empty_acc, len(empties)


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
        # The control, on the only held-out empty pitch this dataset has. Reported beside the
        # recall and never under it: read alone, recall on these folds rewards a variant for
        # answering PLAY more often, because every held-out venue is 100% active play.
        fp, empty_acc, n_empty = false_play_for(rows, X, name)
        summary[name] = (float(arr.mean()), float(arr.min()), fp, empty_acc)
        records.append(
            {"variant": name, "venue": "CONTROL_held_out_empty", "n_test": n_empty,
             "false_play_rate": round(fp, 4), "empty_accuracy": round(empty_acc, 4)}
        )
        print(f"  mean play-recall {arr.mean():.3f}   worst {arr.min():.3f}   "
              f"false-play {fp:.3f}   empty-accuracy {empty_acc:.3f}  (n={n_empty})\n")

    base = summary["full"][0]
    base_fp = summary["full"][2]
    print("=== change from the unmodified input ===")
    print("    recall alone is not readable here: the held-out venues hold no empty pitch,")
    print("    so a variant that answers PLAY more often earns recall for doing worse.")
    for name, (mean, _worst, fp, empty_acc) in summary.items():
        if name == "full":
            continue
        delta = mean - base
        fp_delta = fp - base_fp
        # A recall gain that arrives with a false-play gain is the model shifting toward
        # PLAY, which is the thing the axis cannot distinguish from an improvement.
        if delta > 0 and fp_delta > 0.02:
            verdict = "recall bought by answering PLAY more often"
        elif delta > -0.05:
            verdict = "signal survives"
        else:
            verdict = "signal lost"
        print(f"  {name:<12} recall {mean:.3f} ({delta:+.3f})   "
              f"false-play {fp:.3f} ({fp_delta:+.3f})   empty-acc {empty_acc:.3f}   {verdict}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "input_ablation.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["variant", "venue", "n_test", "play_recall", "false_play_rate",
                        "empty_accuracy"],
            restval="",
        )
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {out}")

    record(
        "input ablation",
        "`python experiments/input_ablation.py`",
        f"`{out.name}`",
        f"{len(VARIANTS)} variants x {len(folds)} folds, {BACKBONE}",
    )


if __name__ == "__main__":
    main()
