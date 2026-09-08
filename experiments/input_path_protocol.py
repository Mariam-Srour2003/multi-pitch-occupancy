"""Is the input-path finding established, or only observed? (WP3-T3 option (b), RQ2/RQ3)

`geometry_convention_probe.py` found that `build_cache` hands **raw** frames to the HF
processor - which resizes a 1920x1080 frame shortest-edge to 256 and centre-crops 224,
keeping roughly the middle *half* of the pitch - so `preprocess.py` has never touched the
features any headline experiment reads. Letterboxing instead moved ConvNeXtV2's false-play
from 0.9918 to 0.0206 while raising cross-venue recall, which would reverse the production
recommendation.

The probe deliberately stopped there. Its own note says what was missing: *"no CIs, one
seed, no paired test"*, and a false-play column of 243 frames amounting to three to ten
distinct scenes. This is that protocol, and the decision it is meant to inform - adopting
`preprocess.py` in the main cache path, which changes the input to every published number -
is expensive enough to deserve it.

**Nothing here rebuilds a cache.** Both arms already exist: `data/cache/<backbone>.npz` is
the published raw arm and `data/cache/geom_probe/<backbone>.npz` is the letterboxed one, at
the geometry convention WP3-T3 settled on (keep the processor's). The cost is a few minutes
of probe fits.

What it adds, axis by axis
--------------------------

**Cross-venue PLAY recall - seven venues, paired.** The two arms are scored on identical
folds, so the comparison is paired at the level of the *venue*, which is the unit H3
bootstraps over and the unit generalisation is claimed across. Reported with a bootstrap CI
over folds and an exact sign-flip test. Seven pairs put the smallest attainable two-sided p
at **0.0156**, and the test reports that floor beside its p-value so a null result can be
told apart from a design that could not have produced anything else.

**False play - measured in both directions, not one.** The published control trains on
venue_01 camera A and scores camera B's 243 empty frames. Swapping the cameras gives a
second measurement on 251 different frames from a different training set, and an effect
that reverses under the swap is an artefact of one camera pair. This is the only replication
this dataset can offer, because **every EMPTY frame in the corpus is venue_01** - 494 of
them, from two cameras across two slots.

**The effective sample, three ways.** Nominal frames; then distinct scenes at two
near-duplicate thresholds; then the (camera x slot) cells the frames actually come from,
which is **two per direction**. Every frame-level interval below is narrower than the truth
by construction, and the scene-level recomputation says which conclusions survive being
counted honestly.

**The seed axis is checked and then dropped.** `LinearProbe` takes a `seed`, and the probe's
note listed "one seed" among the gaps - but the head is a scikit-learn `LogisticRegression`
on lbfgs, which is deterministic. This verifies that across five seeds every prediction is
identical, so running more of them would have produced identical numbers and dressed a fixed
quantity as a robustness check. That is recorded rather than quietly skipped.

    uv run python experiments/input_path_protocol.py

**Reproduce before extending.** The raw arm is checked against `h3_with_false_play.csv` and
both arms against `geometry_convention_probe.csv` before any new number is computed; a
mismatch aborts. A harness that cannot reproduce what it extends is measuring something
else, and this project has already been saved once by exactly that guard.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.dedup import DEFAULT_THRESHOLD, dhash, distinct_subset
from pitch_occupancy.data.manifest import ManifestRow, read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.stats import (
    bootstrap_ci,
    holm_bonferroni,
    mcnemar,
    sign_flip_test,
)
from pitch_occupancy.vision.heads import LinearProbe

PLAY, EMPTY = "C2_ACTIVE_PLAY", "C1_EMPTY"
SEED = 42
BACKBONES = ("convnextv2", "dinov2", "vit")
ARMS = ("raw", "preproc")

OUT = settings.results_dir / "input_path_protocol.csv"
PUBLISHED_H3 = settings.results_dir / "h3_with_false_play.csv"
PUBLISHED_PROBE = settings.results_dir / "geometry_convention_probe.csv"

#: Where each arm's features live. `raw` is the cache every published number was computed
#: from; `preproc` is the probe's `preproc+geom` arm - `preprocess.py`'s letterbox, then the
#: processor's own geometry, which WP3-T3 measured as the better convention.
CACHE_DIR = {"raw": settings.feature_cache_dir, "preproc": settings.feature_cache_dir / "geom_probe"}

#: How close a reproduced number must be to the published one before this script will
#: extend it. The published CSVs carry four decimals, so 5e-4 is "the same number".
TOLERANCE = 5e-4

#: Near-duplicate thresholds the effective sample is recounted at. Two bits is a strict
#: reading of "the same scene", six is the project default; the answer moves with the
#: choice, so both are reported rather than one being picked.
SCENE_THRESHOLDS = (2, DEFAULT_THRESHOLD)


def physical(row: ManifestRow) -> str:
    """The camera that took the frame, not the slot-and-camera tag naming it.

    The two are not the same: `slot_20260712_2030_camB` is physically camera A. Training on
    the tag would put one physical view on both sides of the false-play control.
    """
    return PHYSICAL_CAMERA.get(row.camera, row.camera)


# --- loading -----------------------------------------------------------------------


def load_arm(backbone: str, arm: str, rows: list[ManifestRow]):
    """Features for one arm, aligned to ``rows``; missing frames are dropped, not filled."""
    path = CACHE_DIR[arm] / f"{backbone}.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {arm} cache for {backbone}: {path}\n"
            "  raw     -> uv run pitch cache\n"
            "  preproc -> uv run python experiments/geometry_convention_probe.py"
        )
    data = np.load(path, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(data["files"])}
    kept = [r for r in rows if r.file in index]
    return kept, data["features"][[index[r.file] for r in kept]]


def assert_arms_differ(backbone: str, features: dict[str, np.ndarray]) -> float:
    """Guard: the two arms must actually be different features.

    Cheap, and it forecloses the failure this experiment would be worst at reporting -
    comparing a cache against itself and concluding the input path does not matter. The
    geometry probe hit the neighbouring version of this: ViT's two conventions are
    bit-identical *and that was the expected answer*, so "identical" is not automatically
    a bug, only automatically worth knowing.
    """
    a, b = features["raw"], features["preproc"]
    if a.shape != b.shape:
        raise SystemExit(f"{backbone}: arms have different shapes {a.shape} vs {b.shape}")
    diff = float(np.abs(a - b).max())
    if diff == 0.0:
        raise SystemExit(
            f"{backbone}: the raw and preproc caches are bit-identical, so there is "
            "nothing to compare - check that geom_probe/ holds the letterboxed build"
        )
    return diff


# --- axis 1: cross-venue PLAY recall ------------------------------------------------


def fold_recalls(rows: list[ManifestRow], X: np.ndarray) -> dict[str, float]:
    """ACTIVE_PLAY recall per held-out venue, on the folds H3 uses.

    The `venue_01` fold is skipped for the reason H3 skips it: it trains on a single class,
    so its "recall" measures nothing. Seven folds, not eight.
    """
    pos = {r.file: i for i, r in enumerate(rows)}
    out: dict[str, float] = {}
    for fold in leave_one_group_out(rows):
        play = [r for r in fold.test if r.class3 == PLAY]
        if not play or len({r.class3 for r in fold.train}) < 2:
            continue
        head = LinearProbe("probe", seed=SEED).fit(X[[pos[r.file] for r in fold.train]], fold.train)
        pred = head.predict(X[[pos[r.file] for r in play]], play)
        out[fold.name] = float(np.mean([p == PLAY for p in pred]))
    return out


# --- axis 2: false play, in both directions -----------------------------------------


def false_play_correctness(
    rows: list[ManifestRow], X: np.ndarray, *, train_camera: str
) -> tuple[np.ndarray, list[ManifestRow]]:
    """Per-frame "did not say PLAY" on the *other* camera's empty frames.

    Correctness rather than a rate, because McNemar needs the per-frame vector and a rate
    throws away the pairing that makes the test worth running.
    """
    test_camera = "camera_B" if train_camera == "camera_A" else "camera_A"
    pos = {r.file: i for i, r in enumerate(rows)}
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if physical(r) == train_camera]
    empties = [r for r in venue if physical(r) == test_camera and r.class3 == EMPTY]
    if not empties or len({r.class3 for r in train}) < 2:
        raise SystemExit(f"no usable false-play split training on {train_camera}")
    head = LinearProbe("probe", seed=SEED).fit(X[[pos[r.file] for r in train]], train)
    pred = head.predict(X[[pos[r.file] for r in empties]], empties)
    return np.array([p != PLAY for p in pred]), empties


def training_composition(rows: list[ManifestRow], *, train_camera: str) -> dict[str, int]:
    """What the direction actually trains on. The swap is a replication, not a mirror.

    All six C3 frames sit on camera A, so training on camera B is a two-class fit. That is
    a real difference between the directions and is reported rather than smoothed over -
    it is also the reason no third direction exists.
    """
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if physical(r) == train_camera]
    counts = {"n_train": len(train)}
    for row in train:
        counts[row.class3] = counts.get(row.class3, 0) + 1
    return counts


def scene_indices(empties: list[ManifestRow], threshold: int) -> list[int]:
    """Positions of one frame per distinct scene, at a near-duplicate threshold."""
    import cv2

    hashes: dict[int, int] = {}
    for i, row in enumerate(empties):
        image = cv2.imread(str(settings.dataset_dir / row.file))
        if image is not None:
            hashes[i] = dhash(image)
    return distinct_subset(hashes, threshold=threshold)


def camera_slot_cells(empties: list[ManifestRow]) -> int:
    """How many (camera x slot) cells the held-out empties come from.

    The coarsest and least arguable effective-sample count: it needs no hash threshold and
    no judgement about what "the same scene" means. On this corpus it is **two**.
    """
    return len({(physical(r), r.slot_id) for r in empties})


# --- the seed axis, checked rather than assumed --------------------------------------


def seed_changes_anything(rows: list[ManifestRow], X: np.ndarray, seeds=(0, 1, 42, 7, 12345)) -> bool:
    """Does `LinearProbe`'s seed move a single prediction? On lbfgs it does not."""
    pos = {r.file: i for i, r in enumerate(rows)}
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if physical(r) == "camera_A"]
    test = [r for r in venue if physical(r) == "camera_B"]
    seen = {
        tuple(
            LinearProbe("probe", seed=s)
            .fit(X[[pos[r.file] for r in train]], train)
            .predict(X[[pos[r.file] for r in test]], test)
        )
        for s in seeds
    }
    return len(seen) > 1


# --- reproduction guards --------------------------------------------------------------


def _published(path: Path, key_fields: tuple[str, ...], value_field: str) -> dict[tuple, float]:
    if not path.exists():
        return {}
    out: dict[tuple, float] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        raw = row.get(value_field, "")
        if raw:
            out[tuple(row[k] for k in key_fields)] = float(raw)
    return out


def check_reproduces(label: str, computed: dict[tuple, float], published: dict[tuple, float]) -> int:
    """Abort unless every number this run recomputes matches the committed one."""
    shared = sorted(set(computed) & set(published))
    if not shared:
        print(f"  {label}: nothing to check against - published table absent or renamed")
        return 0
    bad = [(k, computed[k], published[k]) for k in shared if abs(computed[k] - published[k]) > TOLERANCE]
    for key, got, want in bad:
        print(f"  MISMATCH {label} {key}: computed {got:.4f}, published {want:.4f}")
    if bad:
        raise SystemExit(
            f"{label}: {len(bad)} of {len(shared)} published numbers did not reproduce. "
            "Nothing new is computed on a harness that disagrees with the table it extends."
        )
    print(f"  {label}: {len(shared)} published numbers reproduce to {TOLERANCE:g}")
    return len(shared)


# --- the run ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resamples", type=int, default=10_000, help="bootstrap resamples")
    ap.add_argument("--skip-scene-audit", action="store_true",
                    help="skip the near-duplicate recount, which reads every empty frame")
    args = ap.parse_args()

    all_rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    print(f"development rows: {len(all_rows)}")
    print(f"arms: raw={CACHE_DIR['raw']}  preproc={CACHE_DIR['preproc']}\n")

    loaded: dict[tuple[str, str], tuple[list[ManifestRow], np.ndarray]] = {}
    recalls: dict[tuple[str, str], dict[str, float]] = {}
    correct: dict[tuple[str, str, str], np.ndarray] = {}
    empties_for: dict[str, list[ManifestRow]] = {}

    for backbone in BACKBONES:
        feats = {}
        for arm in ARMS:
            rows, X = load_arm(backbone, arm, all_rows)
            if len(rows) != len(all_rows):
                print(f"  {backbone}/{arm}: {len(all_rows) - len(rows)} row(s) uncached - dropped")
            loaded[(backbone, arm)] = (rows, X)
            feats[arm] = X
        spread = assert_arms_differ(backbone, feats)
        print(f"{backbone}: arms differ, max |raw - preproc| = {spread:.3f}")

        for arm in ARMS:
            rows, X = loaded[(backbone, arm)]
            recalls[(backbone, arm)] = fold_recalls(rows, X)
            for train_camera in ("camera_A", "camera_B"):
                ok, empties = false_play_correctness(rows, X, train_camera=train_camera)
                correct[(backbone, arm, train_camera)] = ok
                empties_for[train_camera] = empties

    # ---- the seed axis -----------------------------------------------------------
    rows, X = loaded[("dinov2", "raw")]
    seed_matters = seed_changes_anything(rows, X)
    print(
        "\nseed axis: LinearProbe's seed changes "
        + ("some predictions - vary it" if seed_matters else "NO prediction across five seeds")
        + " (lbfgs is deterministic; 'one seed' is not a gap that more seeds would close)"
    )

    # ---- reproduction guards -----------------------------------------------------
    print("\nreproducing the published tables before extending them")
    computed_h3 = {
        (backbone, fold): value
        for (backbone, arm), by_fold in recalls.items() if arm == "raw"
        for fold, value in by_fold.items()
    }
    check_reproduces(
        "h3_with_false_play (raw recall)",
        computed_h3,
        _published(PUBLISHED_H3, ("model", "held_out_venue"), "play_recall"),
    )
    computed_probe = {
        (b, {"raw": "raw+geom", "preproc": "preproc+geom"}[arm], "MEAN_ACROSS_FOLDS"):
        float(np.mean(list(folds.values())))
        for (b, arm), folds in recalls.items()
    }
    check_reproduces(
        "geometry_convention_probe (mean recall)",
        computed_probe,
        _published(PUBLISHED_PROBE, ("backbone", "arm", "held_out_venue"), "play_recall"),
    )
    computed_fp = {
        (b, {"raw": "raw+geom", "preproc": "preproc+geom"}[arm], "MEAN_ACROSS_FOLDS"):
        float(np.mean(~correct[(b, arm, "camera_A")]))
        for b in BACKBONES for arm in ARMS
    }
    check_reproduces(
        "geometry_convention_probe (false play)",
        computed_fp,
        _published(PUBLISHED_PROBE, ("backbone", "arm", "held_out_venue"), "false_play_rate"),
    )

    records: list[dict] = []

    # ---- axis 1: cross-venue recall, paired over venues --------------------------
    folds = sorted(recalls[(BACKBONES[0], "raw")])
    print(f"\n=== cross-venue PLAY recall: {len(folds)} venue folds, paired ===")
    print(f"{'backbone':12} {'raw':>7} {'preproc':>8} {'delta':>8} "
          f"{'95% CI over venues':>22} {'p':>8} {'floor':>7} {'wins':>6}")

    recall_tests = []
    for backbone in BACKBONES:
        raw = np.array([recalls[(backbone, "raw")][f] for f in folds])
        pre = np.array([recalls[(backbone, "preproc")][f] for f in folds])
        delta = pre - raw
        ci = bootstrap_ci(delta, np.mean, resamples=args.resamples, seed=SEED)
        test = sign_flip_test(delta)
        recall_tests.append((backbone, ci, test, delta))
        wins = f"{int((delta > 0).sum())}/{int((delta != 0).sum())}"
        print(f"{backbone:12} {raw.mean():7.4f} {pre.mean():8.4f} {ci.estimate:+8.4f} "
              f"[{ci.low:+.4f}, {ci.high:+.4f}] {test.p_value:8.4f} "
              f"{test.min_achievable_p:7.4f} {wins:>6}")
        for fold, r, p in zip(folds, raw, pre, strict=True):
            for arm, value in (("raw", r), ("preproc", p)):
                records.append({
                    "axis": "cross_venue_recall", "backbone": backbone,
                    "unit_key": fold.split("__")[-1], "measure": "fold_recall", "arm": arm,
                    "estimate": f"{value:.4f}", "ci_low": "", "ci_high": "",
                    "p_value": "", "p_holm": "", "significant": "",
                    "n_units": 1, "unit": "venue",
                })

    adj, rej = holm_bonferroni([t.p_value for _, _, t, _ in recall_tests])
    for (backbone, ci, test, _), p_adj, sig in zip(recall_tests, adj, rej, strict=True):
        records.append({
            "axis": "cross_venue_recall", "backbone": backbone, "unit_key": "ALL_VENUES",
            "measure": "delta_preproc_minus_raw", "arm": "preproc-raw",
            "estimate": f"{ci.estimate:+.4f}", "ci_low": f"{ci.low:+.4f}", "ci_high": f"{ci.high:+.4f}",
            "p_value": f"{test.p_value:.4f}", "p_holm": f"{p_adj:.4f}", "significant": sig,
            "n_units": test.n_pairs, "unit": "venue",
        })
    print(f"  Holm over {len(recall_tests)} backbones: "
          + ", ".join(f"{b} p={p:.4f}{'*' if s else ''}"
                      for (b, _, _, _), p, s in zip(recall_tests, adj, rej, strict=True)))
    worst_floor = max(t.min_achievable_p for _, _, t, _ in recall_tests)
    if worst_floor > 0.05:
        needed = math.ceil(math.log2(2 / 0.05))
        print(f"  NOTE: the floor is {worst_floor:.4f} - no result on this axis could have")
        print(f"  reached p<0.05. Two folds tie for every backbone, so seven venues carry")
        print(f"  five informative pairs; {needed} all pointing one way is the minimum that can.")

    # ---- axis 2: false play, both directions -------------------------------------
    print("\n=== false play on held-out EMPTY frames, both camera directions ===")
    print(f"{'backbone':12} {'train on':9} {'raw':>7} {'preproc':>8} {'delta':>8} "
          f"{'95% CI (frames)':>22} {'McNemar p':>10} {'n':>5}")

    fp_tests = []
    for backbone in BACKBONES:
        for train_camera in ("camera_A", "camera_B"):
            ok_raw = correct[(backbone, "raw", train_camera)]
            ok_pre = correct[(backbone, "preproc", train_camera)]
            fp_raw, fp_pre = 1.0 - ok_raw.mean(), 1.0 - ok_pre.mean()
            ci = bootstrap_ci(
                (1.0 - ok_pre.astype(float)) - (1.0 - ok_raw.astype(float)),
                np.mean, resamples=args.resamples, seed=SEED,
            )
            m = mcnemar(ok_raw, ok_pre)
            fp_tests.append((backbone, train_camera, fp_raw, fp_pre, ci, m, ok_raw.size))
            print(f"{backbone:12} {train_camera:9} {fp_raw:7.4f} {fp_pre:8.4f} "
                  f"{ci.estimate:+8.4f} [{ci.low:+.4f}, {ci.high:+.4f}] "
                  f"{m.p_value:10.2e} {ok_raw.size:5d}")

    adj, rej = holm_bonferroni([m.p_value for *_, m, _ in fp_tests])
    for (backbone, cam, fp_raw, fp_pre, ci, m, n), p_adj, sig in zip(fp_tests, adj, rej, strict=True):
        for arm, value in (("raw", fp_raw), ("preproc", fp_pre)):
            records.append({
                "axis": "false_play", "backbone": backbone, "unit_key": f"train_{cam}",
                "measure": "rate", "arm": arm, "estimate": f"{value:.4f}",
                "ci_low": "", "ci_high": "", "p_value": "", "p_holm": "", "significant": "",
                "n_units": n, "unit": "frame",
            })
        records.append({
            "axis": "false_play", "backbone": backbone, "unit_key": f"train_{cam}",
            "measure": "delta_preproc_minus_raw", "arm": "preproc-raw",
            "estimate": f"{ci.estimate:+.4f}", "ci_low": f"{ci.low:+.4f}", "ci_high": f"{ci.high:+.4f}",
            "p_value": f"{m.p_value:.3e}", "p_holm": f"{p_adj:.3e}", "significant": sig,
            "n_units": n, "unit": "frame",
        })

    by_direction = {(b, cam): ci.estimate for b, cam, _, _, ci, _, _ in fp_tests}

    def replication(backbone: str) -> str:
        """How the two camera directions agree - in three states, not two.

        A backbone that measures exactly zero in one direction has no effect there to
        replicate or reverse, and calling that a reversal would inflate the failure count
        as surely as calling it agreement would inflate the success count.
        """
        a, b = by_direction[(backbone, "camera_A")], by_direction[(backbone, "camera_B")]
        if a == 0.0 or b == 0.0:
            return "no effect one side"
        return "replicates" if np.sign(a) == np.sign(b) else "REVERSES"

    agreement = {b: replication(b) for b in BACKBONES}
    replicates = sum(v == "replicates" for v in agreement.values())
    reverses = sum(v == "REVERSES" for v in agreement.values())
    print(f"\n  direction replication: of {len(BACKBONES)} backbones the effect keeps its sign "
          f"for {replicates}, reverses for {reverses}, and is absent on one side for "
          f"{len(BACKBONES) - replicates - reverses}")

    # The swap is a replication, not a mirror, and the asymmetry belongs in the output
    # rather than in a footnote: all six C3 frames sit on camera A, so training on camera
    # B is a two-class fit on two thirds the data.
    reference_rows, _ = loaded[("dinov2", "raw")]
    for train_camera in ("camera_A", "camera_B"):
        comp = training_composition(reference_rows, train_camera=train_camera)
        classes = {k: v for k, v in comp.items() if k != "n_train"}
        print(f"  training on {train_camera}: {comp['n_train']} frames, "
              + ", ".join(f"{k.split('_')[0]}={v}" for k, v in sorted(classes.items())))
        records.append({
            "axis": "false_play", "backbone": "", "unit_key": f"train_{train_camera}",
            "measure": "n_train", "arm": "", "estimate": comp["n_train"],
            "ci_low": "", "ci_high": "", "p_value": "", "p_holm": "", "significant": "",
            "n_units": len(classes), "unit": "class",
        })

    # ---- the effective sample ------------------------------------------------------
    cells_seen = 0
    if not args.skip_scene_audit:
        print("\n=== how many independent observations are those frames? ===")
        for train_camera in ("camera_A", "camera_B"):
            empties = empties_for[train_camera]
            cells = camera_slot_cells(empties)
            cells_seen = max(cells_seen, cells)
            print(f"\n  training on {train_camera}: {len(empties)} held-out empty frames, "
                  f"from {cells} (camera x slot) cell(s)")
            for threshold in SCENE_THRESHOLDS:
                picks = scene_indices(empties, threshold)
                line = f"    at {threshold} bits: {len(picks)} distinct scene(s)"
                if len(picks) < 3:
                    print(line + " - too few to test")
                    records.append({
                        "axis": "false_play", "backbone": "", "unit_key": f"train_{train_camera}",
                        "measure": f"distinct_scenes@{threshold}", "arm": "", "estimate": len(picks),
                        "ci_low": "", "ci_high": "", "p_value": "", "p_holm": "",
                        "significant": "", "n_units": len(empties), "unit": "frame",
                    })
                    continue
                ps, keys = [], []
                for backbone in BACKBONES:
                    a = correct[(backbone, "raw", train_camera)][picks]
                    b = correct[(backbone, "preproc", train_camera)][picks]
                    ps.append(mcnemar(a, b).p_value)
                    keys.append(backbone)
                p_adj, sig = holm_bonferroni(ps)
                print(line + "; McNemar after Holm: "
                      + ", ".join(f"{k} p={p:.3f}{'*' if s else ''}"
                                  for k, p, s in zip(keys, p_adj, sig, strict=True)))
                for k, p, s in zip(keys, p_adj, sig, strict=True):
                    records.append({
                        "axis": "false_play", "backbone": k, "unit_key": f"train_{train_camera}",
                        "measure": f"delta_on_distinct_scenes@{threshold}", "arm": "preproc-raw",
                        "estimate": "", "ci_low": "", "ci_high": "", "p_value": "",
                        "p_holm": f"{p:.3e}", "significant": s,
                        "n_units": len(picks), "unit": "distinct_scene",
                    })

    records.append({
        "axis": "method", "backbone": "", "unit_key": "", "measure": "probe_seed_changes_predictions",
        "arm": "", "estimate": str(seed_matters), "ci_low": "", "ci_high": "",
        "p_value": "", "p_holm": "", "significant": "", "n_units": 5, "unit": "seed",
    })

    # ---- the balanced score, as description only ----------------------------------
    print("\n=== balanced score (recall - false play), point estimates only ===")
    print("  The two axes have different sampling units - seven venues and one pair of")
    print("  cameras - so their difference has no honest interval and none is given.")
    for backbone in BACKBONES:
        raw_r = float(np.mean(list(recalls[(backbone, 'raw')].values())))
        pre_r = float(np.mean(list(recalls[(backbone, 'preproc')].values())))
        raw_f = float(np.mean(~correct[(backbone, 'raw', 'camera_A')]))
        pre_f = float(np.mean(~correct[(backbone, 'preproc', 'camera_A')]))
        print(f"  {backbone:12} raw {raw_r - raw_f:+.4f}   preproc {pre_r - pre_f:+.4f}   "
              f"delta {(pre_r - pre_f) - (raw_r - raw_f):+.4f}")
        records.append({
            "axis": "balanced_score", "backbone": backbone, "unit_key": "train_camera_A",
            "measure": "delta_preproc_minus_raw", "arm": "preproc-raw",
            "estimate": f"{(pre_r - pre_f) - (raw_r - raw_f):+.4f}",
            "ci_low": "", "ci_high": "", "p_value": "", "p_holm": "", "significant": "",
            "n_units": "", "unit": "mixed",
        })

    # ---- the verdict, derived rather than written -----------------------------------
    #
    # Every sentence below is computed from the numbers above. The most-quoted figure in
    # this project was once a plot with its ranks and scores hardcoded, contradicting its
    # own source table while the reproduction stage reported success; a summary that
    # restates a conclusion typed in by hand is the same defect in prose.
    print("\n=== does the input-path finding survive the protocol? ===")
    survived = [b for (b, _, t, _) in recall_tests if t.p_value <= 0.05]
    print(f"  recall  : {len(survived)} of {len(BACKBONES)} backbones show a significant gain "
          f"(floor {worst_floor:.4f}, so {'none could' if worst_floor > 0.05 else 'they could'})")
    print(f"  false play: the effect keeps its sign under the camera swap for "
          f"{replicates} of {len(BACKBONES)} backbones")
    for backbone in BACKBONES:
        a, b = by_direction[(backbone, "camera_A")], by_direction[(backbone, "camera_B")]
        print(f"    {backbone:12} train A {a:+.4f}   train B {b:+.4f}   {agreement[backbone]}")
    if not args.skip_scene_audit:
        print(f"  effective sample: {cells_seen} (camera x slot) cell(s) per direction, so the")
        print("    frame-level intervals above are narrower than the truth by construction")
    if replicates < len(BACKBONES) or worst_floor > 0.05:
        print("\n  The finding is OBSERVED, not ESTABLISHED. Adopting `preprocess.py` in the")
        print("  main cache path (WP3-T3 option (b)) changes the input to every published")
        print("  number, and this evidence does not carry that decision.")
    else:
        print("\n  The finding survives every check applied here.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\nwrote {OUT.name}")

    # The log is an index, so an entry is written once per day's run rather than once per
    # invocation. Debugging this script appended the same line three times before the guard
    # existed, which is how a log stops being read.
    log = settings.results_dir / "EXPERIMENT_LOG.md"
    entry = (
        f"- {datetime.now(timezone.utc):%Y-%m-%d} | WP3-T3(b) input path under protocol | "
        f"`python experiments/input_path_protocol.py` | `{OUT.name}` | "
        f"paired over {len(folds)} venues + both camera directions, with the effective sample"
    )
    if entry not in log.read_text(encoding="utf-8"):
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n{entry}\n")


if __name__ == "__main__":
    main()
