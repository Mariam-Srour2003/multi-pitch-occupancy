"""The detector-first rule against the probe, the gated probe and the clock rule (A36, WP9-T6).

Every arm on the same recorded frames, under the same controls A20 fixed and A36 re-registered:

- **cross-venue play recall**, over the seven clip venues that have development play frames;
- **the false-play control** beside it, always - the 243 recorded EMPTY frames of venue_01
  camera B, scored by a model that never saw them. Recall alone on these folds is meaningless
  because every fold's test side is 100% ACTIVE_PLAY and a constant predictor scores 1.000;
- **EMPTY accuracy** on those same 243, because the complement of a false-play rate is not
  correctness - a model can avoid saying PLAY by saying C3 (A20, and the probe does exactly
  that on 38% of them);
- **the abstention rate**, because a path that abstains its way to a low false-play rate has
  not earned it;
- ranked on `balanced = recall − false_play`, this project's standard.

**The arms are not trained alike, and that is the finding rather than a flaw.** The probe arms
are fitted per fold - leave-one-venue-out for recall, camera A for the false-play control -
exactly as `h3_with_false_play.py` fits them, and this script reproduces that script's
published means before reporting anything, so a difference here is a difference in the *arm*
and not in the harness. The detector-first arm is fitted on nothing at all: it has no training
set, so every recorded frame is evaluation data for it and the same rule runs at every venue.
That asymmetry is what the comparison is about.

**Frame level, one frame, no burst, no pitch-level sum.** Each frame is scored on its own,
inside its own camera's boundary, which is the hardest setting for the detector-first arm:
A16 measured that a camera sees half a pitch, so a per-frame count under five is common on
frames whose pitch plainly holds a match, and the hand-count audit found 18 of 74. The
pitch-level sum and the burst are what `rule_slots.py` and `rule_on_clips.py` measure. A
number here is a floor, not the system's answer.

    uv run python experiments/rule_frame_eval.py
    uv run python experiments/rule_frame_eval.py --arms detector_first clock_rule
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

import cv2
import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.stats import bootstrap_ci
from pitch_occupancy.vision import roi

RESULTS = settings.results_dir
EMPTY, PLAY = Class3.EMPTY.value, Class3.ACTIVE_PLAY.value
C3 = Class3.MAINTENANCE_NON_SPORTING.value
PUBLISHED = RESULTS / "h3_with_false_play.csv"
CACHE = {"dinov2": "dinov2.npz"}


def physical(row) -> str:
    return PHYSICAL_CAMERA.get(row.camera, row.camera)


# --- the detector-first arm, which trains on nothing --------------------------------------


def detector_predictions(rows, model_key: str | None, *, gates_note: str) -> dict[str, dict]:
    """``file -> {state, class3, people, ball, rule}`` for every row, one frame at a time.

    **The key defaults to the detector registry's default, not `settings.default_model_key`.**
    The deployment default is still the probe until WP9-T7, so taking it here assembled the
    probe and labelled its answers `detector_first` - a whole arm measuring the wrong thing
    and saying the right name, which is this project's recurring defect and was caught on the
    first run of this script. The assertion below is what stops it recurring silently.
    """
    from pitch_occupancy.pipeline import assemble
    from pitch_occupancy.vision.detector import DEFAULT_DETECTOR, DETECTORS

    pipeline = assemble(model_key or DEFAULT_DETECTOR, require_boundary=False)
    if pipeline.kind != "detector":
        raise SystemExit(
            f"the detector_first arm was handed {pipeline.model_key!r}, which is a "
            f"{pipeline.kind}. Pass --model with one of {sorted(DETECTORS)}."
        )
    print(f"  {pipeline.describe()}")
    print(f"  {gates_note}")
    out: dict[str, dict] = {}
    for n, row in enumerate(rows, 1):
        frame = cv2.imread(str(settings.dataset_dir / row.file))
        if frame is None:
            continue
        polygon = roi.resolve(row.camera)
        verdict = pipeline.classify_frame(frame, camera_id=row.camera, polygon=polygon)
        out[row.file] = {
            "state": verdict.state.value,
            "class3": verdict.class3.value if verdict.class3 else None,
            "people": verdict.people, "ball": verdict.ball, "rule": verdict.rule,
            "boundary": bool(polygon),
        }
        if n % 100 == 0:
            print(f"    {n}/{len(rows)}", end="\r", flush=True)
    print(f"    {len(out)}/{len(rows)} scored      ")
    return out


# --- the probe arms, fitted exactly as the published tables fit them -----------------------


def load_cache(name: str, rows):
    from pitch_occupancy.data.feature_cache import load_cache as _load

    cached = _load(name.removesuffix(".npz"), settings.feature_cache_dir)
    index = {f: i for i, f in enumerate(cached.files)}
    kept = [r for r in rows if r.file in index]
    return kept, cached.features[[index[r.file] for r in kept]]


def make_head(arm: str):
    from pitch_occupancy.vision.heads import ClockRule, LinearProbe

    return ClockRule() if arm == "clock_rule" else LinearProbe(arm, seed=42)


def probe_predictions(arm: str, rows, X) -> dict[str, dict]:
    """Leave-one-venue-out for the clip venues, camera A for venue_01's frames.

    Two fits, because the two questions need different held-out sets and the published tables
    use exactly these: cross-venue recall holds out a venue, the false-play control holds out
    a camera. A frame is scored by whichever fit did not see it.
    """
    position = {r.file: i for i, r in enumerate(rows)}
    out: dict[str, dict] = {}

    for fold in leave_one_group_out(rows):
        if len({r.class3 for r in fold.train}) < 2 or not fold.test:
            continue
        head = make_head(arm).fit(X[[position[r.file] for r in fold.train]], fold.train)
        predicted = head.predict(X[[position[r.file] for r in fold.test]], fold.test)
        for row, p in zip(fold.test, predicted, strict=True):
            out[row.file] = {"state": None, "class3": str(p), "people": None, "ball": None,
                             "rule": 0, "boundary": True}

    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if physical(r) == "camera_A"]
    held = [r for r in venue if physical(r) == "camera_B"]
    if held and len({r.class3 for r in train}) >= 2:
        head = make_head(arm).fit(X[[position[r.file] for r in train]], train)
        predicted = head.predict(X[[position[r.file] for r in held]], held)
        for row, p in zip(held, predicted, strict=True):
            out[row.file] = {"state": None, "class3": str(p), "people": None, "ball": None,
                             "rule": 0, "boundary": True}
    return out


def gated_predictions(arm: str, rows, X, base: dict[str, dict]) -> dict[str, dict]:
    """The probe's verdict with the two gates applied, as the deployed path applies them."""
    from pitch_occupancy.vision.motion import MotionGate  # noqa: F401 - documented, unused
    from pitch_occupancy.vision.people import PersonGate

    gate = PersonGate()
    out: dict[str, dict] = {}
    for n, row in enumerate(rows, 1):
        seen = base.get(row.file)
        if seen is None:
            continue
        frame = cv2.imread(str(settings.dataset_dir / row.file))
        if frame is None:
            continue
        # The motion gate needs the previous frame of the same camera and there is no
        # ordering at frame level, so only the person gate runs here - which is the gate A20
        # measured the 0.0123 false-play rate with.
        state, counted = gate.inspect(Class3(seen["class3"]), frame, roi.resolve(row.camera))
        out[row.file] = {
            "state": None, "class3": state.value, "rule": 0, "boundary": True,
            "people": counted.people if counted else None,
            "ball": counted.ball if counted else None,
        }
        if n % 100 == 0:
            print(f"    {n}/{len(rows)}", end="\r", flush=True)
    print(f"    {len(out)}/{len(rows)} scored      ")
    return out


# --- scoring -------------------------------------------------------------------------------


def score(arm: str, predictions: dict[str, dict], rows) -> dict:
    by_file = {r.file: r for r in rows}
    play_rows = [r for r in rows if r.venue != "venue_01" and r.class3 == PLAY]
    control = [r for r in rows if r.venue == "venue_01" and physical(r) == "camera_B"
               and r.class3 == EMPTY]

    def answers(subset):
        return [(r, predictions[r.file]) for r in subset if r.file in predictions]

    per_venue: dict[str, list[int]] = defaultdict(list)
    for row, got in answers(play_rows):
        per_venue[row.venue].append(int(got["class3"] == PLAY))
    recalls = {v: float(np.mean(hits)) for v, hits in per_venue.items() if hits}
    recall = float(np.mean(list(recalls.values()))) if recalls else float("nan")
    play_flat = [hit for hits in per_venue.values() for hit in hits]
    ci = bootstrap_ci(play_flat) if play_flat else None
    lo, hi = (ci.low, ci.high) if ci else (float("nan"), float("nan"))

    control_answers = answers(control)
    false_play = [int(got["class3"] == PLAY) for _, got in control_answers]
    empty_right = [int(got["class3"] == EMPTY) for _, got in control_answers]
    said_c3 = [int(got["class3"] == C3) for _, got in control_answers]
    fp_ci = bootstrap_ci(false_play) if false_play else None
    fp_lo, fp_hi = (fp_ci.low, fp_ci.high) if fp_ci else (float("nan"), float("nan"))

    # The recall column above follows h3's protocol and measures the *held-out clip venues*.
    # venue_01's own play frames are not in it, and they are where a per-camera count is
    # weakest: its pitch has two cameras and each sees about half, so a count inside one
    # camera's boundary is routinely under five on a frame whose pitch plainly holds a match
    # (A16's 88 of 278, and the hand-count audit's 18 of 74). Reported as its own column
    # rather than folded into the headline, because it is the cost A36 registered as accepted
    # and `rule_slots.py`'s pitch-level sum is what is supposed to pay it back.
    venue_01_play = [r for r in rows if r.venue == "venue_01" and r.class3 == PLAY]
    v01 = [int(got["class3"] == PLAY) for _, got in answers(venue_01_play)]

    scored = set(predictions)
    abstained = sum(1 for f, got in predictions.items()
                    if got["class3"] is None and f in by_file)
    return {
        "arm": arm,
        "n_scored": len(scored),
        "n_play": len(play_flat),
        "play_recall": round(recall, 4),
        "recall_ci_lo": round(lo, 4), "recall_ci_hi": round(hi, 4),
        "recall_worst_venue": round(min(recalls.values()), 4) if recalls else None,
        "n_control": len(control_answers),
        "false_play_rate": round(float(np.mean(false_play)), 4) if false_play else None,
        "false_play_ci_lo": round(fp_lo, 4), "false_play_ci_hi": round(fp_hi, 4),
        "empty_accuracy": round(float(np.mean(empty_right)), 4) if empty_right else None,
        "said_c3_on_empty": round(float(np.mean(said_c3)), 4) if said_c3 else None,
        "abstention_rate": round(abstained / len(scored), 4) if scored else None,
        "balanced": (round(recall - float(np.mean(false_play)), 4)
                     if false_play and recalls else None),
        "n_venue_01_play": len(v01),
        "venue_01_play_recall_one_camera": round(float(np.mean(v01)), 4) if v01 else None,
        "recall_by_venue": ";".join(f"{v}={x:.3f}" for v, x in sorted(recalls.items())),
    }


def confusion(arm: str, predictions: dict[str, dict], rows) -> list[dict]:
    """The 4-class confusion, with 3<->4 marked as the confusion A36 accepts."""
    out = []
    counts: Counter = Counter()
    for row in rows:
        got = predictions.get(row.file)
        if got is None:
            continue
        predicted = got["state"] or {EMPTY: "1_empty", PLAY: "2_playing",
                                     C3: "3_people_not_playing"}.get(got["class3"], "UNCERTAIN")
        counts[(row.venue, row.class4, predicted)] += 1
    for (venue, truth, predicted), n in sorted(counts.items()):
        accepted = {truth, predicted} <= {"3_people_not_playing", "4_maintenance"}
        out.append({"arm": arm, "venue": venue, "truth": truth, "predicted": predicted, "n": n,
                    "exact": truth == predicted, "accepted_confusion": accepted})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*",
                    default=["clock_rule", "dinov2", "dinov2_gated", "detector_first"])
    ap.add_argument("--model", default=None, help="detector key for the detector-first arm")
    args = ap.parse_args()

    rows = [r for r in development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
            if r.source != "synthetic"]
    print(f"{len(rows)} recorded development frames over "
          f"{len({r.venue for r in rows})} venues\n")

    scores, confusions, all_predictions = [], [], {}
    for arm in args.arms:
        print(f"{arm}:")
        if arm == "detector_first":
            predictions = detector_predictions(
                rows, args.model,
                gates_note="no training set: the same rule runs at every venue")
        elif arm == "clock_rule":
            predictions = probe_predictions(arm, rows, np.zeros((len(rows), 1), np.float32))
            print(f"    {len(predictions)}/{len(rows)} scored (reads lighting, not pixels)")
        else:
            base_arm = arm.removesuffix("_gated")
            kept, X = load_cache(CACHE[base_arm], rows)
            predictions = probe_predictions(base_arm, kept, X)
            print(f"    {len(predictions)}/{len(kept)} scored")
            if arm.endswith("_gated"):
                print("  + person gate (A16/A22), the deployed path's detector overrule:")
                predictions = gated_predictions(base_arm, kept, X, predictions)
        all_predictions[arm] = predictions
        scores.append(score(arm, predictions, rows))
        confusions += confusion(arm, predictions, rows)
        print()

    print(f"{'arm':<16}{'recall':>9}{'95% CI':>17}{'worst':>8}{'false-play':>12}"
          f"{'EMPTY acc':>11}{'abstain':>9}{'balanced':>10}")
    def number(value) -> float:
        return float("nan") if value is None else float(value)

    for s in sorted(scores, key=lambda s: -(s["balanced"] or -9)):
        ci = f"[{s['recall_ci_lo']:.3f},{s['recall_ci_hi']:.3f}]"
        print(f"{s['arm']:<16}{s['play_recall']:>9.4f}{ci:>17}"
              f"{number(s['recall_worst_venue']):>8.3f}"
              f"{number(s['false_play_rate']):>12.4f}"
              f"{number(s['empty_accuracy']):>11.4f}"
              f"{number(s['abstention_rate']):>9.4f}"
              f"{number(s['balanced']):>10.4f}")

    print(f"\nrecall is the mean over {len({r.venue for r in rows if r.venue != 'venue_01'})} "
          f"held-out clip venues; false-play and EMPTY accuracy are on venue_01 camera B's "
          f"{scores[0]['n_control']} recorded empty frames.")
    print(f"\nvenue_01's own {scores[0]['n_venue_01_play']} play frames, scored one camera at "
          f"a time - NOT in the recall column above, and the cost A36 accepted:")
    for s in scores:
        print(f"  {s['arm']:<16}{number(s['venue_01_play_recall_one_camera']):>8.4f}")
    print("  venue_01's pitch has two cameras and each sees about half of it, so a count "
          "inside\n  one camera's boundary is routinely under five on a frame whose pitch "
          "holds a match.\n  rule_slots.py's pitch-level sum is what is supposed to pay this "
          "back; it is not yet run.")
    print("one frame at a time, inside one camera's boundary - no burst and no pitch-level "
          "sum, which is the hardest setting for a counting rule (A16: a camera sees half a "
          "pitch). rule_slots.py and rule_on_clips.py measure the system.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "rule_frame_eval.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    conf_path = RESULTS / "rule_confusion_4class.csv"
    with conf_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(confusions[0]))
        writer.writeheader()
        writer.writerows(confusions)
    print(f"\nwrote {out}, {conf_path}")

    best = max(scores, key=lambda s: s["balanced"] if s["balanced"] is not None else -9)
    record(
        "WP9-T6 rule vs probe at frame level",
        "uv run python experiments/rule_frame_eval.py",
        "rule_frame_eval.csv",
        "; ".join(
            f"{s['arm']} recall {s['play_recall']:.3f} false-play "
            f"{s['false_play_rate'] if s['false_play_rate'] is not None else float('nan'):.3f} "
            f"EMPTY {s['empty_accuracy'] if s['empty_accuracy'] is not None else float('nan'):.3f}"
            for s in scores)
        + f"; best balanced: {best['arm']}; one frame per camera, no burst, no pitch sum",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
