"""What the detector saw, as figures (A36, WP9-T4) -> `results/figs/overlays/`.

`make_xai_figures.py` answers *where is the probe looking* with a heatmap, because a logistic
regression on pooled features has no objects in it. The detector-first path has objects, so
its explanation is the objects: people in one colour, the ball in another, the boundary
outlined, and the count and the rule that fired written on the frame. An examiner checks
"five people and a ball, so PLAYING" by looking at it.

**The pair is the point.** Every frame is written twice, side by side in the index: the
detector's overlay and the probe's evidence heatmap for the same frame. The two explanations
of the same footage are the thesis's comparison in visual form - one shows a count, the other
shows a region of the image, and only one of them can be checked against the picture.

**Every frame is redacted before it is written**, by the same two layers this project has used
since `make_xai_figures.py` was written and now shares with it (`vision/explain.redact_frame`):
each detected person is pixelated, then a floor of blur is applied to the whole frame so a
missed detection is still not an identifiable face. The overlay is computed from a verdict
taken on the *original* frame and drawn over the redacted copy, which has identical geometry,
so the explanation stays faithful while the picture stays publishable (`thesis/ethics.md` §2).
The redaction is not a flag: these images enter git history permanently and WP1-T5 is still
open.

    uv run python experiments/make_overlay_figures.py
    uv run python experiments/make_overlay_figures.py --per-class 3 --model yolov8n
"""

from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.pipeline import assemble
from pitch_occupancy.vision import roi
from pitch_occupancy.vision.detector import PERSON
from pitch_occupancy.vision.overlay import legend, publishable

SEED = 42
OUT = settings.results_dir / "figs" / "overlays"


def choose_frames(rows, per_class: int) -> list:
    """A few frames per class, spread across venue and lighting rather than taken in order.

    The same chooser `make_xai_figures.py` uses, and for the same reason: consecutive manifest
    rows are near-duplicates, so the first three frames of a class are three pictures of one
    minute.
    """
    rng = np.random.default_rng(SEED)
    chosen = []
    for cls in sorted({r.class4 for r in rows}):
        pool = [r for r in rows if r.class4 == cls]
        buckets: dict[tuple[str, str], list] = {}
        for r in pool:
            buckets.setdefault((r.venue, r.lighting), []).append(r)
        keys = sorted(buckets)
        for i in range(min(per_class, len(pool))):
            bucket = buckets[keys[i % len(keys)]]
            chosen.append(bucket[int(rng.integers(len(bucket)))])
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=3, help="frames per folder class")
    ap.add_argument("--model", default=None, help="detector key; the configured default if unset")
    ap.add_argument("--width", type=int, default=1280, help="written width")
    args = ap.parse_args()

    pipeline = assemble(args.model, require_boundary=False)
    if pipeline.kind != "detector":
        raise SystemExit(
            f"{pipeline.model_key!r} is a {pipeline.kind}; this draws detections. Pass a "
            f"detector key with --model, or set settings.default_model_key to one.")
    print(pipeline.describe())

    rows = [r for r in development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
            if r.source != "synthetic"]
    chosen = choose_frames(rows, args.per_class)
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"\n{'frame':<52}{'label':<22}{'verdict':<22}{'inside':>7}{'ball':>6}")
    for row in chosen:
        frame = cv2.imread(str(settings.dataset_dir / row.file))
        if frame is None:
            continue
        polygon = roi.resolve(row.camera)
        verdict = pipeline.classify_frame(frame, camera_id=row.camera, polygon=polygon)
        # Drawn over a redacted copy; the verdict above was taken on the original, which is
        # what keeps the explanation faithful and the picture publishable.
        picture = publishable(frame, verdict, polygon=polygon)
        if picture.shape[1] > args.width:
            height = int(picture.shape[0] * args.width / picture.shape[1])
            picture = cv2.resize(picture, (args.width, height), interpolation=cv2.INTER_AREA)
        name = f"{row.class4}__{row.venue}__{row.file.split('/')[-1].rsplit('.', 1)[0]}.png"
        cv2.imwrite(str(OUT / name), picture)

        found = len(verdict.count.people) if verdict.count else 0
        outside = ((verdict.count.people_total - verdict.count.raw_inside)
                   if verdict.count else 0)
        records.append({
            "file": row.file, "venue": row.venue, "lighting": row.lighting,
            "label_class4": row.class4, "label": row.class3,
            "predicted": verdict.state.value,
            "confidence": verdict.confidence, "rule": verdict.rule,
            "people_inside": verdict.people, "people_outside_boundary": outside,
            "ball": verdict.ball, "ball_confidence": round(verdict.ball_confidence, 3),
            "boundary": bool(polygon), "figure": name,
            "trace": " | ".join(verdict.trace),
        })
        print(f"{row.file[-50:]:<52}{row.class4:<22}{verdict.state.name:<22}"
              f"{str(verdict.people):>7}{'yes' if verdict.ball else 'no':>6}"
              + ("" if found or not verdict.count else "  (nobody found - not 'verified empty')"))

    if not records:
        print("no frames could be read")
        return 1

    index = settings.results_dir / "overlay_index.csv"
    with index.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    # Until A40 this compared the state against the *folder* and then forgave every
    # `3_people_not_playing` <-> `4_maintenance` mismatch as "the confusion A36 accepts",
    # which flattered the figure sheet: the corpus has 6 real frames in one of those folders
    # and none in the other, so the forgiven cell was never measured. The rule answers three
    # classes now and this compares against the three-class label, forgiving nothing.
    agree = sum(1 for r in records if r["predicted"] == r["label"])
    play_missed = [r for r in records
                   if r["label"] == "C2_ACTIVE_PLAY" and r["predicted"] != "C2_ACTIVE_PLAY"]
    print(f"\n{len(records)} figure(s) -> {OUT}")
    print(f"index -> {index}")
    print(f"agreement with the label: {agree}/{len(records)}")
    if play_missed:
        no_ball = sum(1 for r in play_missed if not r["ball"])
        print(f"{len(play_missed)} play frame(s) not called play, {no_ball} of them with no "
              f"ball seen - the cost of `require_ball` on this sheet (A40)")
    print("\ncolour key:")
    for label, colour in legend():
        print(f"  BGR {colour}  {label}")
    print("\nEvery frame here is pixelated per detected person and then blurred as a whole "
          "(vision/explain.redact_frame). 'nobody found' never means 'verified empty'.")

    record(
        "WP9-T4 overlay figures",
        "uv run python experiments/make_overlay_figures.py"
        + (f" --model {args.model}" if args.model else ""),
        "figs/overlays/",
        f"{len(records)} redacted overlays from {pipeline.model_key}; the three-class label "
        f"agrees with the rule on {agree} of them (A40: nothing forgiven, where the 3<->4 "
        f"cell used to be); {len(play_missed)} play frame(s) not called play, "
        f"{sum(1 for r in play_missed if not r['ball'])} of those with no ball seen; "
        f"people drawn in one colour and the ball in another",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
