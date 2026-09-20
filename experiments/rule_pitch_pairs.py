"""Does the pitch-level sum pay back what a single camera loses? (A36, WP9-T6)

`rule_frame_eval.py` measured the cost A36 registered: scored one camera at a time,
venue_01's 278 recorded play frames read as PLAYING only **27.5%** of the time, against the
probe's 100%. venue_01's pitch has two cameras and each sees about half of it, so a count
inside one camera's boundary is routinely under five on a frame whose pitch plainly holds a
match - A16 measured 88 of 278 from the other side, and the hand-count audit 18 of 74.

`slots/fusion.fuse_pitch` exists to pay that back: the counts are summed across the pitch's
cameras and the table is applied once, to the pitch. This measures whether it does, on the one
pitch in the corpus that has two cameras.

**The pairing is by recording and timestamp, not by guessing.** A frame is named
``slot_<date>_<time>_<camA|camB>_t<seconds>``, so the two halves of one moment are the two
files that agree on everything but the camera. Only exact pairs are used; an unpaired frame is
reported and dropped rather than fused with its nearest neighbour, because "the two halves of
this moment" is the claim being tested and a near-miss is a different claim.

**This is not the slot pipeline.** It is the fusion step alone, on labelled stills, with no
burst and no temporal smoothing - so it isolates what summing the cameras is worth.
`rule_slots.py` runs the whole path over the four recordings.

    uv run python experiments/rule_pitch_pairs.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict

import cv2
import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.evaluation.stats import bootstrap_ci
from pitch_occupancy.slots.fusion import fuse_pitch
from pitch_occupancy.vision import roi
from pitch_occupancy.vision.rules import from_label

#: `2_playing/slot_20260712_2030_camA_t003476.jpg` -> recording, camera, seconds
NAME = re.compile(r"(?P<slot>slot_\d{8}_\d{4})_(?P<cam>cam[AB])_t(?P<t>\d+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="detector key; the registry default if unset")
    ap.add_argument("--label", default=None, help="only this labelling folder")
    args = ap.parse_args()

    from pitch_occupancy.pipeline import assemble
    from pitch_occupancy.vision.detector import DEFAULT_DETECTOR, DETECTORS

    pipeline = assemble(args.model or DEFAULT_DETECTOR, require_boundary=False)
    if pipeline.kind != "detector":
        raise SystemExit(f"{pipeline.model_key!r} is a {pipeline.kind}; "
                         f"pass --model from {sorted(DETECTORS)}")
    print(pipeline.describe())

    rows = [r for r in development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
            if r.source != "synthetic" and r.venue == "venue_01"
            and (args.label is None or r.label == args.label)]

    moments: dict[tuple[str, str, str], dict[str, object]] = defaultdict(dict)
    unmatched = 0
    for row in rows:
        m = NAME.search(row.file)
        if not m:
            unmatched += 1
            continue
        moments[(m["slot"], m["t"], row.label)][m["cam"]] = row
    pairs = {k: v for k, v in moments.items() if len(v) == 2}
    singles = len(moments) - len(pairs)
    print(f"{len(rows)} venue_01 frames -> {len(pairs)} paired moments "
          f"({singles} moments with only one camera, dropped; {unmatched} unparseable)\n")
    if not pairs:
        print("no paired moments; nothing to fuse")
        return 1

    records, alone_hits, fused_hits = [], [], []
    for n, ((slot, t, truth), cams) in enumerate(sorted(pairs.items()), 1):
        verdicts, counts = {}, {}
        for cam, row in cams.items():
            frame = cv2.imread(str(settings.dataset_dir / row.file))
            if frame is None:
                break
            verdict = pipeline.classify_frame(frame, camera_id=row.camera,
                                              polygon=roi.resolve(row.camera))
            verdicts[cam] = verdict
            counts[cam] = verdict.people
        if len(verdicts) != 2:
            continue
        pitch = fuse_pitch(verdicts, pipeline.rules)
        # The labelling folder, collapsed onto the three classes the rule answers:
        # `3_people_not_playing` and `4_maintenance` are both C3 since A40.
        want = from_label(truth)
        for v in verdicts.values():
            alone_hits.append(int(v.state is want))
        fused_hits.append(int(pitch.state is want))
        records.append({
            "slot": slot, "t_s": int(t), "truth": truth,
            "camA_state": verdicts["camA"].state.value, "camA_people": counts["camA"],
            "camB_state": verdicts["camB"].state.value, "camB_people": counts["camB"],
            "pitch_people": pitch.people_inside, "pitch_state": pitch.state.value,
            "pitch_confidence": pitch.confidence, "ball": pitch.ball_seen,
            "correct_alone_camA": verdicts["camA"].state is want,
            "correct_alone_camB": verdicts["camB"].state is want,
            "correct_fused": pitch.state is want,
        })
        if n % 50 == 0:
            print(f"  {n}/{len(pairs)}", end="\r", flush=True)
    print(f"  {len(records)}/{len(pairs)} moments fused      \n")

    by_truth: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_truth[r["truth"]].append(r)

    print(f"{'label':<24}{'moments':>9}{'one camera':>13}{'pitch sum':>12}{'gain':>9}")
    for truth, group in sorted(by_truth.items()):
        alone = float(np.mean([r["correct_alone_camA"] for r in group]
                              + [r["correct_alone_camB"] for r in group]))
        fused = float(np.mean([r["correct_fused"] for r in group]))
        print(f"{truth:<24}{len(group):>9}{alone:>13.4f}{fused:>12.4f}{fused - alone:>+9.4f}")
    alone_all, fused_all = float(np.mean(alone_hits)), float(np.mean(fused_hits))
    ci = bootstrap_ci(fused_hits)
    print(f"{'ALL':<24}{len(records):>9}{alone_all:>13.4f}{fused_all:>12.4f}"
          f"{fused_all - alone_all:>+9.4f}   95% CI [{ci.low:.3f}, {ci.high:.3f}]")

    play = by_truth.get("2_playing", [])
    if play:
        median_alone = float(np.median([r["camA_people"] or 0 for r in play]
                                       + [r["camB_people"] or 0 for r in play]))
        median_sum = float(np.median([r["pitch_people"] or 0 for r in play]))
        print(f"\non the {len(play)} play moments the median count is {median_alone:.0f} "
              f"inside one camera's boundary and {median_sum:.0f} across the pitch; the rule's "
              f"threshold is {pipeline.rules.play_min}.")

    out = settings.results_dir / "rule_pitch_pairs.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nwrote {out}")
    print("one pitch, two cameras, labelled stills - no burst and no temporal smoothing, so "
          "this\nisolates what summing the cameras is worth. rule_slots.py runs the whole path.")

    record(
        "WP9-T6 pitch-level sum against a single camera",
        "uv run python experiments/rule_pitch_pairs.py",
        "rule_pitch_pairs.csv",
        f"{len(records)} paired moments at venue_01: one camera {alone_all:.3f} correct, the "
        f"pitch sum {fused_all:.3f} [{ci.low:.3f}, {ci.high:.3f}]"
        + (f"; play moments median {median_alone:.0f} inside one camera and {median_sum:.0f} "
           f"across the pitch against a threshold of {pipeline.rules.play_min}" if play else ""),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
