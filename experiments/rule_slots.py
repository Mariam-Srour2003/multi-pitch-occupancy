"""The whole path over the four recordings, slot verdict included (WP9-T6).

Three experiments cite this file and none of them could run it, because it was never written.
`rule_frame_eval.py` ends with *"rule_slots.py's pitch-level sum is what is supposed to pay
this back; it is not yet run"*, and `rule_pitch_pairs.py` with *"rule_slots.py runs the whole
path"*. Both were describing an intention. This is the file.

**What the two that do exist already settled, so this does not re-measure it.**
`rule_frame_eval` scores one frame inside one camera's boundary - the hardest setting for a
counting rule - and `rule_pitch_pairs` scores the fusion step alone on paired stills. Between
them: requiring a ball cost 69 points of play-recall and bought nothing (both arms identical on
the empty side), and summing the two cameras took venue_01's play moments from 0.7121 to
**1.0000**. What neither touches is the thing an operator actually receives: **a slot verdict**.

**What this adds, and it is the only thing it adds.** `worker.run_slot` over each recording -
per-minute classification, the pitch-level fusion, the capture floor, and
`slots/aggregate.aggregate_slot` turning sixty minutes into USED, NOTUSED or REVIEW. The
question is not whether a frame is classified correctly; it is whether an hour of a pitch
nobody booked comes back NOTUSED.

**The truth is the recording, not a per-frame label.** Two slots, one verdict each:

| slot | labelled frames | what the hour was |
|---|---|---|
| `venue_01_2026-07-11_1000` | 485 EMPTY, 6 PLAY, 6 C3 | an unused morning - **NOTUSED** |
| `venue_01_2026-07-12_2030` | 790 PLAY, 9 EMPTY | a booked evening - **USED** |

The six play frames in the morning are people crossing the pitch, not a match; they are why
the slot thresholds exist rather than a rule that reports any play at all.

**The burst cannot be reproduced from stills, and this says so rather than faking it.** The
rule reads three frames about a second apart (`burst_frames: 3, burst_spacing_s: 1.0`). These
recordings were sampled every **15 seconds**. Three frames 15 s apart are not a burst: the
persistence check becomes "seen in two of three frames across 30 seconds", which is a weaker
claim than the rule was written for, and a motion cue at 15 s is the mismatch `vision/motion.py`
exists to warn about. So the default is **one frame per minute**, with motion unmeasured and
the trace saying so - and `--burst` runs the other way for comparison, labelled at its real
spacing. Neither is the deployed burst; a live camera supplies that and a still corpus cannot.

    uv run python experiments/rule_slots.py
    uv run python experiments/rule_slots.py --arms yolov8n dinov2 --burst
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.pipeline import assemble
from pitch_occupancy.worker import run_slot

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

#: What each recorded hour was, as a booking. Not derived from the frame labels: a slot is
#: USED when the pitch was booked and played on, and six people crossing an empty pitch on
#: their way somewhere does not make a morning a booking.
SLOT_TRUTH: dict[str, SlotStatus] = {
    "venue_01_2026-07-11_1000": SlotStatus.NOTUSED,
    "venue_01_2026-07-12_2030": SlotStatus.USED,
}

#: The recordings are an hour each, sampled every 15 s.
MINUTES = 60
SECONDS_PER_MINUTE = 60


class StillSlotSource(FrameSource):
    """One slot's labelled stills, served as minutes.

    The frames of a recording are named ``slot_<date>_<time>_<cam>_t<seconds>.jpg`` and the
    manifest carries ``t_s``, so a minute is just the frames whose second falls inside it.
    A minute with no frame returns ``None`` and is counted as missed - the corpus does not
    cover every minute of either hour, and inventing the gaps would inflate the capture rate
    that the verdict's trustworthiness rests on.
    """

    def __init__(self, rows, *, burst: bool = False) -> None:
        self._by_camera_minute: dict[tuple[str, int], list] = defaultdict(list)
        for row in rows:
            self._by_camera_minute[(row.camera, row.t_s // SECONDS_PER_MINUTE)].append(row)
        for frames in self._by_camera_minute.values():
            frames.sort(key=lambda r: r.t_s)
        self._cameras = sorted({r.camera for r in rows})
        self._burst = burst

    def cameras(self) -> list[str]:
        return list(self._cameras)

    @property
    def n_minutes(self) -> int:
        return MINUTES

    def _load(self, row) -> Frame | None:
        image = cv2.imread(str(DATASET / row.file))
        if image is None:
            return None
        return Frame(row.camera, row.t_s // SECONDS_PER_MINUTE, image, str(row.file))

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        rows = self._by_camera_minute.get((camera_id, minute_index), [])
        return self._load(rows[0]) if rows else None

    def read_burst(self, camera_id: str, minute_index: int, *, n: int = 3,
                   spacing_s: float = 1.0) -> list[Frame]:
        """The minute's frames at their **real** spacing, or one frame.

        ``spacing_s`` is accepted and ignored, deliberately: this source cannot honour it and
        pretending otherwise would hand the rule three frames it believes are a second apart
        when they are fifteen. `--burst` opts into the wider spacing knowingly.
        """
        if not self._burst:
            frame = self.read(camera_id, minute_index)
            return [frame] if frame is not None else []
        rows = self._by_camera_minute.get((camera_id, minute_index), [])[:n]
        return [f for f in (self._load(r) for r in rows) if f is not None]


def minute_truth(rows) -> dict[int, Class3]:
    """The majority label of each minute's frames, for the per-minute column only."""
    by_minute: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        by_minute[row.t_s // SECONDS_PER_MINUTE][Class3(row.class3)] += 1
    return {m: c.most_common(1)[0][0] for m, c in by_minute.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["yolov8n", "dinov2"],
                    help="model keys; a detector runs the rule, a backbone the probe")
    ap.add_argument("--burst", action="store_true",
                    help="read up to 3 frames per minute at their real 15 s spacing")
    args = ap.parse_args()

    every = [r for r in read_manifest(DATASET / "manifest.csv")
             if r.source in ("regular", "motion")]
    by_slot: dict[str, list] = defaultdict(list)
    for row in every:
        by_slot[row.slot_id].append(row)

    print(f"burst: {'up to 3 frames per minute at their real 15 s spacing' if args.burst else
                    'one frame per minute (the stills cannot supply a 1 s burst)'}")
    print(f"{len(by_slot)} recorded slots, {MINUTES}-minute hours\n")

    records = []
    for arm in args.arms:
        pipeline = assemble(arm)
        print(f"=== {arm} ({pipeline.kind}) ===")
        for slot_id, rows in sorted(by_slot.items()):
            truth = SLOT_TRUTH.get(slot_id)
            source = StillSlotSource(rows, burst=args.burst)
            venue = rows[0].venue
            # `on_minute` carries the **fused** state - the pitch-level verdict after the
            # cameras are summed. `run.samples` are per camera, and reading those would score
            # one camera's half of the pitch against a label for the whole of it, which is
            # the artefact fusion exists to remove. Measured: on the evening slot that
            # mistake reads 0.65 where the fused states read 0.98.
            fused: dict[int, Class3] = {}
            run = run_slot(slot_id, source, None, pipeline=pipeline,
                           venue=venue, slot_key=slot_id,
                           on_minute=lambda m, state, _f=fused: _f.setdefault(m, state))

            per_minute = minute_truth(rows)
            scored = [(per_minute[m], fused[m]) for m in sorted(per_minute) if m in fused]
            right = sum(1 for t, p in scored if t is p)

            status = run.verdict.status
            mark = "ok" if truth is not None and status is truth else "WRONG"
            want = f"(truth {truth.value})" if truth else ""
            print(f"  {slot_id:<26}{status.value:<9}{want:<20}{mark:>6}")
            print(f"      minutes captured {run.minutes_captured}/{MINUTES}, missed "
                  f"{run.minutes_missed}   per-minute agreement {right}/{len(scored)}"
                  f"{f' = {right / len(scored):.2f}' if scored else ''}")
            print(f"      {run.verdict.reason}")
            records.append({
                "arm": arm, "kind": pipeline.kind, "slot": slot_id,
                "burst": args.burst, "truth": truth.value if truth else "",
                "status": status.value, "correct": bool(truth and status is truth),
                "minutes_captured": run.minutes_captured,
                "minutes_missed": run.minutes_missed,
                "play_ratio": round(run.verdict.play_ratio, 4),
                "empty_ratio": round(run.verdict.empty_ratio, 4),
                "maintenance_ratio": round(run.verdict.maintenance_ratio, 4),
                "uncertain_ratio": round(getattr(run.verdict, "uncertain_ratio", 0.0), 4),
                "minute_agreement": round(right / len(scored), 4) if scored else "",
                "n_minutes_scored": len(scored),
            })
        print()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "rule_slots.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {out}")
    print("\nthe slot verdict is what an operator receives; the per-minute column is "
          "diagnostic\nand is scored against each minute's majority frame label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
