"""Run a video through the deployed slot path, exactly as the worker would.

`worker.run_slot`, with the pipeline the worker assembles (`pipeline.assemble`): the same
detector, the same rule, the same boundary lookup. The point is to check the wiring: a run
that reproduces a result by calling the components directly proves nothing about whether the
system uses them.

    uv run python scripts/run_slot_on_video.py VIDEO --every 15 --truth 9,12,15
    uv run python scripts/run_slot_on_video.py VIDEO --truth-csv configs/unseen_clip_truth.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.frame_source import Frame, FrameSource
from pitch_occupancy.pipeline import assemble
from pitch_occupancy.vision import roi
from pitch_occupancy.worker import run_slot

ROOT = Path(__file__).resolve().parents[1]
CAMERA = "video_under_test"


class VideoSource(FrameSource):
    """One camera, one frame per 'minute', sampled at a fixed spacing."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    def cameras(self) -> list[str]:
        return [CAMERA]

    @property
    def n_minutes(self) -> int:
        return len(self._frames)

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        if minute_index >= len(self._frames):
            return None
        return Frame(camera_id, minute_index, self._frames[minute_index], "video")


def _sample(video: Path, every_s: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Frames at ``every_s`` spacing, and a small pool for deriving a boundary."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(every_s * fps)))
    frames, pool, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 40 == 0 and len(pool) < 40:
            pool.append(frame)
        if i % step == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames, pool


def _truth(args) -> set[int]:
    """Sample indices that hold a person, from ``--truth`` or a tracked truth CSV."""
    if args.truth_csv is not None:
        with Path(args.truth_csv).open(encoding="utf-8") as fh:
            return {
                int(r["sample"])
                for r in csv.DictReader(fh)
                if r.get("label", r.get("class4")) != "1_empty"
            }
    return {int(x) for x in args.truth.split(",") if x.strip().isdigit()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--every", type=float, default=15.0)
    ap.add_argument("--truth", default="", help="sample indices with a person on the pitch")
    ap.add_argument("--truth-csv", type=Path, default=None,
                    help="a truth file like configs/unseen_clip_truth.csv, instead of --truth")
    ap.add_argument("--model", default=None,
                    help="detector key; the one configs/rules.json names if unset")
    ap.add_argument("--no-boundary", action="store_true",
                    help="score the whole frame, ignoring any boundary")
    args = ap.parse_args()

    frames, pool = _sample(args.video, args.every)

    pipeline = assemble(args.model, require_boundary=not args.no_boundary)

    # A camera the store has never seen has no boundary, which is what production would find
    # on the first night at a new site. Deriving one from the footage is the step that fixes
    # that - `worker --derive-roi` does the same - so it is run here rather than skipped, and
    # the fact that it is *needed* is the finding, not a detail.
    polygon = None if args.no_boundary else pipeline.polygon(CAMERA)
    if polygon is None and not args.no_boundary:
        polygon = roi.derive_from_frames(pool)
        if polygon is None:
            raise SystemExit("no stored boundary and none could be derived from the footage")
        print(f"no stored boundary for {CAMERA!r}; derived one from the footage "
              f"({len(polygon)} points, keeps {roi.coverage(polygon):.0%})")
    elif args.no_boundary:
        print("boundary disabled: the whole frame is scored")

    print(pipeline.describe())
    print(f"{len(frames)} minutes at {args.every:g}s spacing\n")

    seen: list[tuple[int, Class3]] = []
    run = run_slot(
        "video", VideoSource(frames), None,
        pipeline=pipeline,
        polygon_for=(lambda _c: polygon),
        on_minute=lambda m, s: seen.append((m, s)),
    )

    person = _truth(args)
    wrong = []
    print(f"{'#':>3}{'verdict':>26}{'truth':>10}")
    for m, state in seen:
        truth = "person" if m in person else "empty"
        bad = state is Class3.ACTIVE_PLAY
        if bad:
            wrong.append((m, truth))
        print(f"{m:>3}{state.name:>26}{truth:>10}{'   <-- PLAY' if bad else ''}")

    empt = [m for m in range(len(frames)) if m not in person]
    fp = [m for m, t in wrong if t == "empty"]
    right_empty = sum(1 for m, s in seen if m in empt and s is Class3.EMPTY)
    # What the detector saw, not just what the rule decided.
    if run.people_counts:
        print(f"\npeople inside the boundary, per minute the detector ran: "
              f"{list(run.people_counts)}")
    if run.ball_minutes:
        print(f"a ball was seen inside the boundary on minutes {list(run.ball_minutes)}")
    if run.minutes_uncertain:
        print(f"{run.minutes_uncertain} minute(s) UNCERTAIN (no boundary)")

    print(f"\nslot verdict: {run.verdict.status} - {run.verdict.reason}")
    print(f"false-play on the {len(empt)} empty minutes: {len(fp)}/{len(empt)} "
          f"= {len(fp) / max(len(empt), 1):.2f}   minutes {fp}")
    print(f"empty minutes read EMPTY: {right_empty}/{len(empt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
