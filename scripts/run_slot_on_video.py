"""Run a video through the deployed slot path, exactly as the worker would.

Not `classify_batch` with arguments chosen by hand - `worker.run_slot`, with the boundary
looked up the way production looks it up and the motion gate constructed the way production
constructs it. The point is to check the wiring, not to re-measure the idea: an experiment
that reproduces a result by calling the components directly proves nothing about whether the
system uses them.

    uv run python scripts/run_slot_on_video.py VIDEO --every 15 --truth 9,12,15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision import roi
from pitch_occupancy.vision.motion import MotionGate
from pitch_occupancy.vision.people import PersonGate
from pitch_occupancy.frame_source import Frame, FrameSource
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--every", type=float, default=15.0)
    ap.add_argument("--truth", default="")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-person", action="store_true")
    ap.add_argument("--no-boundary", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(args.every * fps)))
    frames, pool, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % 40 == 0 and len(pool) < 40:
            pool.append(cv2.resize(fr, (320, 180)))
        if i % step == 0:
            frames.append(fr)
        i += 1
    cap.release()

    # A camera the store has never seen has no boundary, which is what production would find
    # on the first night at a new site. `derive_roi` is the step that fixes that, so it is run
    # here rather than skipped - and the fact that it is *needed* is the finding, not a detail.
    polygon = roi.get(CAMERA)
    if polygon is None and not args.no_boundary:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from derive_roi import polygon_from_mask, turf_mask

        med = np.median(np.stack(pool), axis=0).astype(np.uint8)
        polygon = polygon_from_mask(turf_mask(med))
        print(f"no stored boundary for {CAMERA!r}; derived one from the footage "
              f"({len(polygon)} points, keeps {roi.coverage(polygon):.0%})")
    elif args.no_boundary:
        polygon = None
        print("boundary disabled")

    from pitch_occupancy.vision.classifier import load_classifier

    clf = load_classifier()
    gate = None if args.no_gate else MotionGate()
    pgate = None if args.no_person else PersonGate()
    print(f"motion gate: {'off' if gate is None else f'threshold {gate.threshold}'}")
    print(f"person gate: {'off' if pgate is None else 'on'}")
    print(f"{len(frames)} minutes at {args.every:g}s spacing\n")

    seen: list[tuple[int, Class3]] = []
    run = run_slot(
        "video", VideoSource(frames),
        clf,
        polygon_for=(lambda _c: polygon),
        motion_gate=gate,
        person_gate=pgate,
        on_minute=lambda m, s: seen.append((m, s)),
    )

    person = {int(x) for x in args.truth.split(",") if x.strip().isdigit()}
    wrong = []
    print(f"{'#':>3}{'verdict':>26}{'truth':>10}")
    for m, state in seen:
        truth = "person" if m in person else "empty"
        bad = state is Class3.ACTIVE_PLAY
        if bad:
            wrong.append((m, truth))
        print(f"{m:>3}{state.name:>26}{truth:>10}{'   <-- PLAY' if bad else ''}")

    empt = [m for m, _ in enumerate(seen) if m not in person]
    fp = [m for m, t in wrong if t == "empty"]
    print(f"\nverdict: {run.verdict.state.name if hasattr(run.verdict, 'state') else run.verdict}")
    print(f"false-play on the {len(empt)} empty minutes: {len(fp)}/{len(empt)} "
          f"= {len(fp) / len(empt):.2f}   minutes {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
