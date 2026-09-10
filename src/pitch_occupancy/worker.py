"""The sampling scheduler (WP6-T2).

Runs as its own process, independent of the API. For each slot it pulls one frame per
camera per minute, classifies it, fuses the two halves, and at slot end aggregates a
verdict, selects evidence and reconciles against the booking record.

Keeping this out of the web layer is deliberate: a dashboard restart must never drop a
sample, and inference must never happen inside a request handler.

Run against recorded footage:

    uv run python -m pitch_occupancy.worker --source video

The classifier is injected rather than constructed here, so the pipeline can be exercised
with a stub in tests and with a real probe in production without the orchestration knowing
which it has.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.db.store import Sample
from pitch_occupancy.frame_source import FrameSource
from pitch_occupancy.slots.aggregate import SlotVerdict, Thresholds, aggregate_slot
from pitch_occupancy.slots.conditions import SlotConditions, summarise_conditions
from pitch_occupancy.slots.evidence import (
    EvidenceFrame,
    Transition,
    find_transitions,
    select_evidence_around_transitions,
)
from pitch_occupancy.slots.fusion import fuse

__all__ = ["Classifier", "SlotRun", "run_slot", "main"]


@runtime_checkable
class Classifier(Protocol):
    """Anything that turns one frame into a class and a confidence.

    Runtime-checkable so that a caller assembling the pipeline can assert it has one. The
    check is shallow - it sees a `__call__` and nothing about its signature - so a test
    that means to pin the contract should compare signatures, as `test_classifier.py` does.
    """

    def __call__(self, image_bgr: np.ndarray) -> tuple[Class3, float]: ...


@dataclass(frozen=True, slots=True)
class SlotRun:
    slot_id: str
    verdict: SlotVerdict
    samples: list[Sample]
    evidence: list[EvidenceFrame]
    minutes_captured: int
    minutes_missed: int
    conditions: SlotConditions | None = None
    transitions: tuple[Transition, ...] = ()

    @property
    def capture_rate(self) -> float:
        total = self.minutes_captured + self.minutes_missed
        return self.minutes_captured / total if total else 0.0


def _write_evidence(slot_dir: Path, minute: int, camera: str, image) -> Path | None:
    """Write one minute's evidence frame. Returns the path, or None if it could not be saved.

    A failed write must not take the slot down: the verdict is still valid without a picture,
    and losing an hour's classification because a disk was full would be a much worse outcome
    than losing the illustration of it.
    """
    import cv2

    path = slot_dir / f"minute_{minute:03d}_{camera}.jpg"
    try:
        ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    except Exception:  # noqa: BLE001 - any write failure degrades to "no picture"
        return None
    return path if ok else None


def run_slot(
    slot_id: str,
    source: FrameSource,
    classify: Classifier,
    *,
    thresholds: Thresholds | None = None,
    on_minute: Callable[[int, Class3], None] | None = None,
    evidence_dir: Path | None = None,
) -> SlotRun:
    """Sample, classify, fuse and aggregate one slot.

    Minutes where no camera produced a frame are counted as missed rather than filled in.
    They lower the capture rate, which is what a verdict's trustworthiness should depend
    on - a slot evaluated from ten of sixty minutes is not the same claim as one evaluated
    from all sixty.

    ``evidence_dir`` turns on **saving the evidence images**, and until 2026-09-09 nothing
    did. `EvidenceFrame.image_path` was populated with a literal ``None`` on every minute, so
    the selection machinery ran, chose three good frames, and recorded three paths to nothing.
    The dashboard's evidence inspector said "no evidence images bound" and was right; the
    override harvest (WP6-T7) would have found every frame missing. A verdict an operator
    cannot see the evidence for is the one thing this system must not produce, since a human
    confirming every anomaly is what `slots/authority.py` rests on.

    The winning camera's frame is written for **every** observed minute and the unselected ones
    are deleted once the choice is made, because which three minutes matter is not knowable
    until the whole slot has been seen. Sixty small JPEGs written and fifty-seven removed is
    cheaper than holding sixty full-resolution frames in memory, which at 1080p is most of a
    gigabyte per slot.

    Left off by default: writing frames of identifiable people to disk is a decision a caller
    makes, not something that happens because a function was called.
    """
    cameras = source.cameras()
    samples: list[Sample] = []
    fused_states: list[Class3] = []
    fused_conf: list[float] = []
    evidence_rows: list[tuple[int, Class3, float, str | None]] = []
    disagreements: list[bool] = []
    written: list[Path] = []
    missed = 0

    slot_dir = None
    if evidence_dir is not None:
        slot_dir = Path(evidence_dir) / slot_id
        slot_dir.mkdir(parents=True, exist_ok=True)

    for minute in range(source.n_minutes):
        observations: dict[str, tuple[Class3, float]] = {}
        images: dict[str, object] = {}
        for camera in cameras:
            frame = source.read(camera, minute)
            if frame is None:
                continue  # a gap; never a fabricated observation
            state, confidence = classify(frame.image_bgr)
            observations[camera] = (state, confidence)
            if slot_dir is not None:
                images[camera] = frame.image_bgr
            samples.append(
                Sample(
                    camera_id=camera,
                    minute_index=minute,
                    predicted=state.value,
                    confidence=confidence,
                    captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    image_path=frame.source_path,
                )
            )

        if not observations:
            missed += 1
            continue

        fused = fuse(observations)
        fused_states.append(fused.state)
        fused_conf.append(fused.confidence)
        disagreements.append(fused.disagreed)

        # The frame from the camera whose observation won the fusion - the one that actually
        # justifies the minute's state. Saving an averaged or arbitrary camera would hand an
        # operator a picture of an empty half to explain a verdict of "play on the other one".
        path = None
        if slot_dir is not None:
            winner = max(
                (c for c, _, _ in fused.per_camera if c in images),
                key=lambda c: observations[c][1],
                default=None,
            )
            if winner is not None:
                path = _write_evidence(slot_dir, minute, winner, images[winner])
                if path is not None:
                    written.append(path)
        evidence_rows.append((minute, fused.state, fused.confidence,
                              str(path) if path else None))
        if on_minute is not None:
            on_minute(minute, fused.state)

    # The slot's real length, so a slot that lost most of its minutes is downgraded to
    # REVIEW rather than decided from the fragment that survived (WP7-T5).
    verdict = aggregate_slot(
        fused_states, fused_conf, thresholds, minutes_expected=source.n_minutes
    )
    evidence = select_evidence_around_transitions(
        evidence_rows, fused_states, verdict.status
    )

    # Which three minutes matter is only knowable once the whole slot has been seen, so every
    # observed minute was written and the rest are removed now. Deleting only files this call
    # created, by identity rather than by pattern: a glob would also sweep up a frame an
    # operator had already been shown and disputed.
    if slot_dir is not None:
        keep = {e.image_path for e in evidence if e.image_path}
        for path in written:
            if str(path) not in keep:
                path.unlink(missing_ok=True)

    return SlotRun(
        slot_id=slot_id,
        verdict=verdict,
        samples=samples,
        # a slot that changed state is better explained by the moment it changed than by
        # three frames from its thirds; thirds still apply when nothing changed
        evidence=evidence,
        minutes_captured=len(fused_states),
        minutes_missed=missed,
        conditions=summarise_conditions(
            minutes_expected=source.n_minutes,
            confidences=fused_conf,
            disagreements=disagreements,
            cameras_seen=len({s.camera_id for s in samples}),
        ),
        transitions=tuple(
            find_transitions(fused_states, minutes=[r[0] for r in evidence_rows])
        ),
    )


def main() -> None:
    """Run the sampler over recorded slots, with the real classifier (WP6-T2).

    This is the acceptance criterion the work package was written around - *"runs
    continuously; DB fills; verdicts correct on recorded slots"* - and until
    `vision/classifier.py` existed it could not be met, because nothing in the repository
    could turn a frame into a class outside an experiment. This function raised
    `NotImplementedError` instead, and its message named the missing piece.

    ``--source video`` replays footage through the scheduler rather than pretending to be
    live: the same `run_due` a deployment calls, given recordings instead of cameras. The
    live path is `scheduler.live_sources` and it has still never been pointed at a camera
    (WP7-T3).

    ``--dry-run`` lists what would run and exits before the backbone is loaded, which is the
    shape `pitch retention` and `pitch schedule` already use: deciding is separable from
    doing, and a run that writes to the database should be something you asked for twice.
    """
    from pitch_occupancy.db.schema import connect, initialise
    from pitch_occupancy.scheduler import (
        describe,
        run_due,
        schedule_from_recordings,
        sources_from_recordings,
    )

    parser = argparse.ArgumentParser(description="Sample and evaluate slots.")
    parser.add_argument("--source", choices=["video"], default="video",
                        help="video replays recordings; there is no live choice here yet")
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir / "venue_01")
    parser.add_argument("--model", default=settings.default_model_key)
    parser.add_argument("--dry-run", action="store_true",
                        help="list the slots and exit, loading no model and writing nothing")
    parser.add_argument("--evidence-dir", type=Path, default=None,
                        help="save evidence frames here; off by default, because these are "
                             "images of identifiable people")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many slots")
    args = parser.parse_args()

    schedule, days = schedule_from_recordings(args.raw_dir)
    if not schedule.slots:
        raise SystemExit(f"no recorded slots under {args.raw_dir}")

    print(f"{len(schedule.slots)} slot(s) over {len(days)} day(s) under {args.raw_dir}")
    for day in days:
        for line in describe(schedule, datetime.combine(day, time.min), days=1):
            print(f"  {line}")
    if args.dry_run:
        print("\ndry run: nothing was classified and nothing was written")
        return

    from pitch_occupancy.vision.classifier import load_classifier

    classify = load_classifier(args.model)
    print(f"\nclassifier: {classify.backbone} probe fitted on {classify.n_train} "
          f"development frames (the locked venues are not among them)")

    source_for = sources_from_recordings(args.raw_dir)

    def report(slot_id: str, outcome: object) -> None:
        """Print one line per slot, whichever way it went.

        `run_due` hands this callback **the exception** when a slot fails, because one dead
        camera must not stop the other pitches being observed. A callback that assumed a
        `SlotRun` therefore crashed inside the handler for the failure it existed to report,
        which turns a skipped slot into a dead run. Found by running it: the schedule
        derived from recordings names four slot instances and only two were ever exported.
        """
        if isinstance(outcome, BaseException):
            print(f"  {slot_id:<34} SKIPPED  {type(outcome).__name__}: {outcome}")
            return
        print(f"  {slot_id:<34} {outcome.verdict.status:<8} "
              f"{outcome.minutes_captured:>3}/"
              f"{outcome.minutes_captured + outcome.minutes_missed} minutes  "
              f"capture {outcome.capture_rate:.0%}  "
              f"play {outcome.verdict.play_ratio:.2f} empty {outcome.verdict.empty_ratio:.2f}")

    # `run_due` persists only when it is given a connection - without one it classifies the
    # whole slot and stores nothing. The first version of this function did exactly that and
    # then printed "2 slot(s) written to the database", which is the shape of failure this
    # project keeps finding: not a crash, a confident sentence about something that did not
    # happen. `pitch info` names the file this opens.
    connection = connect(settings.db_path)
    initialise(connection)
    ran: list[str] = []
    seen: set[str] = set()
    try:
        for day in days:
            for slot in schedule.slots:
                if args.limit is not None and len(ran) >= args.limit:
                    break
                # `run_due` runs everything due at that moment, so two slots whose windows
                # overlap are both reached by the first call. Without this, the second call
                # would run the pair again and write one slot's verdict twice.
                if slot.slot_id(day) in seen:
                    continue
                # It selects by clock, so a slot is reached by standing inside its own
                # window rather than by calling `run_slot` directly - the point is to
                # exercise the scheduler's path, not to step around it.
                begins, _ = slot.window(day)
                just_ran = run_due(
                    schedule, begins,
                    source_for=source_for, classify=classify, model_key=args.model,
                    connection=connection, evidence_dir=args.evidence_dir, on_slot=report,
                )
                seen.update(just_ran)
                ran += just_ran
    finally:
        connection.close()

    print(f"\n{len(ran)} slot(s) written to the database")


if __name__ == "__main__":
    main()
