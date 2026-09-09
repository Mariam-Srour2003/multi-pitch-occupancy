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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

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


class Classifier(Protocol):
    """Anything that turns one frame into a class and a confidence."""

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
    parser = argparse.ArgumentParser(description="Sample and evaluate slots.")
    parser.add_argument("--source", choices=["video"], default="video")
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir / "venue_01")
    parser.add_argument("--model", default=settings.default_model_key)
    args = parser.parse_args()

    raise NotImplementedError(
        "The live loop still needs the classifier wiring (a probe over cached features) "
        "and the schedule reader. run_slot() is complete and covered by tests; "
        "experiments/end_to_end_slots.py exercises the same path on the recorded slots."
    )


if __name__ == "__main__":
    main()
