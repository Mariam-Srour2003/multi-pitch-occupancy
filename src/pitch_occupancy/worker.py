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


def run_slot(
    slot_id: str,
    source: FrameSource,
    classify: Classifier,
    *,
    thresholds: Thresholds | None = None,
    on_minute: Callable[[int, Class3], None] | None = None,
) -> SlotRun:
    """Sample, classify, fuse and aggregate one slot.

    Minutes where no camera produced a frame are counted as missed rather than filled in.
    They lower the capture rate, which is what a verdict's trustworthiness should depend
    on - a slot evaluated from ten of sixty minutes is not the same claim as one evaluated
    from all sixty.
    """
    cameras = source.cameras()
    samples: list[Sample] = []
    fused_states: list[Class3] = []
    fused_conf: list[float] = []
    evidence_rows: list[tuple[int, Class3, float, str | None]] = []
    disagreements: list[bool] = []
    missed = 0

    for minute in range(source.n_minutes):
        observations: dict[str, tuple[Class3, float]] = {}
        for camera in cameras:
            frame = source.read(camera, minute)
            if frame is None:
                continue  # a gap; never a fabricated observation
            state, confidence = classify(frame.image_bgr)
            observations[camera] = (state, confidence)
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
        evidence_rows.append((minute, fused.state, fused.confidence, None))
        if on_minute is not None:
            on_minute(minute, fused.state)

    verdict = aggregate_slot(fused_states, fused_conf, thresholds)
    return SlotRun(
        slot_id=slot_id,
        verdict=verdict,
        samples=samples,
        # a slot that changed state is better explained by the moment it changed than by
        # three frames from its thirds; thirds still apply when nothing changed
        evidence=select_evidence_around_transitions(
            evidence_rows, fused_states, verdict.status
        ),
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
