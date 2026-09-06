"""The conditions a verdict was formed under (IDEAS #3).

`REVIEW` is currently a verdict with a sentence attached. An operator opening one has to
work out from three images why the system hesitated, when the numbers that would answer it
were computed during the run and then discarded.

This keeps them. A REVIEW that says *"mean confidence 0.55, contrast 30% below this
camera's baseline, 8 of 60 minutes missing, the halves disagreed for a third of the slot"*
is actionable in seconds; one that says "intermittent activity" is a puzzle.

It is also the shape the weather idea plugs into. Precipitation, once detectable, becomes
another condition on the record - and the record is what lets reconciliation tell a
weather cancellation from a no-show, which is the single cheapest improvement available to
anomaly precision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

__all__ = ["SlotConditions", "summarise_conditions"]


@dataclass(frozen=True, slots=True)
class SlotConditions:
    """Everything observed about *how* a slot was watched, as opposed to what was seen."""

    minutes_expected: int
    minutes_captured: int
    minutes_missed: int
    mean_confidence: float
    min_confidence: float
    low_confidence_minutes: int
    camera_disagreements: int
    cameras_seen: int
    lighting: str  # day | night | mixed | unknown
    mean_contrast: float | None = None
    weather: str | None = None  # reserved; nothing measures it yet (see IDEAS #1)

    @property
    def capture_rate(self) -> float:
        return self.minutes_captured / self.minutes_expected if self.minutes_expected else 0.0

    @property
    def disagreement_rate(self) -> float:
        return self.camera_disagreements / self.minutes_captured if self.minutes_captured else 0.0

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["capture_rate"] = round(self.capture_rate, 4)
        d["disagreement_rate"] = round(self.disagreement_rate, 4)
        return d

    def concerns(self, *, low_confidence: float = 0.6) -> list[str]:
        """Plain-language reasons this verdict deserves a second look.

        Ordered by how much they undermine the verdict, so an operator reads the worst
        first. An empty list means nothing about the *observation* was unusual - the
        verdict may still be REVIEW because the pitch genuinely was ambiguous, and that is
        a different conversation.
        """
        out: list[str] = []
        if self.minutes_expected and self.capture_rate < 0.8:
            out.append(
                f"only {self.minutes_captured} of {self.minutes_expected} minutes were "
                f"captured ({self.capture_rate:.0%})"
            )
        if self.cameras_seen < 2:
            out.append(f"only {self.cameras_seen} camera contributed; half the pitch is unseen")
        if self.mean_confidence < low_confidence:
            out.append(f"mean confidence {self.mean_confidence:.2f} is low")
        if self.low_confidence_minutes > self.minutes_captured * 0.25:
            out.append(
                f"{self.low_confidence_minutes} minutes were individually uncertain"
            )
        if self.disagreement_rate > 0.3:
            out.append(
                f"the two halves disagreed for {self.disagreement_rate:.0%} of the slot, "
                f"which can mean an occluded or dirty lens"
            )
        if self.mean_contrast is not None and self.mean_contrast < 25:
            out.append(f"contrast averaged {self.mean_contrast:.0f}; the footage is hard to read")
        return out


def summarise_conditions(
    *,
    minutes_expected: int,
    confidences: Sequence[float],
    disagreements: Sequence[bool],
    cameras_seen: int,
    lighting: Sequence[str] = (),
    contrasts: Sequence[float] = (),
    low_confidence: float = 0.6,
) -> SlotConditions:
    """Fold a slot's per-minute observations into one record."""
    captured = len(confidences)
    lights = {light for light in lighting if light}
    return SlotConditions(
        minutes_expected=minutes_expected,
        minutes_captured=captured,
        minutes_missed=max(0, minutes_expected - captured),
        mean_confidence=sum(confidences) / captured if captured else 0.0,
        min_confidence=min(confidences) if confidences else 0.0,
        low_confidence_minutes=sum(1 for c in confidences if c < low_confidence),
        camera_disagreements=sum(1 for d in disagreements if d),
        cameras_seen=cameras_seen,
        lighting=(lights.pop() if len(lights) == 1 else "mixed" if lights else "unknown"),
        mean_contrast=(sum(contrasts) / len(contrasts)) if contrasts else None,
    )
