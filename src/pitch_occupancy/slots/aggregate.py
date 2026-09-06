"""Slot aggregation: per-minute states to one verdict (WP3 of the blueprint, RQ4).

A rental slot yields ~60-90 fused per-minute observations. The baseline rule turns them
into USED / NOTUSED / REVIEW by ratio thresholds:

    ACTIVE_PLAY ratio >= 0.35                        -> USED
    ACTIVE_PLAY ratio <  0.10 and EMPTY ratio >= 0.75 -> NOTUSED
    otherwise                                         -> REVIEW

**These thresholds are hyper-parameters, not constants.** They are the baseline STAN must
beat (WP5-T1), and a fair comparison tunes them on the same training slots rather than
leaving them at the values someone guessed. :func:`tune_thresholds` exists so the
comparison is honest.

REVIEW is not a failure mode - it is the design working. A slot with intermittent activity,
a lone groundsman, or footage too dark to read should reach a human rather than be
converted into a billing decision.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Sequence

from pitch_occupancy.data.taxonomy import Class3, SlotStatus

__all__ = ["Thresholds", "SlotVerdict", "aggregate_slot", "tune_thresholds"]


@dataclass(frozen=True, slots=True)
class Thresholds:
    used_min_play: float = 0.35
    notused_max_play: float = 0.10
    notused_min_empty: float = 0.75
    #: Below this mean confidence the slot goes to REVIEW regardless of ratios, so the
    #: audit never accuses on footage it could barely read.
    review_below_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class SlotVerdict:
    status: SlotStatus
    reason: str
    play_ratio: float
    empty_ratio: float
    maintenance_ratio: float
    n_samples: int
    mean_confidence: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.status.value:<8} play {self.play_ratio:.2f} empty {self.empty_ratio:.2f} "
            f"maint {self.maintenance_ratio:.2f} (n={self.n_samples}) - {self.reason}"
        )


def aggregate_slot(
    states: Sequence[Class3 | str],
    confidences: Sequence[float] | None = None,
    thresholds: Thresholds | None = None,
) -> SlotVerdict:
    """Aggregate one slot's fused per-minute states into a verdict."""
    th = thresholds or Thresholds()
    if not states:
        return SlotVerdict(
            SlotStatus.REVIEW, "no samples captured for this slot", 0.0, 0.0, 0.0, 0, 0.0
        )

    values = [Class3(s) for s in states]
    n = len(values)
    counts = Counter(values)
    play = counts[Class3.ACTIVE_PLAY] / n
    empty = counts[Class3.EMPTY] / n
    maint = counts[Class3.MAINTENANCE_NON_SPORTING] / n
    conf = sum(confidences) / len(confidences) if confidences else 1.0

    if confidences and conf < th.review_below_confidence:
        return SlotVerdict(
            SlotStatus.REVIEW,
            f"mean confidence {conf:.2f} below {th.review_below_confidence:.2f}; "
            f"footage too uncertain to decide",
            play, empty, maint, n, conf,
        )
    if play >= th.used_min_play:
        return SlotVerdict(
            SlotStatus.USED,
            f"active play in {play:.0%} of samples (>= {th.used_min_play:.0%})",
            play, empty, maint, n, conf,
        )
    if play < th.notused_max_play and empty >= th.notused_min_empty:
        return SlotVerdict(
            SlotStatus.NOTUSED,
            f"empty in {empty:.0%} of samples with only {play:.0%} active play",
            play, empty, maint, n, conf,
        )
    if maint >= th.used_min_play:
        return SlotVerdict(
            SlotStatus.REVIEW,
            f"maintenance-dominated ({maint:.0%}); pitch occupied but not played on",
            play, empty, maint, n, conf,
        )
    return SlotVerdict(
        SlotStatus.REVIEW,
        f"intermittent activity: {play:.0%} play, {empty:.0%} empty - neither threshold met",
        play, empty, maint, n, conf,
    )


def tune_thresholds(
    slots: Sequence[tuple[Sequence[Class3 | str], SlotStatus]],
    *,
    grid: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
) -> tuple[Thresholds, float]:
    """Grid-search thresholds against labelled slots; returns the best and its accuracy.

    Exists so that "STAN beats the thresholds" is a real claim. Comparing a learned model
    against *unlearned* constants would prove only that fitting beats guessing.
    """
    if not slots:
        raise ValueError("cannot tune thresholds without labelled slots")

    best, best_acc = Thresholds(), -1.0
    for used_min, notused_max, notused_empty in product(grid, grid, grid):
        if notused_max >= used_min:
            continue  # the bands would overlap
        th = Thresholds(used_min, notused_max, notused_empty)
        hits = sum(aggregate_slot(states, None, th).status is truth for states, truth in slots)
        acc = hits / len(slots)
        if acc > best_acc:
            best, best_acc = th, acc
    return best, best_acc
