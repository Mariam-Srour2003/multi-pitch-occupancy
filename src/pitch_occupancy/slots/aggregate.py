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
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.data.taxonomy import to_class3 as folder_to_class3

__all__ = ["Thresholds", "SlotVerdict", "UNCERTAIN", "aggregate_slot", "tune_thresholds"]


@dataclass(frozen=True, slots=True)
class Thresholds:
    used_min_play: float = 0.35
    notused_max_play: float = 0.10
    notused_min_empty: float = 0.75
    #: Below this mean confidence the slot goes to REVIEW regardless of ratios, so the
    #: audit never accuses on footage it could barely read.
    #:
    #: **Zero means this guard is off, and it is off everywhere.** `conf < 0.0` is never
    #: true, no non-test caller raises it, and `experiments/end_to_end_slots.py` passes
    #: `review_below_confidence=0.0` explicitly. So the sentence above describes a wired
    #: code path rather than an active protection - which is worth saying plainly, because
    #: three claims in this project have now turned out to describe guards that were not
    #: guarding (the final-test-set lock's cwd-relative path and the processor-geometry
    #: fingerprint were the others).
    #:
    #: It is left at zero deliberately rather than fixed here: picking a threshold by hand
    #: is exactly the "hyper-parameters, not constants" mistake this module's own header
    #: warns about. Calibrating it belongs with RQ6 - `evaluation/calibration.py` already
    #: has the risk-coverage machinery - and is blocked by the same degenerate test set.
    #: See TODO WP6-T5.
    #:
    #: The ethics commitment does **not** rest on this. A human confirms every anomaly and
    #: the system takes no automated financial action (WP6-T12); this guard is
    #: defence-in-depth on top of that, not the thing standing between the audit and a
    #: false accusation.
    review_below_confidence: float = 0.0

    #: A verdict computed from fewer than this share of the slot's minutes is downgraded
    #: to REVIEW.
    #:
    #: **Unlike `review_below_confidence`, this one is on.** That distinction is the point.
    #: A confidence threshold is a model hyper-parameter and picking one by hand is the
    #: mistake this module's header warns about - it is blocked on RQ6's calibration, and it
    #: sits at 0.0 with a note saying so. Capture rate is not a model quantity at all: it is
    #: how much of the hour was observed, and "we saw eleven minutes of this hour" is not a
    #: statement about a pitch whatever the classifier says about those eleven minutes.
    #:
    #: 0.5 is a judgement and is recorded as one. It is the weakest defensible reading -
    #: a verdict from less than half the slot is not a verdict about the slot - rather than
    #: the value that best separates anything, because nothing here has been calibrated
    #: against real degraded slots. `SlotConditions.concerns()` already flags anything below
    #: 0.8 for a human to look at; this is the floor below which the system stops asserting.
    #:
    #: The failure it exists for is the one WP7-T5 asks about: a camera dies twenty minutes
    #: into a slot and the system confidently reports NOTUSED on the twenty minutes before
    #: the pitch filled up. Degraded mode is REVIEW, never a fabricated verdict.
    review_below_capture: float = 0.5


@dataclass(frozen=True, slots=True)
class SlotVerdict:
    status: SlotStatus
    reason: str
    play_ratio: float
    empty_ratio: float
    maintenance_ratio: float
    n_samples: int
    mean_confidence: float
    #: The share of the slot's minutes on which the system declined to say (A36). Abstained
    #: minutes are not samples: they lower the capture rate like a minute with no frame, and
    #: can push the verdict toward REVIEW and nowhere else.
    uncertain_ratio: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.status.value:<8} play {self.play_ratio:.2f} empty {self.empty_ratio:.2f} "
            f"maint {self.maintenance_ratio:.2f} (n={self.n_samples}) - {self.reason}"
        )


#: What a caller may write for a minute the system declined to decide (A36).
UNCERTAIN = "UNCERTAIN"


def _as_class3(state: Class3 | str | None) -> Class3 | None:
    """A per-minute state as its reporting class, or None for an abstention.

    Accepts the three-class values the probe path has always written, the four folder names
    the detector-first path speaks (`vision/rules.MinuteState`), ``"UNCERTAIN"`` and None.
    """
    if state is None or state == UNCERTAIN:
        return None
    if isinstance(state, Class3):
        return state
    try:
        return Class3(state)
    except ValueError:
        return folder_to_class3(str(state))


def aggregate_slot(
    states: Sequence[Class3 | str | None],
    confidences: Sequence[float] | None = None,
    thresholds: Thresholds | None = None,
    *,
    minutes_expected: int | None = None,
) -> SlotVerdict:
    """Aggregate one slot's fused per-minute states into a verdict.

    ``minutes_expected`` is how long the slot was *supposed* to be. Without it the ratios
    are computed over whatever arrived and the caller is asserting that is the whole slot;
    with it, a slot that lost most of its minutes is downgraded to REVIEW rather than
    decided from the fragment that survived. Optional rather than required so existing
    callers keep working, and absent rather than defaulted to ``len(states)``, which would
    make the check silently vacuous - the failure mode this project keeps meeting.

    **An abstained minute is not a sample** (A36). ``None`` or ``"UNCERTAIN"`` in ``states``
    is a minute the system declined to decide - no boundary, no detector, movement without
    anyone found. The ratios are computed over the decided minutes only, and the abstained
    ones count against the capture floor exactly as a minute with no frame does: they can
    push the slot toward REVIEW and nowhere else, so the REVIEW-never-accuses rule of
    `slots/reconcile.py` is untouched. ``confidences`` are the decided minutes' confidences,
    in order.
    """
    th = thresholds or Thresholds()
    if not states:
        return SlotVerdict(
            SlotStatus.REVIEW, "no samples captured for this slot", 0.0, 0.0, 0.0, 0, 0.0
        )

    mapped = [_as_class3(s) for s in states]
    values = [v for v in mapped if v is not None]
    abstained = len(mapped) - len(values)
    total = minutes_expected or len(states)
    uncertain_ratio = abstained / total if total else 0.0
    n = len(values)
    conf = sum(confidences) / len(confidences) if confidences else 1.0
    if not values:
        return SlotVerdict(
            SlotStatus.REVIEW,
            f"no decided minutes: the system abstained on all {abstained} observed minute(s)",
            0.0, 0.0, 0.0, 0, conf, uncertain_ratio,
        )

    counts = Counter(values)
    play = counts[Class3.ACTIVE_PLAY] / n
    empty = counts[Class3.EMPTY] / n
    maint = counts[Class3.MAINTENANCE_NON_SPORTING] / n

    if minutes_expected and n / minutes_expected < th.review_below_capture:
        why = (f" ({abstained} of them observed but abstained)" if abstained else "")
        return SlotVerdict(
            SlotStatus.REVIEW,
            f"only {n} of {minutes_expected} minutes decided "
            f"({n / minutes_expected:.0%}){why}; too little of the slot was observed to decide",
            play, empty, maint, n, conf, uncertain_ratio,
        )
    if confidences and conf < th.review_below_confidence:
        return SlotVerdict(
            SlotStatus.REVIEW,
            f"mean confidence {conf:.2f} below {th.review_below_confidence:.2f}; "
            f"footage too uncertain to decide",
            play, empty, maint, n, conf, uncertain_ratio,
        )
    if play >= th.used_min_play:
        return SlotVerdict(
            SlotStatus.USED,
            f"active play in {play:.0%} of samples (>= {th.used_min_play:.0%})",
            play, empty, maint, n, conf, uncertain_ratio,
        )
    if play < th.notused_max_play and empty >= th.notused_min_empty:
        return SlotVerdict(
            SlotStatus.NOTUSED,
            f"empty in {empty:.0%} of samples with only {play:.0%} active play",
            play, empty, maint, n, conf, uncertain_ratio,
        )
    if maint >= th.used_min_play:
        return SlotVerdict(
            SlotStatus.REVIEW,
            f"maintenance-dominated ({maint:.0%}); pitch occupied but not played on",
            play, empty, maint, n, conf, uncertain_ratio,
        )
    return SlotVerdict(
        SlotStatus.REVIEW,
        f"intermittent activity: {play:.0%} play, {empty:.0%} empty - neither threshold met",
        play, empty, maint, n, conf, uncertain_ratio,
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
