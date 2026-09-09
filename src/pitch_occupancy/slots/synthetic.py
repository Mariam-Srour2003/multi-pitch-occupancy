"""Composing slot sequences from real frames (WP5-T8).

A slot classifier needs slots, and this project has **two** real labelled ones. That is not a
sample; it is an anecdote. So training sequences are composed from manifest frames, and the
boundary between composed and real is kept sharp enough that no table can blur it: everything
here returns :class:`SlotSequence` with ``synthetic=True``, and the one function that builds
real slots sets it False. Nothing sets it by argument.

**The frames are real; the ordering is invented.** Each minute of a composed slot is an actual
manifest frame of the required class, so the classifier meets the noise it will actually meet -
its own confusions, this dataset's lighting, these venues' turf. What is invented is which
class each minute *should* be, and that is the part a reader must discount.

**Labels come from the template, not from the aggregation rule.** A composed full match is
USED because it is a full match, not because its play ratio clears 0.35. Labelling by
:func:`~pitch_occupancy.slots.aggregate.aggregate_slot` would make the threshold baseline
correct by construction and any comparison against it circular - the model would be graded on
reproducing the rule it is supposed to beat. The five templates encode a semantic judgement
about what each situation means, and the thresholds are one guess at that mapping.

The templates are the ones the plan names:

* **full match** - play throughout. USED.
* **no show** - empty throughout. NOTUSED.
* **late start** - empty for the first third, then play. USED: the pitch was used.
* **maintenance window** - a groundsman, no game. REVIEW: occupied but not played on, and
  not a billing decision a machine should make.
* **intermittent** - play and empty alternating, neither dominant. REVIEW, and the honest
  answer: nobody looking at the footage would be sure either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from pitch_occupancy.data.taxonomy import Class3, SlotStatus

__all__ = [
    "SlotSequence",
    "TEMPLATES",
    "template_states",
    "compose_slot",
    "compose_dataset",
]

PLAY, EMPTY, MAINT = Class3.ACTIVE_PLAY, Class3.EMPTY, Class3.MAINTENANCE_NON_SPORTING


@dataclass(frozen=True, slots=True)
class SlotSequence:
    """One slot: an ordered per-minute probability matrix and a verdict.

    ``probabilities`` is ``(T, 3)`` over :data:`CLASS_ORDER`, ``truth`` the slot's verdict,
    and ``synthetic`` says whether the *ordering* was composed. That flag is not decoration:
    :mod:`pitch_occupancy.slots.stan` refuses to report a headline from fewer than
    :data:`~pitch_occupancy.slots.stan.MIN_REAL_SLOTS` real ones, and it counts them by
    reading this field.
    """

    slot_id: str
    probabilities: np.ndarray
    truth: SlotStatus
    synthetic: bool
    template: str = ""
    true_states: tuple[Class3, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.probabilities.ndim != 2 or self.probabilities.shape[1] != 3:
            raise ValueError(
                f"{self.slot_id}: probabilities are {self.probabilities.shape}, expected (T, 3)"
            )

    @property
    def n_minutes(self) -> int:
        return int(self.probabilities.shape[0])


#: name -> (per-minute class plan, the verdict the situation means)
TEMPLATES: dict[str, tuple[str, SlotStatus]] = {
    "full_match": ("play throughout", SlotStatus.USED),
    "no_show": ("empty throughout", SlotStatus.NOTUSED),
    "late_start": ("empty for the first third, then play", SlotStatus.USED),
    "maintenance_window": ("maintenance throughout the middle", SlotStatus.REVIEW),
    "intermittent": ("play and empty alternating, neither dominant", SlotStatus.REVIEW),
}


def template_states(name: str, n_minutes: int, rng: np.random.Generator) -> list[Class3]:
    """The *true* per-minute states for one template. No classifier involved.

    Raises:
        KeyError: on an unknown template, rather than falling back to a default - a silent
            default would put a differently-shaped slot under a label meaning something else.
    """
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; known: {', '.join(TEMPLATES)}")
    if n_minutes < 3:
        raise ValueError(f"a {n_minutes}-minute slot is not a slot")

    # Every boundary below is jittered, and that is not decoration. With the fractions fixed
    # - a late start at exactly n/3, a maintenance window at exactly the middle two thirds -
    # each template becomes one stereotyped shape, the label is recoverable by recognising
    # which of five shapes a sequence is, and any model that reads shape scores 1.000. The
    # first run of `stan_preliminary.py` did exactly that. Jitter does not make the task
    # hard, but it does stop the boundary itself being the answer.
    if name == "full_match":
        # even a full match has a warm-up and a walk-off
        edge = int(rng.integers(0, max(1, n_minutes // 12) + 1))
        return [EMPTY] * edge + [PLAY] * (n_minutes - edge)
    if name == "no_show":
        return [EMPTY] * n_minutes
    if name == "late_start":
        late = int(round(n_minutes * float(rng.uniform(0.20, 0.50))))
        return [EMPTY] * max(1, late) + [PLAY] * (n_minutes - max(1, late))
    if name == "maintenance_window":
        length = int(round(n_minutes * float(rng.uniform(0.35, 0.80))))
        start = int(rng.integers(0, max(1, n_minutes - length + 1)))
        states = [EMPTY] * n_minutes
        for i in range(start, min(start + length, n_minutes)):
            states[i] = MAINT
        return states
    # intermittent: play in short runs adding to between an eighth and a third of the slot.
    # The upper end deliberately overlaps a short late start, because in reality it does: a
    # slot that is 30% play with the play scattered is genuinely ambiguous, which is what
    # REVIEW is for. A generator whose classes never overlap has a ceiling of 1.000 and
    # measures separability rather than difficulty.
    states = [EMPTY] * n_minutes
    target = max(2, int(round(n_minutes * float(rng.uniform(0.125, 0.33)))))
    placed = 0
    guard = 0
    while placed < target and guard < 1000:
        guard += 1
        run = int(rng.integers(2, 5))
        start = int(rng.integers(0, max(1, n_minutes - run)))
        for i in range(start, min(start + run, n_minutes)):
            if states[i] is EMPTY:
                states[i] = PLAY
                placed += 1
    return states


def compose_slot(
    slot_id: str,
    template: str,
    n_minutes: int,
    frames_by_class: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[SlotSequence, list[int]]:
    """Compose one slot, returning it and the manifest row index chosen for each minute.

    ``frames_by_class`` maps a :class:`Class3` value to the row indices available for it. The
    probability matrix is left as zeros here and filled by :func:`compose_dataset`, because
    scoring frames is the caller's job - this module must not know what a backbone is.

    Sampling is **with replacement**, which matters and is a limitation rather than a choice:
    the dataset holds 6 MAINTENANCE frames in total, so a maintenance window of fifty minutes
    reuses each of them roughly eight times. A sequence model shown the same eight frames all
    hour has an easier problem than the real one, and that is one of the reasons the result
    this feeds is preliminary.
    """
    states = template_states(template, n_minutes, rng)
    chosen = []
    for state in states:
        pool = frames_by_class.get(state.value)
        if pool is None or len(pool) == 0:
            raise ValueError(
                f"no frames available for {state.value}; a {template} slot cannot be composed "
                f"without them, and composing it from another class would relabel the minute"
            )
        chosen.append(int(rng.choice(pool)))
    sequence = SlotSequence(
        slot_id=slot_id,
        probabilities=np.zeros((n_minutes, 3)),
        truth=TEMPLATES[template][1],
        synthetic=True,
        template=template,
        true_states=tuple(states),
    )
    return sequence, chosen


def compose_dataset(
    frames_by_class: dict[str, np.ndarray],
    score: Callable[[np.ndarray], np.ndarray],
    *,
    n_per_template: int = 40,
    minutes: tuple[int, int] = (45, 75),
    seed: int = 42,
    templates: Sequence[str] | None = None,
) -> list[SlotSequence]:
    """Compose a balanced set of slots and fill in each minute's probabilities.

    ``score`` takes an array of manifest row indices and returns their ``(n, 3)`` probability
    matrix under whatever classifier the caller is studying. Keeping it a parameter is what
    lets the experiment feed real probe outputs while the tests feed something trivial.

    Balanced by *template*, which is not balanced by verdict: two templates mean USED and two
    mean REVIEW, so NOTUSED is a quarter of the set. Left that way deliberately - the
    templates are the situations, and flattening them to balance verdicts would compose more
    no-shows than the situation list justifies.
    """
    rng = np.random.default_rng(seed)
    names = list(templates) if templates else list(TEMPLATES)
    out: list[SlotSequence] = []
    for name in names:
        for i in range(n_per_template):
            n = int(rng.integers(minutes[0], minutes[1] + 1))
            sequence, chosen = compose_slot(
                f"synth_{name}_{i:03d}", name, n, frames_by_class, rng
            )
            probabilities = np.asarray(score(np.array(chosen)), dtype=float)
            if probabilities.shape != (n, 3):
                raise ValueError(
                    f"score() returned {probabilities.shape} for {n} minutes, expected ({n}, 3)"
                )
            out.append(
                SlotSequence(
                    sequence.slot_id, probabilities, sequence.truth, True,
                    sequence.template, sequence.true_states,
                )
            )
    return out
