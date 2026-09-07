"""Choosing the three images that justify a verdict (WP6).

Every verdict is bound to three snapshots so an operator can agree or disagree in seconds
rather than scrubbing an hour of footage. Which three matters more than it sounds - they
are the entire basis on which a disputed slot gets settled.

The rule: **one frame from each third of the slot, each the most confident of its third**.

Spreading across thirds rather than taking the top three by confidence is deliberate. The
three most confident frames of a match tend to come from the same passage of play, which
proves the pitch was busy for one minute and says nothing about the other fifty-nine. A
frame from the start, middle and end is a claim about the whole slot, which is what the
verdict is.

Frames matching the verdict are preferred; when a third has none - a late kick-off leaves
an empty first third of a USED slot - the most confident frame of any class backfills, so
the operator still sees that period rather than a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pitch_occupancy.data.taxonomy import Class3, SlotStatus

__all__ = [
    "EvidenceFrame",
    "Transition",
    "select_evidence",
    "find_transitions",
    "STATUS_TARGET_CLASS",
]

#: The class a verdict is claiming; evidence should show it where possible.
STATUS_TARGET_CLASS: dict[SlotStatus, Class3 | None] = {
    SlotStatus.USED: Class3.ACTIVE_PLAY,
    SlotStatus.NOTUSED: Class3.EMPTY,
    SlotStatus.REVIEW: None,  # nothing is being claimed, so show the most confident frames
}


@dataclass(frozen=True, slots=True)
class EvidenceFrame:
    minute_index: int
    predicted: Class3
    confidence: float
    image_path: str | None
    third: int  # 0, 1 or 2 - which part of the slot it represents
    is_backfill: bool  # True when no frame of the target class existed in this third


def select_evidence(
    samples: Sequence[tuple[int, Class3 | str, float, str | None]],
    status: SlotStatus,
    *,
    n_thirds: int = 3,
) -> list[EvidenceFrame]:
    """Pick one representative frame per third of the slot.

    ``samples`` are ``(minute_index, predicted_class, confidence, image_path)``. Returns up
    to ``n_thirds`` frames ordered by time; a third with no samples at all is skipped
    rather than filled from a neighbour, because a fabricated timeline is worse than a
    short one.
    """
    if not samples:
        return []

    rows = [(int(m), Class3(c), float(conf), path) for m, c, conf, path in samples]
    lo = min(r[0] for r in rows)
    hi = max(r[0] for r in rows)
    span = max(hi - lo + 1, 1)
    target = STATUS_TARGET_CLASS[status]

    chosen: list[EvidenceFrame] = []
    for third in range(n_thirds):
        start = lo + span * third // n_thirds
        end = lo + span * (third + 1) // n_thirds
        # the final third absorbs the remainder so the last minute is never dropped
        in_third = [r for r in rows if start <= r[0] < end or (third == n_thirds - 1 and r[0] >= end)]
        if not in_third:
            continue

        preferred = [r for r in in_third if target is None or r[1] is target]
        backfill = not preferred
        pool = in_third if backfill else preferred
        best = max(pool, key=lambda r: r[2])
        chosen.append(
            EvidenceFrame(
                minute_index=best[0], predicted=best[1], confidence=best[2],
                image_path=best[3], third=third, is_backfill=backfill and target is not None,
            )
        )
    return sorted(chosen, key=lambda e: e.minute_index)


@dataclass(frozen=True, slots=True)
class Transition:
    """A sustained change of state within a slot."""

    minute: int  # the first minute of the new state
    before: Class3
    after: Class3
    stable_before: int  # consecutive minutes held before the change
    stable_after: int  # consecutive minutes held after it
    #: Position in the ``states`` sequence. Equal to ``minute`` only when the sequence has
    #: one entry per minute - which it does not when a camera dropped out, because
    #: `worker.py` appends nothing for a minute it could not observe. Keeping both is what
    #: stops a position being read as a clock time: it was, and the evidence for every
    #: gapped slot pointed at the wrong part of it.
    index: int = -1

    def describe(self) -> str:
        return (
            f"{self.before.name.lower().replace('_', ' ')} to "
            f"{self.after.name.lower().replace('_', ' ')} at minute {self.minute}"
        )


def find_transitions(
    states: Sequence[Class3 | str],
    *,
    min_stable: int = 5,
    minutes: Sequence[int] | None = None,
) -> list[Transition]:
    """Points where the slot changed state and stayed changed.

    ``min_stable`` minutes are required on both sides, which is what separates a match
    ending from a player walking through frame. Without it every flicker is a transition
    and the signal is worthless.

    This is what makes an abandoned match distinguishable from a slot that was never used:
    play for twenty minutes then empty for forty is a different event from empty
    throughout, and only the transition tells them apart.

    ``minutes`` maps each entry of ``states`` to the minute it was observed in. **Pass it
    whenever the sequence can have holes.** `worker.py` appends nothing for a minute no
    camera could be read, so ``states`` is compacted while the clock is not: on a slot with
    a ten-minute outage the run boundary sits at position 10 and minute 20. Without this
    argument the returned ``minute`` is really a position, and every caller treating it as
    a time - the evidence selector, ``describe()``, ``SlotRun.transitions`` - was pointed at
    the wrong part of the slot.
    """
    if not states:
        return []
    if minutes is not None and len(minutes) != len(states):
        raise ValueError(
            f"minutes and states must align: {len(minutes)} vs {len(states)}. They are "
            f"appended together per observed minute, so a mismatch means one of them was "
            f"filtered and the other was not."
        )
    seq = [Class3(s) for s in states]

    # collapse into runs, then keep only the boundaries where both sides are long enough
    runs: list[tuple[Class3, int, int]] = []  # (state, start, length)
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] is not seq[start]:
            runs.append((seq[start], start, i - start))
            start = i

    out: list[Transition] = []
    for a, b in zip(runs, runs[1:], strict=False):
        if a[2] >= min_stable and b[2] >= min_stable:
            out.append(
                Transition(
                    minute=b[1] if minutes is None else int(minutes[b[1]]),
                    before=a[0], after=b[0],
                    stable_before=a[2], stable_after=b[2],
                    index=b[1],
                )
            )
    return out


def select_evidence_around_transitions(
    samples: Sequence[tuple[int, Class3 | str, float, str | None]],
    states: Sequence[Class3 | str],
    status: SlotStatus,
    *,
    min_stable: int = 5,
) -> list[EvidenceFrame]:
    """Evidence chosen around a state change, falling back to thirds when there is none.

    For a slot that changed - play starting late, stopping early, maintenance arriving -
    "here is minute 20 and here is minute 22" settles a dispute far better than three
    frames that all show the same thing. Thirds remain right for a uniform slot, so this
    only takes over when a transition is actually found.
    """
    rows = [(int(m), Class3(c), float(conf), p) for m, c, conf, p in samples]
    if not rows:
        return []

    # The minutes are what `states` was observed in, and they are passed to
    # `find_transitions` so its `minute` is a clock time rather than a list position. It
    # used to be a position looked up in a dict keyed by minute: identical while every
    # minute was observed, and wrong for every slot with a camera gap. A ten-minute outage
    # put the evidence for a change at minute 20 on minutes 10 and 12 - both still showing
    # the old state - and silently returned two frames instead of three.
    minutes = [r[0] for r in rows]
    transitions = find_transitions(states, min_stable=min_stable, minutes=minutes)
    if not transitions:
        return select_evidence(samples, status)

    by_minute = {r[0]: r for r in rows}
    # the largest change: the one a dispute would turn on
    t = max(transitions, key=lambda x: min(x.stable_before, x.stable_after))

    # Neighbours are taken from the observed sequence rather than by arithmetic on the
    # clock, so a gap around the transition still yields three frames: minute-2 does not
    # exist after an outage, but the observation two *samples* earlier does.
    picks: list[tuple[int, int, bool]] = []  # (minute, third, is_backfill)
    i = t.index
    for pos, third in ((max(0, i - 2), 0), (i, 1), (min(len(minutes) - 1, i + 2), 2)):
        minute = minutes[pos]
        if minute in by_minute:
            picks.append((minute, third, False))

    seen: set[int] = set()
    out: list[EvidenceFrame] = []
    for minute, third, backfill in picks:
        if minute in seen:
            continue
        seen.add(minute)
        _, cls, conf, path = by_minute[minute]
        out.append(EvidenceFrame(minute, cls, conf, path, third, backfill))
    return sorted(out, key=lambda e: e.minute_index)
