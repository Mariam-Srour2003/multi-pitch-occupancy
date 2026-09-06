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

__all__ = ["EvidenceFrame", "select_evidence", "STATUS_TARGET_CLASS"]

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
