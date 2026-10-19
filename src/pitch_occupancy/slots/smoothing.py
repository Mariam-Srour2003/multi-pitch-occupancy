"""Majority smoothing over a sequence of per-sample states.

Shared by the single-camera clip reviewer (`clip_analysis.py`) and the two-camera reviewer
(`pair_analysis.py`), so there is one implementation rather than two that drift.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

__all__ = ["majority_smooth"]

def majority_smooth[S: Hashable](states: Sequence[S], window: int) -> list[S]:
    """Majority vote in a sliding window of ``window`` entries centred on each position.

    An isolated disagreeing entry between two agreeing neighbours is replaced by them, which
    is the cheapest correction for a single misread frame. It is also the cheapest way to
    erase a real brief event, so a caller showing this to a person should show what it
    changed rather than only the result - `clip_analysis.py` does.

    Windows are odd by convention; an even one leans earlier, and at the sequence edges the
    window is truncated rather than padded, so the first and last entries are smoothed
    against fewer neighbours than the middle.
    """
    half = window // 2
    out = []
    for i in range(len(states)):
        chunk = states[max(0, i - half): i + half + 1]
        out.append(Counter(chunk).most_common(1)[0][0])
    return out
