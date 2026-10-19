"""Are the people on the pitch moving? Measured per person, not per pixel.

`vision/motion.motion_cue` is a whole-boundary pixel difference. Floodlights flicker, cameras
re-expose and compression blocks shimmer, so on its own it moves with the light as much as with
the players. This reads the detections the burst already paid for instead: each person's
foot in one frame is matched to the nearest foot in the next, and a person **moved** when the
match is further away than a fraction of their own box height.

Measured in box heights, like the ball is measured in diameters (`vision/ball.py`), so the
same number holds for a player near the camera and one at the far touchline, and does not
need fitting per venue. A match must be within ``MATCH_HEIGHTS`` box heights to count at all;
further than that it is two different people, and the person is left unmatched rather than
read as having sprinted.

This is not a tracker. Two players crossing can swap matches, which changes nothing here
because both moved. What it gives the rule is one number per burst: of the people matched
between consecutive frames, how many moved. Five people standing still for a team talk, a
coaching drill paused for instructions, a queue at the gate - all read as people inside the
boundary and none of them is a game.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pitch_occupancy.vision.detector import Detection

__all__ = ["Movement", "assess", "MOVED_HEIGHTS", "MATCH_HEIGHTS"]

#: A foot that moved by at least this many of its own box heights between burst frames
#: (about a second apart) moved. A walking person covers well over half their height a second.
MOVED_HEIGHTS = 0.15
#: Further than this between frames is a different person, not a fast one.
MATCH_HEIGHTS = 1.5


@dataclass(frozen=True, slots=True)
class Movement:
    """How many of the people matched across the burst moved."""

    matched: int
    moved: int

    @property
    def fraction(self) -> float | None:
        return self.moved / self.matched if self.matched else None


def _pair(a: Sequence[Detection], b: Sequence[Detection]) -> tuple[int, int]:
    """Greedy nearest-foot matching from ``a`` to ``b``: ``(matched, moved)``."""
    candidates = []
    for i, p in enumerate(a):
        height = max(float(p.height), 1.0)
        px, py = p.foot
        for j, q in enumerate(b):
            qx, qy = q.foot
            step = ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5 / max(height, float(q.height), 1.0)
            if step <= MATCH_HEIGHTS:
                candidates.append((step, i, j))
    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = moved = 0
    for step, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched += 1
        moved += step >= MOVED_HEIGHTS
    return matched, moved


def assess(frames: Sequence[Sequence[Detection]]) -> Movement | None:
    """Movement over a burst of per-frame people, or None with fewer than two frames.

    Pairs are summed over every consecutive pair of frames, so a burst of three gives each
    person two chances to be seen moving.
    """
    if len(frames) < 2:
        return None
    matched = moved = 0
    for a, b in zip(frames, frames[1:], strict=False):
        m, v = _pair(a, b)
        matched += m
        moved += v
    return Movement(matched, moved)
