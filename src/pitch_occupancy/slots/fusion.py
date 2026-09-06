"""Fusing the two cameras that watch one pitch.

Every pitch has exactly two fixed cameras, each seeing roughly one half. A per-camera
verdict is not a pitch verdict: the reference footage contains a half showing only a
goalkeeper while a full match runs on the other half. A camera-level classifier calls that
half EMPTY and is wrong at pitch level.

The baseline rule is **strongest activity wins**, with EMPTY requiring agreement:

    ACTIVE_PLAY on either half        -> ACTIVE_PLAY
    else MAINTENANCE on either half   -> MAINTENANCE
    else (both halves empty)          -> EMPTY

:func:`fuse` also returns whether the halves disagreed. That flag matters twice over: it
is an operational signal (a dirty lens or an occluded view makes one half disagree
persistently), and it is information the max-rule discards. STAN consumes both camera
sequences rather than this fused output precisely so it can learn from the disagreement -
see WP5-T6.
"""

from __future__ import annotations

from dataclasses import dataclass

from pitch_occupancy.data.taxonomy import Class3

__all__ = ["FusedState", "fuse", "PRIORITY"]

#: Strongest activity first. A half showing play outranks a half showing nothing.
PRIORITY: tuple[Class3, ...] = (
    Class3.ACTIVE_PLAY,
    Class3.MAINTENANCE_NON_SPORTING,
    Class3.EMPTY,
)


@dataclass(frozen=True, slots=True)
class FusedState:
    state: Class3
    confidence: float
    disagreed: bool
    per_camera: tuple[tuple[str, Class3, float], ...]

    @property
    def n_cameras(self) -> int:
        return len(self.per_camera)


def fuse(observations: dict[str, tuple[Class3 | str, float]]) -> FusedState:
    """Fuse per-camera predictions into one pitch state.

    ``observations`` maps camera id -> (class, confidence). Confidence for the fused state
    is that of the camera whose observation won, not an average: averaging across a camera
    that saw nothing would dilute the evidence of the one that saw the match.

    Raises:
        ValueError: if no observations are supplied - a pitch with no working camera has
            no state, and inventing EMPTY would silently bill a slot as unused.
    """
    if not observations:
        raise ValueError(
            "no camera observations to fuse; a pitch with no readable camera has no "
            "state, and defaulting to EMPTY would record real usage as unused"
        )

    per_camera = tuple(
        (cam, Class3(state), float(conf)) for cam, (state, conf) in sorted(observations.items())
    )
    states = [s for _, s, _ in per_camera]

    for candidate in PRIORITY:
        if candidate in states:
            winning = max((c for c in per_camera if c[1] == candidate), key=lambda c: c[2])
            return FusedState(
                state=candidate,
                confidence=winning[2],
                disagreed=len(set(states)) > 1,
                per_camera=per_camera,
            )
    raise AssertionError("unreachable: every Class3 value appears in PRIORITY")
