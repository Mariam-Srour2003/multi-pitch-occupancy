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
from itertools import chain

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.counting import PitchCount
from pitch_occupancy.vision.rules import FrameVerdict, MinuteState, RuleConfig, decide

__all__ = ["FusedState", "fuse", "PRIORITY", "PitchVerdict", "fuse_pitch"]

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


# --- the detector-first path (A36) ---------------------------------------------------------
#
# The max-rule above fuses *classes*. The detector-first path fuses *counts*, and the
# difference is the finding A16 recorded and A36 acted on: a camera sees half a pitch, so
# "one to four people means not playing" applied per camera is wrong on 88 of 278 real
# matches at venue_01, while the two halves summed are not. The hand-count audit for WP9-T2
# measured the same thing on the sample it drew - 18 of 74 play frames had fewer than five
# people inside their own camera's boundary. So the pitch count is the sum over cameras, the
# ball is seen by the pitch if any camera saw it, and the decision table is applied once, to
# the pitch. Per-camera verdicts are kept as evidence and for the disagreement signal.


@dataclass(frozen=True, slots=True)
class PitchVerdict:
    """One pitch, one minute, decided from every camera that could be scored."""

    state: MinuteState
    confidence: float
    trace: tuple[str, ...]
    per_camera: tuple[tuple[str, MinuteState, float], ...]
    disagreed: bool
    count: PitchCount | None = None
    motion: float | None = None
    #: How many cameras contributed a count; fewer than were observed halves the confidence.
    n_scored: int = 0

    @property
    def class3(self) -> Class3 | None:
        from pitch_occupancy.vision.rules import to_class3

        return to_class3(self.state)

    @property
    def decided(self) -> bool:
        return self.state.decided

    @property
    def people_inside(self) -> int | None:
        return None if self.count is None else self.count.people_inside

    @property
    def ball_seen(self) -> bool | None:
        return None if self.count is None else self.count.ball_seen

    @property
    def winning_camera(self) -> str | None:
        """The camera whose frame justifies the minute: one that agrees with the pitch's
        state, the most confident of those; failing that, the most confident decided one.

        An empty half at confidence 1.0 does not explain a PLAYING minute - the half with the
        match does, whatever its confidence - which is why this is not simply the maximum.
        """
        decided = [(cam, state, conf) for cam, state, conf in self.per_camera if state.decided]
        if not decided:
            return None
        agreeing = [entry for entry in decided if entry[1] is self.state]
        pool = agreeing or decided
        return max(pool, key=lambda entry: entry[2])[0]


def fuse_pitch(observations: dict[str, FrameVerdict], cfg: RuleConfig) -> PitchVerdict:
    """Sum the cameras' counts and decide once for the pitch.

    ``observations`` maps camera id -> that camera's `FrameVerdict` from `rules.decide`. A
    camera whose verdict carries no count (row 1, detector unavailable) or a whole-frame count
    (row 2, no boundary) is not summed - it is reported, and its absence halves the pitch
    confidence, because a pitch seen by one of its two cameras is half a pitch. A camera that
    abstained on row 4 (motion without people) *is* summed: its zero and its motion are both
    evidence, and the table reads them again at pitch level.

    Raises on no observations at all, for the reason `fuse` does.
    """
    if not observations:
        raise ValueError(
            "no camera observations to fuse; a pitch with no readable camera has no "
            "state, and defaulting to EMPTY would record real usage as unused"
        )
    per_camera = tuple((cam, v.state, float(v.confidence))
                       for cam, v in sorted(observations.items()))
    scored = {cam: v for cam, v in observations.items()
              if v.count is not None and v.rule != 2}
    if not scored:
        reasons = "; ".join(f"{cam}: {v.trace[-1] if v.trace else v.state.name}"
                            for cam, v in sorted(observations.items()))
        return PitchVerdict(MinuteState.UNCERTAIN, 0.0,
                            (f"no camera could be scored - {reasons}",), per_camera,
                            disagreed=False)

    counts = [v.count for v in scored.values()]
    biggest = max(counts, key=lambda c: c.people_inside)
    balls = [c for c in counts if c.ball_seen]
    best_ball = max(balls, key=lambda c: c.ball_confidence, default=None)
    merged = PitchCount(
        people_inside=sum(c.people_inside for c in counts),
        people_total=sum(c.people_total for c in counts),
        raw_inside=sum(c.raw_inside for c in counts),
        ball_seen=best_ball is not None,
        ball_confidence=best_ball.ball_confidence if best_ball else 0.0,
        ball_area_px=best_ball.ball_area_px if best_ball else 0,
        vehicles_inside=sum(c.vehicles_inside for c in counts),
        hi_vis_people=sum(c.hi_vis_people for c in counts),
        # Spread is a within-camera quantity; the camera that saw the most people speaks.
        spread=biggest.spread,
        people=tuple(chain.from_iterable(c.people for c in counts)),
        balls=tuple(chain.from_iterable(c.balls for c in counts)),
        bounded=True,
    )
    motions = [v.motion for v in scored.values() if v.motion is not None]
    motion = max(motions) if motions else None

    verdict = decide(merged, motion=motion, cfg=cfg)
    trace = [f"{cam}: {v.people} inside" + (", ball seen" if v.ball else "")
             for cam, v in sorted(scored.items())]
    trace += list(verdict.trace)
    confidence = verdict.confidence
    missing = len(observations) - len(scored)
    if missing and verdict.decided:
        confidence = round(confidence * 0.5, 4)
        trace.append(f"{missing} camera(s) could not be scored; confidence halved")
    return PitchVerdict(
        verdict.state, confidence, tuple(trace), per_camera,
        disagreed=len({v.state for v in scored.values()}) > 1,
        count=merged, motion=motion, n_scored=len(scored),
    )
