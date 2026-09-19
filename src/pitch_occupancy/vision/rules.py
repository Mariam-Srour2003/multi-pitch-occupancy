"""The per-minute vocabulary and the decision table of the detector-first path (A36, WP9).

Three things live here.

**`MinuteState`** is what one camera, or one pitch, is in for one minute: the four folder
classes of `data/taxonomy.Class4`, plus **UNCERTAIN** - an abstention. The four values are
the folder names, deliberately, so a state reads the same whether it came from this path or
from a label; a test pins that the two enums agree. UNCERTAIN is not a fifth class. It is the
state a minute is in when the system declines to say - no detector, no boundary, movement
without anyone found - and `slots/aggregate.py` treats it like a minute that was never
captured: it counts against the capture floor and can push a slot toward REVIEW and nowhere
else.

**`FrameVerdict`** is what a classification *records*, as opposed to what it returns.
`worker.Classifier` returns two values, a class and a confidence, and every surface that
wanted to know *why* - which boundary, how many people, whether a gate spoke - rebuilt the
answer from pieces (A35). A verdict carries the pieces, and a `trace`: the steps that led to
the state, in order, in words a reviewer can read.

**`decide`** is the decision table registered in `thesis/preregistration.md` A36, as code,
with `RuleConfig` holding every number it reads. Two of those numbers are requirements and
not fits - `play_min = 5` and `small_group_max = 4` are the facility's rule - and the rest
are fitted on venue_01 camera A alone (WP9-T5) and frozen in `configs/rules.json` with the
commit that froze them. Until then the file says ``frozen_at: null`` and the deployed entry
points refuse it; experiments pass through, because measuring an unfrozen rule is how it
gets frozen.

The table, first matching row wins. ``n`` is people whose feet stand inside the boundary,
persisted over the burst and summed across the pitch's cameras (`slots/fusion.fuse_pitch`);
``ball`` is a ball seen inside in any burst frame; ``m`` is the burst motion cue.

| # | condition | state | note |
|---|---|---|---|
| 1 | detector unavailable | UNCERTAIN | a missing detector is not an empty pitch |
| 2 | no boundary | UNCERTAIN | mandatory on the deployed path |
| 3 | n = 0, m low or unmeasured | EMPTY | confidence lowered by what the filter dropped |
| 4 | n = 0, m high | UNCERTAIN | something moved and nobody was found |
| 5 | vehicle inside, or hi-vis with n <= 4 | MAINTENANCE | best effort; unevaluable |
| 6 | 1 <= n <= 4 | PEOPLE_NOT_PLAYING | with or without a ball |
| 7 | n >= 5, ball | PLAYING | highest confidence |
| 8 | n >= 5, no ball, still, clustered | PEOPLE_NOT_PLAYING | only if `row8_enabled` |
| 9 | n >= 5, no ball | PLAYING | lower confidence; absence of a ball is not evidence (A17) |

Rows 4 and 8 read the motion cue and rows 5 and 8 read cues that have no fitted threshold
yet; each is skipped, not guessed, while its threshold is ``null``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

from pitch_occupancy.config import CONFIGS_DIR
from pitch_occupancy.data.taxonomy import Class3, Class4
from pitch_occupancy.vision.counting import PitchCount

__all__ = [
    "MinuteState", "FrameVerdict", "RuleConfig", "RULES_PATH", "decide",
    "from_class3", "from_class4", "to_class3",
]

RULES_PATH = CONFIGS_DIR / "rules.json"


class MinuteState(StrEnum):
    """One minute's state for one camera or one pitch. Values are the folder names."""

    EMPTY = "1_empty"
    PLAYING = "2_playing"
    PEOPLE_NOT_PLAYING = "3_people_not_playing"
    MAINTENANCE = "4_maintenance"
    #: The system declined to say. See the module docstring for what that is not.
    UNCERTAIN = "UNCERTAIN"

    @property
    def decided(self) -> bool:
        return self is not MinuteState.UNCERTAIN


#: The three-class collapse the whole thesis reports in. UNCERTAIN has no Class3: it is the
#: absence of one, and a caller asking for it gets None rather than a guess.
_TO_CLASS3 = {
    MinuteState.EMPTY: Class3.EMPTY,
    MinuteState.PLAYING: Class3.ACTIVE_PLAY,
    MinuteState.PEOPLE_NOT_PLAYING: Class3.MAINTENANCE_NON_SPORTING,
    MinuteState.MAINTENANCE: Class3.MAINTENANCE_NON_SPORTING,
}

#: The probe cannot tell `3_people_not_playing` from `4_maintenance` - it was trained on
#: three classes - so its C3 lands on the folder that has real frames. Six of them.
_FROM_CLASS3 = {
    Class3.EMPTY: MinuteState.EMPTY,
    Class3.ACTIVE_PLAY: MinuteState.PLAYING,
    Class3.MAINTENANCE_NON_SPORTING: MinuteState.PEOPLE_NOT_PLAYING,
}


def to_class3(state: MinuteState | str) -> Class3 | None:
    """The reporting class of a state, or None for UNCERTAIN."""
    return _TO_CLASS3.get(MinuteState(state))


def from_class3(state: Class3 | str) -> MinuteState:
    """The state a three-class verdict maps to. See `_FROM_CLASS3` for where C3 lands."""
    return _FROM_CLASS3[Class3(state)]


def from_class4(label: Class4 | str) -> MinuteState:
    """A folder label as a state; the values coincide, and this says so in one place."""
    return MinuteState(Class4(label).value)


@dataclass(frozen=True, slots=True)
class FrameVerdict:
    """What one classification found, and how it got there.

    ``trace`` is the rule path in order - "boundary slot_..._camA keeps 49% of the frame",
    "5 people persisted over 3 frames, ball seen 0.31", "row 7 -> PLAYING 0.80". It is the
    first thing an operator disputing a verdict asks for and was, until A36, reconstructed
    by hand from four fields on three pages.

    ``people``, ``ball`` and ``motion`` are None when the corresponding step did not run,
    which is a different fact from zero and is kept different: a detector that was not
    consulted has not established that nobody was there (`vision/explain.py`).
    """

    state: MinuteState
    confidence: float
    trace: tuple[str, ...] = ()
    polygon: list[list[float]] | None = None
    #: The store key the boundary was found under, ``"given"`` when the caller supplied one,
    #: None when there was none.
    boundary_key: str | None = None
    people: int | None = None
    ball: bool | None = None
    ball_confidence: float = 0.0
    motion: float | None = None
    #: What the probe said before any gate spoke, when a probe spoke at all.
    probed: Class3 | None = None
    gated: bool = False
    model_key: str = ""
    #: The detector path's full count, for the overlay and the fusion; None on the probe path.
    count: PitchCount | None = None
    #: Which row of the table decided; 0 when no table was consulted (the probe path).
    rule: int = 0

    @property
    def class3(self) -> Class3 | None:
        return to_class3(self.state)

    @property
    def decided(self) -> bool:
        return self.state.decided

    def as_pair(self) -> tuple[Class3, float]:
        """The `worker.Classifier` shape, for callers that still want it.

        Raises on UNCERTAIN rather than inventing a class: the two-value contract has no way
        to say "declined", and a caller that needs one has to look at the verdict.
        """
        cls = self.class3
        if cls is None:
            raise ValueError("an UNCERTAIN verdict has no (class, confidence) form")
        return cls, self.confidence


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Every number `decide` reads, and where each one came from.

    ``play_min`` and ``small_group_max`` are requirements (A36). The detector fields name the
    model the counts come from (`vision/detector.DETECTORS`). Everything else is fitted on
    venue_01 camera A in WP9-T5; a ``None`` threshold means the cue it gates is not consulted,
    which is how rows 4, 5 and 8 stay off until they have been measured rather than guessed.
    """

    detector: str = "yolov8n"
    imgsz: int = 1280
    tiles: int = 1
    play_min: int = 5
    small_group_max: int = 4
    person_conf: float = 0.25
    ball_conf: float = 0.10
    #: Minimum person box height in pixels as a line in the foot row's frame fraction:
    #: ``intercept + slope * (foot_y / frame_height)``. Both None = filter off.
    min_height_intercept: float | None = None
    min_height_slope: float | None = None
    #: Burst-gap motion thresholds (`vision/motion.motion_cue` between consecutive burst
    #: frames). Not `motion.MOTION_THRESHOLD`, which was fitted at a 15 s gap.
    motion_lo: float | None = None
    motion_hi: float | None = None
    #: Mean pairwise foot distance over the boundary diagonal below which a group is
    #: "clustered" (row 8).
    cluster_max: float | None = None
    row8_enabled: bool = False
    hi_vis_min_fraction: float | None = None
    burst_frames: int = 3
    burst_spacing_s: float = 1.0
    frozen_at: str | None = None
    frozen_commit: str | None = None
    tuned_on: str | None = None

    @property
    def frozen(self) -> bool:
        return bool(self.frozen_at)

    def min_height_at(self, frame_height: int) -> Callable[[int], float] | None:
        if self.min_height_intercept is None and self.min_height_slope is None:
            return None
        intercept = self.min_height_intercept or 0.0
        slope = self.min_height_slope or 0.0
        height = max(int(frame_height), 1)
        return lambda foot_y: intercept + slope * (foot_y / height)

    @classmethod
    def load(cls, path: Path | None = None) -> RuleConfig:
        """Read `configs/rules.json`. Unknown keys and underscored comment keys are ignored;
        a missing file raises, because a deployment with no rule file has no rule."""
        source = path or RULES_PATH
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def decide(
    count: PitchCount | None,
    *,
    motion: float | None,
    cfg: RuleConfig,
    require_boundary: bool = True,
) -> FrameVerdict:
    """The decision table. Pure: a count, a motion cue and a config in, a verdict out.

    ``count`` is None when the detector did not run (row 1). ``count.bounded`` False means no
    boundary was applied: row 2 on the deployed path, a labelled whole-frame answer on the
    interactive one (``require_boundary=False``). ``motion`` is the burst motion cue, None
    on a single frame.
    """
    key = cfg.detector
    if count is None:
        return FrameVerdict(MinuteState.UNCERTAIN, 0.0, model_key=key, rule=1,
                            trace=("row 1: detector unavailable - not checked, not empty",))
    trace: list[str] = []
    if not count.bounded:
        if require_boundary:
            return FrameVerdict(
                MinuteState.UNCERTAIN, 0.0, model_key=key, count=count, rule=2,
                trace=("row 2: no boundary - the whole frame is not this pitch",))
        trace.append("no boundary: the whole frame was counted, neighbours included")

    n = count.people_inside
    ball = count.ball_seen
    trace.append(
        f"{n} people inside" + (f" ({count.raw_inside} before the height filter)"
                                if count.raw_inside != n else "")
        + (f", ball seen {count.ball_confidence:.2f}" if ball else ", no ball seen")
        + (f", motion {motion:.3f}" if motion is not None else ", motion unmeasured"))

    def verdict(state: MinuteState, confidence: float, rule: int, why: str) -> FrameVerdict:
        trace.append(f"row {rule}: {why} -> {state.name} {confidence:.2f}")
        return FrameVerdict(state, round(confidence, 4), trace=tuple(trace), people=n,
                            ball=ball, ball_confidence=count.ball_confidence, motion=motion,
                            model_key=key, count=count, rule=rule)

    if n == 0:
        if motion is not None and cfg.motion_hi is not None and motion >= cfg.motion_hi:
            return verdict(MinuteState.UNCERTAIN, 0.0, 4,
                           f"nobody found but motion {motion:.3f} >= {cfg.motion_hi:.3f}")
        return verdict(MinuteState.EMPTY, _clip(1.0 - 0.15 * count.raw_inside, 0.6, 1.0), 3,
                       "nobody standing inside the boundary")

    if count.vehicles_inside >= 1 or (count.hi_vis_people >= 1 and n <= cfg.small_group_max):
        why = (f"{count.vehicles_inside} vehicle(s) inside" if count.vehicles_inside
               else f"{count.hi_vis_people} hi-vis in a group of {n}")
        return verdict(MinuteState.MAINTENANCE, 0.5, 5, why + " (best effort, unevaluated)")

    if n <= cfg.small_group_max:
        confidence = min(0.9, 0.5 + 0.1 * (cfg.play_min - n)) - (0.2 if ball else 0.0)
        return verdict(MinuteState.PEOPLE_NOT_PLAYING, confidence, 6,
                       f"{n} <= {cfg.small_group_max} people" + (", ball or not" if ball else ""))

    if ball:
        return verdict(MinuteState.PLAYING, min(1.0, 0.8 + 0.05 * (n - cfg.play_min)), 7,
                       f"{n} >= {cfg.play_min} people and a ball")

    if (cfg.row8_enabled and motion is not None and cfg.motion_lo is not None
            and motion < cfg.motion_lo and count.spread is not None
            and cfg.cluster_max is not None and count.spread < cfg.cluster_max):
        return verdict(MinuteState.PEOPLE_NOT_PLAYING, 0.5, 8,
                       f"{n} people, no ball, motion {motion:.3f} < {cfg.motion_lo:.3f}, "
                       f"spread {count.spread:.3f} < {cfg.cluster_max:.3f}")

    return verdict(MinuteState.PLAYING, min(0.85, 0.6 + 0.05 * (n - cfg.play_min)), 9,
                   f"{n} >= {cfg.play_min} people, no ball seen (ball recall is low)")
