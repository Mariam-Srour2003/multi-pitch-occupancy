"""The per-minute vocabulary and the decision table of the detector-first path (A36, WP9).

Three things live here.

**`MinuteState`** is what one camera, or one pitch, is in for one minute: the four folder
classes of `data/taxonomy.Label`, plus **UNCERTAIN** - an abstention. The values are
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
``ball`` is a ball seen inside in **any** burst frame; ``m`` is the burst motion cue.

| # | condition | state | note |
|---|---|---|---|
| 1 | detector unavailable | UNCERTAIN | a missing detector is not an empty pitch |
| 2 | no boundary | UNCERTAIN | mandatory on the deployed path |
| 3 | n = 0, m low or unmeasured | **EMPTY** | nobody and nothing moving |
| 4 | n = 0, m high | UNCERTAIN | something moved and nobody was found |
| 5 | 1 <= n <= 4 | **C3** | too few for a game, ball or no ball |
| 6 | n > 4, **and** a ball, **and** motion | **C2 ACTIVE_PLAY** | the only way into play |
| 7 | n > 4, otherwise | **C3** | a crowd that is not playing |

**Rows 6 and 7 are the change of 2026-09-20**, and they invert what A36 registered. A36 made
play the default above the head count and let the ball raise confidence only, because A17
measured cross-venue ball recall at 0.40 and reasoned that absence of a ball is not evidence
of absence of play. The facility's rule is the opposite: a game has a ball in it and people
moving, and a crowd standing on a pitch with no ball is not a booking being used. So play now
has to be **shown**, not assumed - which is a stricter claim and costs recall exactly where
the detector cannot see the ball. That cost is measured rather than argued about; see the
amendment and `results/rule_frame_eval.csv`.

**A required cue that cannot be evaluated is reported, never assumed.** If `require_ball` is
on, the ball is checked and that is that. If `require_motion` is on but the cue was not
measured (a still image has no predecessor) or its threshold is unfitted (`motion_play_min`
is null until WP9-T5), the clause is **skipped and the trace says so** - because the
alternative is a requirement that silently never fires, which is the shape of every guard
this project has found not guarding.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

from pitch_occupancy.config import CONFIGS_DIR
from pitch_occupancy.data.taxonomy import Class3, Label
from pitch_occupancy.vision.counting import PitchCount

__all__ = [
    "MinuteState", "FrameVerdict", "RuleConfig", "RULES_PATH", "decide",
    "from_class3", "from_label", "to_class3",
]

RULES_PATH = CONFIGS_DIR / "rules.json"


class MinuteState(StrEnum):
    """One minute's state for one camera or one pitch: **the three reporting classes, plus
    an abstention**.

    It used to carry the four *folder* classes, so the rule could answer `4_maintenance`
    separately from `3_people_not_playing`. That distinction is gone from the prediction path
    (2026-09-20) and the values are now `Class3`'s own, so a stored state and a reported class
    are the same string and no mapping can drift between them.

    **Why it went.** The corpus holds **6 real `3_people_not_playing` frames and 0 real
    `4_maintenance` frames**, so nothing could ever measure the split - `make_overlay_figures`
    reported it as the confusion A36 pre-accepted, which is a polite way of saying it was
    never checked. A branch the data cannot evaluate is a branch that should not be in the
    deployed path, and the labelling folders keep the finer distinction for whenever the data
    arrives (`data/taxonomy.to_class3` still collapses them).
    """

    EMPTY = "C1_EMPTY"
    ACTIVE_PLAY = "C2_ACTIVE_PLAY"
    MAINTENANCE_NON_SPORTING = "C3_MAINTENANCE_NON_SPORTING"
    #: The system declined to say. See the module docstring for what that is not.
    UNCERTAIN = "UNCERTAIN"

    @property
    def decided(self) -> bool:
        return self is not MinuteState.UNCERTAIN


def to_class3(state: MinuteState | str) -> Class3 | None:
    """The reporting class of a state, or None for UNCERTAIN."""
    state = MinuteState(state)
    return None if state is MinuteState.UNCERTAIN else Class3(state.value)


def from_class3(state: Class3 | str) -> MinuteState:
    """A three-class verdict as a state. One-to-one now, and kept as a function so the
    call sites read the same as they did when it was a mapping."""
    return MinuteState(Class3(state).value)


def from_label(label: Label | str) -> MinuteState:
    """A labelling folder as a state.

    One-to-one since the folders collapsed to three on 2026-09-21, and it still accepts the
    pre-collapse `3_people_not_playing` and `4_maintenance` - both land on C3, which is the
    collapse `data/taxonomy.to_class3` has always made.
    """
    from pitch_occupancy.data.taxonomy import to_class3 as folder_to_class3

    return MinuteState(folder_to_class3(label).value)


@dataclass(frozen=True, slots=True)
class FrameVerdict:
    """What one classification found, and how it got there.

    ``trace`` is the rule path in order - "boundary slot_..._camA keeps 49% of the frame",
    "5 people persisted over 3 frames, ball seen 0.31", "row 6 -> ACTIVE_PLAY 0.80". It is
    the first thing an operator disputing a verdict asks for and was, until A36, rebuilt
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
    #: "clustered" - a third signal that a crowd is standing rather than playing. Optional:
    #: rows 6 and 7 already turn a ball-less or motionless crowd into C3 without it.
    cluster_max: float | None = None
    #: **And it has to be in play, not lying on the grass.** With this on, a ball that
    #: persisted across the burst but moved less than its own width does not satisfy
    #: `require_ball`. Reported from use on 2026-09-21: the operator's one clip with a
    #: rock-solid ball - six sightings of six at 0.30 - is a maintenance clip, and the ball
    #: is furniture beside three people working. `labelling_protocol.md` §2.3 has always
    #: said an unattended ball does not make a pitch occupied. Inert on a single frame,
    #: which cannot tell movement from stillness, and the trace says so.
    require_ball_in_play: bool = True
    #: **A game has a ball in it.** With this on, ACTIVE_PLAY requires one seen inside the
    #: boundary in at least one burst frame. It is the facility's rule and it is strict: A17
    #: measured cross-venue ball recall at 0.40, 0.06-0.89 by venue, so the venues where the
    #: detector cannot see the ball lose genuine matches. The switch exists because that cost
    #: is a decision, and whoever turns it off should be able to.
    require_ball: bool = True
    #: **And people moving.** Checked only when the cue was measured *and* `motion_play_min`
    #: is fitted; otherwise the clause is skipped and the verdict's trace records that it was.
    require_motion: bool = True
    #: The burst motion cue at or above which there is enough movement for a game. Fitted on
    #: venue_01 camera A (WP9-T5); null until then, which leaves `require_motion` inert and
    #: visibly so.
    motion_play_min: float | None = None
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
        + (f", ball seen {count.ball_confidence:.2f}"
           + (f" in {count.ball_frames} of the burst" if count.ball_frames > 1 else "")
           if ball else ", no ball seen")
        + (f", motion {motion:.3f}" if motion is not None else ", motion unmeasured"))

    def verdict(state: MinuteState, confidence: float, rule: int, why: str) -> FrameVerdict:
        trace.append(f"row {rule}: {why} -> {state.name} {confidence:.2f}")
        return FrameVerdict(state, round(confidence, 4), trace=tuple(trace), people=n,
                            ball=ball, ball_confidence=count.ball_confidence, motion=motion,
                            model_key=key, count=count, rule=rule)

    # --- nobody on the pitch ------------------------------------------------------------
    if n == 0:
        if motion is not None and cfg.motion_hi is not None and motion >= cfg.motion_hi:
            return verdict(MinuteState.UNCERTAIN, 0.0, 4,
                           f"nobody found but motion {motion:.3f} >= {cfg.motion_hi:.3f}")
        return verdict(MinuteState.EMPTY, _clip(1.0 - 0.15 * count.raw_inside, 0.6, 1.0), 3,
                       "nobody standing inside the boundary and nothing moving")

    # --- too few for a game -------------------------------------------------------------
    if n <= cfg.small_group_max:
        confidence = min(0.9, 0.5 + 0.1 * (cfg.play_min - n)) - (0.2 if ball else 0.0)
        why = f"{n} <= {cfg.small_group_max} people" + (", ball or not" if ball else "")
        if count.vehicles_inside or count.hi_vis_people:
            why += (f" ({count.vehicles_inside} vehicle(s), {count.hi_vis_people} hi-vis "
                    f"- maintenance, though C3 does not distinguish it)")
            confidence = max(confidence, 0.6)
        return verdict(MinuteState.MAINTENANCE_NON_SPORTING, confidence, 5, why)

    # --- more than four: a game has to be shown, not assumed ----------------------------
    #
    # Each required cue is examined in turn and its verdict recorded, so a frame that fails
    # says *which* clause failed, and a clause that could not be checked says that instead
    # of passing silently.
    missing: list[str] = []
    if cfg.require_ball and not ball:
        missing.append("no ball seen inside the boundary")
    elif cfg.require_ball and cfg.require_ball_in_play:
        # A ball that persisted across the burst but never moved is not one being played
        # with. None means the burst was too short to tell, which is skipped and said.
        if count.ball_in_play is None:
            trace.append("a ball is there, but one frame cannot show whether it moved "
                         "- in-play clause skipped")
        elif not count.ball_in_play:
            missing.append("a ball, but it has not moved between burst frames")
    if ball and count.balls_at_once > 1:
        trace.append(f"{count.balls_at_once} balls detected at once - the burst matched the "
                     f"most confident one, which may not be the same ball each frame")
    still = None
    if cfg.require_motion:
        if motion is None:
            trace.append("motion required but not measured on this frame - clause skipped")
        elif cfg.motion_play_min is None:
            trace.append("motion required but `motion_play_min` is unfitted (WP9-T5) "
                         "- clause skipped")
        elif motion < cfg.motion_play_min:
            still = motion
            missing.append(f"motion {motion:.3f} < {cfg.motion_play_min:.3f}")
    if (still is None and count.spread is not None and cfg.cluster_max is not None
            and count.spread < cfg.cluster_max):
        missing.append(f"clustered, spread {count.spread:.3f} < {cfg.cluster_max:.3f}")

    if missing:
        # A crowd on a pitch that is not playing on it: a team talk, a queue, a group
        # standing about, groundskeeping. C3 covers all of it and the corpus cannot tell
        # them apart, which is why the four-class split left the prediction path.
        return verdict(MinuteState.MAINTENANCE_NON_SPORTING, 0.6, 7,
                       f"{n} > {cfg.small_group_max} people but " + "; ".join(missing))

    shown = ["a ball"] if ball else []
    if motion is not None and cfg.motion_play_min is not None:
        shown.append(f"motion {motion:.3f}")
    return verdict(MinuteState.ACTIVE_PLAY, min(1.0, 0.8 + 0.05 * (n - cfg.play_min)), 6,
                   f"{n} > {cfg.small_group_max} people"
                   + (" with " + " and ".join(shown) if shown else ""))
