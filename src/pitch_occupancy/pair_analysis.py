"""Two videos of the same pitch, one timeline: the rule applied to both cameras at once.

`clip_analysis.py` answers *"what is happening in this video?"* for one camera. A pitch has
two, each seeing roughly one half (`slots/fusion.py`), and a single camera's answer is not the
pitch's: a half showing a goalkeeper while the match runs at the other end reads as "one
person, not playing". The worker has fused the two since A36 - `fuse_pitch` sums the counts and
applies the decision table once - but only over the live schedule. Nothing let a person hand
it **two files** and watch the pitch-level rule work. This does.

What crosses the cameras, and where each piece lives:

* **People are added.** Camera A's count inside its boundary plus camera B's
  (`fuse_pitch`, A36). Five people split three and two is a game; per camera it is two
  small groups.
* **The ball belongs to the pitch.** Seen by *either* camera, it is seen. `require_ball` is
  read from `configs/rules.json` like everywhere else - turned off on 2026-09-29 - so whether
  the ball is *required* is the rule file's decision, and this module only reports where it was.
* **Motion is the stronger of the two halves**, and a camera that could not be scored halves
  the pitch confidence. Both are `fuse_pitch`'s, unchanged.

**And four cross-checks that only exist with two cameras.** They are *flags*, not rows of the
table: none of them changes a verdict, because a new decision rule is an amendment and not a
page feature. They mark the samples a reviewer should look at.

* ``depends_on_sum`` - neither camera alone reaches `play_min`, the two together do. The
  verdict then rests on the halves not overlapping: somebody standing in the strip both cameras
  see is counted twice. The check reruns the table on the *larger single camera's* count (the
  lower bound if the views overlapped completely) and flags the sample when the state changes.
* ``ball_one_side`` - the ball was seen by one camera only. Expected most of the time - it is
  in one half - but it says *which* half the play evidence came from.
* ``split_view`` - the cameras' own verdicts disagree. Normal for a half-pitch drill; a split
  that holds for the whole clip is the dirty-lens or occluded-view signal `fusion.py` names.
* ``one_side_blind`` - one camera's frame could not be read or scored at this moment.

**Synchronisation is the caller's.** Two exports of the same hour rarely start on the same
second, so ``offset_s`` says how far camera B's recording starts *after* camera A's; sampling
runs on camera A's clock over the stretch both files cover. Nothing here guesses the offset
from the pixels - a wrong guess would pair a goal at one end with the kick-off at the other.

**Nothing is retained**, for the reason `clip_analysis.py` gives.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from pitch_occupancy.clip_analysis import DEFAULT_INTERVAL_S, DEFAULT_WINDOW, MAX_SAMPLES
from pitch_occupancy.clip_analysis import _segments as segment_runs
from pitch_occupancy.slots.stan import majority_smooth
from pitch_occupancy.vision.rules import MinuteState, decide

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from pitch_occupancy.clip_analysis import ClipSegment
    from pitch_occupancy.pipeline import Pipeline

__all__ = ["CameraSide", "PairSample", "PairAnalysis", "analyse_pair", "tally", "CAMERAS"]

#: The two camera names on the page and in every trace. Letters rather than ids because an
#: uploaded file names no camera; `analyse_pair` takes real ids for the boundary lookup.
CAMERAS: tuple[str, str] = ("A", "B")


@dataclass(frozen=True, slots=True)
class CameraSide:
    """One camera at one moment: its own verdict, before the other camera is added."""

    camera: str
    #: False when no frame could be decoded at this moment. The other fields are then empty.
    readable: bool
    state: MinuteState | None = None
    confidence: float = 0.0
    people: int | None = None
    ball: bool | None = None
    motion: float | None = None
    rule: int = 0


@dataclass(frozen=True, slots=True)
class PairSample:
    """One moment on both cameras, and the pitch's answer from the two together."""

    index: int
    #: On camera A's clock. Camera B was read at ``t_s - offset_s``.
    t_s: float
    sides: tuple[CameraSide, CameraSide]
    #: The pitch verdict from `Pipeline.fuse` - the sum, the shared ball, the table once.
    raw: MinuteState
    confidence: float
    smoothed: MinuteState
    people: int | None
    ball: bool | None
    rule_trace: tuple[str, ...] = ()
    #: What the table says on the larger single camera's count instead of the sum.
    if_overlapping: MinuteState | None = None

    @property
    def corrected(self) -> bool:
        return self.raw is not self.smoothed

    @property
    def depends_on_sum(self) -> bool:
        return self.if_overlapping is not None and self.if_overlapping is not self.raw

    @property
    def ball_sides(self) -> tuple[str, ...]:
        return tuple(s.camera for s in self.sides if s.ball)

    @property
    def ball_one_side(self) -> bool:
        return len(self.ball_sides) == 1

    @property
    def split_view(self) -> bool:
        states = {s.state for s in self.sides if s.readable and s.state is not None
                  and s.state.decided}
        return len(states) > 1

    @property
    def one_side_blind(self) -> bool:
        return any(not s.readable or s.people is None for s in self.sides)


@dataclass(frozen=True, slots=True)
class PairAnalysis:
    duration_s: float
    interval_s: float
    window: int
    offset_s: float
    #: Where on camera A's clock both files are covered, which is what was sampled.
    start_s: float
    end_s: float
    samples: list[PairSample] = field(default_factory=list)
    segments: list[ClipSegment] = field(default_factory=list)
    interval_widened: bool = False
    model: str = ""

    @property
    def n_corrected(self) -> int:
        return sum(s.corrected for s in self.samples)

    def share(self, state: MinuteState) -> float:
        if not self.samples:
            return 0.0
        return sum(s.smoothed is state for s in self.samples) / len(self.samples)

    def count(self, flag: str) -> int:
        return sum(bool(getattr(s, flag)) for s in self.samples)

    @property
    def dominant(self) -> MinuteState | None:
        if not self.segments:
            return None
        return max(self.segments, key=lambda s: s.duration_s).state

    def summary(self) -> str:
        if not self.samples:
            return "no moment both cameras cover could be read"
        parts = [f"{len(self.samples)} moments on both cameras every {self.interval_s:g}s"]
        if self.n_corrected:
            parts.append(f"{self.n_corrected} corrected by their neighbours")
        if self.count("depends_on_sum"):
            parts.append(f"{self.count('depends_on_sum')} depend on adding the halves")
        if self.count("one_side_blind"):
            parts.append(f"{self.count('one_side_blind')} with one camera unscored")
        return " · ".join(parts)


def _duration(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open {path.name} as video")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        # The single-camera path can walk a file of unknown length; pairing cannot, because
        # the overlap of two unknown lengths is unknown. Refused rather than guessed.
        raise ValueError(f"{path.name} does not report its length, so it cannot be paired")
    return frames / fps


def _read(capture, t_s: float) -> np.ndarray | None:
    import cv2

    if t_s < 0:
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    ok, frame = capture.read()
    return frame if ok and frame is not None else None


def _smooth(states: list[MinuteState], people: list[int | None], window: int,
            play_min: int) -> list[MinuteState]:
    """Majority smoothing over the pitch states, under the same limits as one camera's.

    `clip_analysis._respect_the_count` for the pitch: the neighbours may not make a moment
    play when fewer than `play_min` people were counted on both cameras together, EMPTY when
    somebody was, or anything but EMPTY when nobody was. And an abstention is never smoothed
    into a verdict nor out of one - UNCERTAIN means nobody could tell, which the neighbours do
    not change.
    """
    smoothed = majority_smooth(states, window)
    out = []
    for state, own, n in zip(smoothed, states, people, strict=True):
        abstained = MinuteState.UNCERTAIN in (state, own)
        contradicts = n is not None and ((state is MinuteState.ACTIVE_PLAY and n < play_min)
                                         or (state is MinuteState.EMPTY and n > 0)
                                         or (state is not MinuteState.EMPTY and n == 0))
        if abstained or contradicts:
            state = own
        out.append(state)
    return out


def _if_overlapping(fused, observations, cfg) -> MinuteState | None:
    """The table on the larger single camera's count: the sum's lower bound.

    If the two views overlapped completely, everybody would be on both cameras and the true
    count would be the larger of the two, not their total. A state that changes under that
    reading is one that rests on the cameras *not* overlapping.
    """
    counts = [obs.verdict.count for obs in observations.values()
              if obs.verdict.count is not None and obs.verdict.rule != 2]
    if fused.count is None or len(counts) < 2:
        return None
    larger = max(counts, key=lambda c: c.people_inside)
    lower = replace(fused.count, people_inside=larger.people_inside,
                    raw_inside=larger.raw_inside, bounded=True)
    return decide(lower, motion=fused.motion, cfg=cfg).state


def analyse_pair(
    path_a: Path | str,
    path_b: Path | str,
    pipeline: Pipeline,
    *,
    offset_s: float = 0.0,
    interval_s: float = DEFAULT_INTERVAL_S,
    window: int = DEFAULT_WINDOW,
    max_samples: int = MAX_SAMPLES,
    polygons: tuple[list[list[float]] | None, list[list[float]] | None] = (None, None),
) -> PairAnalysis:
    """Sample both videos at the same moments, observe each camera, and fuse them.

    ``pipeline`` should be the detector-first one (`pipeline.assemble` with a detector key):
    that path *counts*, so its fusion adds the halves. A probe pipeline is accepted and fuses
    by strongest-activity-wins instead, which is `Pipeline.fuse`'s documented behaviour and has
    no counts to add - the overlap check is then silent.

    Raises ``ValueError`` when either file cannot be opened, when the offset leaves no stretch
    both cover, or when no moment could be read on either camera.
    """
    import cv2

    if interval_s <= 0:
        raise ValueError(f"interval_s must be positive, not {interval_s}")
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd number, not {window}")

    sources = (Path(path_a), Path(path_b))
    dur_a, dur_b = (_duration(p) for p in sources)
    burst_n, spacing = pipeline.burst
    tail = (burst_n - 1) * spacing
    # Camera A's clock. B's frame for time t is at t - offset_s in B's own file.
    start = max(0.0, offset_s)
    end = min(dur_a, dur_b + offset_s) - tail
    if end <= start:
        raise ValueError(
            f"with camera B starting {offset_s:g}s after camera A, the two recordings do not "
            f"overlap (A is {dur_a:.0f}s, B is {dur_b:.0f}s)")

    widened = False
    if (end - start) / interval_s > max_samples:
        interval_s = (end - start) / max_samples
        widened = True

    cfg = pipeline.rules
    captures = [cv2.VideoCapture(str(p)) for p in sources]
    rows: list[tuple[float, tuple[CameraSide, CameraSide], object, MinuteState | None]] = []
    try:
        previous: dict[str, np.ndarray | None] = dict.fromkeys(CAMERAS)
        t = start
        while t < end and len(rows) < max_samples:
            observations = {}
            sides = []
            for camera, capture, shift, polygon in zip(
                    CAMERAS, captures, (0.0, offset_s), polygons, strict=True):
                frames = [_read(capture, t - shift + k * spacing) for k in range(burst_n)]
                frames = [f for f in frames if f is not None]
                if not frames:
                    sides.append(CameraSide(camera, readable=False))
                    continue
                obs = pipeline.observe_minute(camera, frames, previous[camera],
                                              polygon=polygon)
                previous[camera] = frames[-1]
                observations[camera] = obs
                v = obs.verdict
                sides.append(CameraSide(camera, True, v.state, v.confidence, v.people,
                                        v.ball, v.motion, v.rule))
            if observations:
                fused = pipeline.fuse(observations)
                lower = (_if_overlapping(fused, observations, cfg)
                         if pipeline.kind == "detector" and cfg is not None else None)
                rows.append((t, tuple(sides), fused, lower))
            t += interval_s
    finally:
        for capture in captures:
            capture.release()

    if not rows:
        raise ValueError("no moment could be read on either camera; the files may be "
                         "truncated or an unsupported codec")

    states = [fused.state for _, _, fused, _ in rows]
    people = [fused.people_inside for _, _, fused, _ in rows]
    play_min = cfg.play_min if cfg is not None else 5
    smoothed = _smooth(states, people, window, play_min)
    samples = [
        PairSample(index=i, t_s=t, sides=sides, raw=fused.state,
                   confidence=fused.confidence, smoothed=smooth, people=fused.people_inside,
                   ball=fused.ball_seen, rule_trace=fused.trace, if_overlapping=lower)
        for i, ((t, sides, fused, lower), smooth) in enumerate(zip(rows, smoothed, strict=True))
    ]
    # Segment boundaries on camera A's clock; the opening run starts where sampling did.
    segments = segment_runs(samples, end + tail, interval_s)
    return PairAnalysis(
        duration_s=end + tail - start, interval_s=interval_s, window=window,
        offset_s=offset_s, start_s=start, end_s=end + tail, samples=samples,
        segments=segments, interval_widened=widened, model=pipeline.model_key,
    )


def tally(samples: list[PairSample]) -> Counter[str]:
    """How often each camera's own verdict matched the pitch's, for the summary tiles."""
    agree: Counter[str] = Counter()
    for s in samples:
        for side in s.sides:
            if side.readable and side.state is s.raw:
                agree[side.camera] += 1
    return agree
