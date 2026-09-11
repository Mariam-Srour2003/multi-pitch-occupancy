"""Classify a short clip at a fixed interval and report when the state changed (WP6-T6).

The slot pipeline answers *"was this booked hour used?"* over an hour, at one frame a
minute, against a schedule. This answers a smaller question that has no schedule in it:
hand it a video, and it says what the pitch was doing and when that changed.

Three decisions shape it, and the first two are about not lying to the person reading it.

**Smoothing is shown, never applied silently.** A single frame classified EMPTY between two
ACTIVE_PLAY frames is more likely a misread than four seconds of genuinely empty pitch, and
:func:`majority_smooth` corrects it. But the same correction erases a real brief event -
somebody crossing an empty pitch, a ball retrieved mid-match - and at a ten-second sampling
interval there is no way to tell those apart from the samples alone. So every corrected
sample is flagged, the raw prediction is kept beside the smoothed one, and the caller is
told how many corrections were made. A reader who disagrees can see exactly what changed.

**A boundary is reported as the interval it falls in, not as a timestamp.** Sampling every
ten seconds locates a change to within ten seconds and no better. Reporting "play started at
1:30" from a sample at 1:30 and its predecessor at 1:20 states more than was observed, so
:class:`ClipSegment` carries the window the boundary lies in and the UI prints it that way.

**Nothing is retained.** This exists to be pointed at footage of identifiable people, so it
takes a path, reads it, and keeps no copy. Whoever writes that file is responsible for
removing it; :func:`analyse_clip` never writes one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.slots.stan import majority_smooth

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "ClipSample", "ClipSegment", "ClipAnalysis", "analyse_clip",
    "DEFAULT_INTERVAL_S", "DEFAULT_WINDOW", "MAX_SAMPLES",
]

#: Sample every ten seconds by default. Fine enough that a substitution or a break shows up,
#: coarse enough that a five-minute clip is thirty inferences rather than thousands.
DEFAULT_INTERVAL_S: float = 10.0

#: Three: correct a sample only when **both** neighbours disagree with it. The narrowest
#: window that does anything, and therefore the one that erases the least. Five would need
#: two consecutive samples to survive, which at this interval is twenty seconds of real
#: event discarded.
DEFAULT_WINDOW: int = 3

#: A hard ceiling on inferences per clip, so an hour-long file at a one-second interval
#: cannot turn one request into an afternoon. Reached means "sampled coarser", not
#: "truncated": :func:`analyse_clip` widens the interval to cover the whole clip and says so.
MAX_SAMPLES: int = 360


class _Classifier(Protocol):
    def __call__(self, image_bgr: np.ndarray) -> tuple[Class3, float]: ...


@dataclass(frozen=True, slots=True)
class ClipSample:
    """One sampled frame: what the model said, and what the neighbours made of it."""

    index: int
    t_s: float
    raw: Class3
    confidence: float
    smoothed: Class3

    @property
    def corrected(self) -> bool:
        """True when its neighbours overruled it. The flag the reviewer is looking for."""
        return self.raw is not self.smoothed


@dataclass(frozen=True, slots=True)
class ClipSegment:
    """A stretch of the clip the model reports as one state.

    ``starts_after``/``starts_by`` bracket the boundary rather than naming it. The change
    happened somewhere between the last sample of the previous state and the first sample of
    this one, and sampling cannot say where in that gap. For the first segment both are 0.0,
    because the clip began already in that state.
    """

    state: Class3
    starts_after: float
    starts_by: float
    ends_by: float
    n_samples: int
    n_corrected: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.ends_by - self.starts_by)

    def describe(self) -> str:
        name = self.state.name.lower().replace("_", " ")
        if self.starts_after == self.starts_by:
            when = f"from {_clock(self.starts_by)}"
        else:
            when = f"from between {_clock(self.starts_after)} and {_clock(self.starts_by)}"
        return f"{name} {when} to {_clock(self.ends_by)}"


@dataclass(frozen=True, slots=True)
class ClipAnalysis:
    """Everything the reviewer needs, including what was changed on its behalf."""

    duration_s: float
    interval_s: float
    window: int
    samples: list[ClipSample] = field(default_factory=list)
    segments: list[ClipSegment] = field(default_factory=list)
    unreadable: list[float] = field(default_factory=list)
    interval_widened: bool = False

    @property
    def n_corrected(self) -> int:
        return sum(s.corrected for s in self.samples)

    @property
    def dominant(self) -> Class3 | None:
        if not self.segments:
            return None
        return max(self.segments, key=lambda s: s.duration_s).state

    def summary(self) -> str:
        if not self.samples:
            return "no frame in this clip could be read"
        parts = [f"{len(self.samples)} samples every {self.interval_s:g}s"]
        if self.n_corrected:
            parts.append(f"{self.n_corrected} corrected by its neighbours")
        if self.unreadable:
            parts.append(f"{len(self.unreadable)} unreadable")
        return " · ".join(parts)


def _clock(t_s: float) -> str:
    total = int(round(t_s))
    return f"{total // 60}:{total % 60:02d}"


def _segments(samples: list[ClipSample], duration_s: float,
              interval_s: float) -> list[ClipSegment]:
    """Collapse the smoothed sequence into runs, bracketing each boundary."""
    if not samples:
        return []
    out: list[ClipSegment] = []
    start = 0
    for i in range(1, len(samples) + 1):
        if i == len(samples) or samples[i].smoothed is not samples[start].smoothed:
            run = samples[start:i]
            # The change happened after the previous run's last sample and by this run's
            # first. For the opening run there is nothing before it, so the bracket collapses.
            starts_after = samples[start - 1].t_s if start > 0 else run[0].t_s
            # It held until at least the next sample, or the end of the clip for the last run.
            ends_by = (samples[i].t_s if i < len(samples)
                       else min(duration_s, run[-1].t_s + interval_s))
            out.append(ClipSegment(
                state=run[0].smoothed,
                starts_after=starts_after,
                starts_by=run[0].t_s,
                ends_by=ends_by,
                n_samples=len(run),
                n_corrected=sum(s.corrected for s in run),
            ))
            start = i
    return out


def analyse_clip(
    path: Path | str,
    classify: _Classifier,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    window: int = DEFAULT_WINDOW,
    max_samples: int = MAX_SAMPLES,
) -> ClipAnalysis:
    """Sample ``path`` every ``interval_s`` seconds, classify each frame, and segment it.

    A frame that cannot be decoded is recorded in ``unreadable`` and skipped rather than
    filled in, for the same reason `worker.run_slot` counts a missed minute instead of
    inventing one: a gap is a fact about the footage and a guess is not.

    Raises ``ValueError`` for a file OpenCV cannot open or one with no readable frames,
    because "zero samples" and "an empty pitch" must not arrive as the same answer.
    """
    import cv2

    if interval_s <= 0:
        raise ValueError(f"interval_s must be positive, not {interval_s}")
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd number, not {window}")

    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open {source.name} as video")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration_s = frames / fps if fps > 0 and frames > 0 else 0.0

        widened = False
        if duration_s > 0 and duration_s / interval_s > max_samples:
            # Cover the whole clip coarsely rather than analysing its first few minutes and
            # silently stopping - a partial answer that looks complete is the worse failure.
            interval_s = duration_s / max_samples
            widened = True

        raw: list[tuple[float, Class3, float]] = []
        unreadable: list[float] = []
        t = 0.0
        while duration_s <= 0 or t < duration_s:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                if duration_s <= 0:
                    break  # unknown length: the first unreadable point is the end
                unreadable.append(t)
            else:
                state, confidence = classify(frame)
                raw.append((t, Class3(state), float(confidence)))
            t += interval_s
            if len(raw) + len(unreadable) >= max_samples and duration_s <= 0:
                break
    finally:
        capture.release()

    if not raw:
        raise ValueError(
            f"no frame of {source.name} could be read; it may be truncated or an "
            f"unsupported codec"
        )

    if duration_s <= 0:  # length was unknown, so derive it from what was actually read
        duration_s = raw[-1][0] + interval_s

    smoothed = majority_smooth([state for _, state, _ in raw], window)
    samples = [
        ClipSample(index=i, t_s=t, raw=state, confidence=conf, smoothed=smooth)
        for i, ((t, state, conf), smooth) in enumerate(zip(raw, smoothed, strict=True))
    ]
    return ClipAnalysis(
        duration_s=duration_s,
        interval_s=interval_s,
        window=window,
        samples=samples,
        segments=_segments(samples, duration_s, interval_s),
        unreadable=unreadable,
        interval_widened=widened,
    )
