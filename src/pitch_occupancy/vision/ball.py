"""Is there a ball, and is it being played with? (2026-09-21)

`count_inside` answers "was a ball detected in this frame". A40 made that answer decide
whether a pitch is in use, which turned out to ask more of it than it can carry. Reported
from use, and then measured on the operator's own clips:

    clip                  truth   frames with a ball   best conf   median movement
    maint night.mp4         C3           6 of 6           0.30         0.0000
    Maint day.mp4           C3           1 of 6           0.17            -
    not playing day.mp4     C3           1 of 6           0.23            -
    playing day.mp4         C2           5 of 6           0.84         0.0429
    playing day 4.mp4       C2           3 of 6           0.56         0.0098
    playing.mp4             C2           2 of 6           0.71         0.0333

Two shapes fall out of that table and they need different treatment.

**A false ball fires once.** `Maint day` and `not playing day` each have exactly one sighting
across six frames, at 0.17 and 0.23 - a bright stud, a bin lid, a patch of line paint. Real
balls recur. `persist` used to set `ball_seen` if **any** frame in the burst had one, so a
single flash was enough to satisfy the rule; now a ball has to be seen in at least
``min_frames`` of them. That is a requirement in the same sense as `play_min`, not a fitted
threshold: "twice" is what distinguishes a thing from a flicker.

**A real ball that never moves is not a ball in play.** `maint night` is the one clip in the
set with a rock-solid ball - six sightings of six, 0.30 confidence, and it is genuinely there;
the overlay shows it plainly on the turf. It has also not moved by a single pixel, because it
is lying on the grass while three people work around it. The facility's rule is that a game
has a ball **in it**, and a ball nobody is touching is furniture. `thesis/labelling_protocol.md`
§2.3 has said the same thing from the start: "an unattended ball ... does not make a pitch
occupied."

**Movement is measured in ball diameters**, which makes the number scale-free: it holds for a
ball seven pixels across at the far touchline and one forty pixels across in the foreground,
and it does not have to be re-fitted per venue the way `motion_lo` does. It is still a number
and `MOVED_DIAMETERS` says how it was chosen and on how little - the first value tried was a
whole diameter, and it turned a genuine match into C3. A scale-free threshold is not the same
thing as no threshold, and writing the first draft of this docstring as though it were is
the mistake this paragraph replaces.

**What this deliberately does not do.** It does not track. Sightings are matched by taking the
most confident ball in each frame, which is wrong whenever two balls are genuinely in play,
and `most_at_once` records when that happened so the trace can say so rather than the code
pretend otherwise. It also cannot tell a fast ball from two different false positives a long
way apart: `playing day 2` shows a "movement" of 0.62 of the frame in half a second, which no
football does. That clip is C2 anyway so the answer is right by luck, and saying so here is
cheaper than a plausibility ceiling fitted on one example.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from pitch_occupancy.vision.counting import PitchCount

__all__ = ["BallEvidence", "assess", "MIN_FRAMES"]

#: Burst frames a ball must appear in before it counts as a ball rather than a flicker.
#: A requirement, not a fit - see the module docstring.
MIN_FRAMES = 2

#: Multiples of its own diameter a ball must travel between sightings to count as in play.
#:
#: **A quarter, and the number is a noise floor rather than a fit.** Measured over five
#: frames half a second apart on the operator's clips:
#:
#:     maint night.mp4   C3   ball in 5 of 5   travelled 0.00 diameters
#:     playing day 4.mp4 C2   ball in 2 of 5   travelled 0.50 diameters
#:     playing day.mp4   C2   ball in 4 of 5   travelled 2.59 diameters
#:
#: The first attempt used one whole diameter, which landed *between* two genuine matches and
#: turned `playing day 4` into C3 - 10 of 13 clips correct became 9. The stationary ball is
#: not merely slow, it is **0.00**: the same 7-pixel blob in the same place across two
#: seconds. A quarter of a diameter is under the localisation noise of a detection that
#: small, so this asks "did it move at all, beyond jitter" rather than "did it move far",
#: which is the question `labelling_protocol.md` §2.3 actually poses.
#:
#: Three clips is not a calibration and this is written down as a floor to be re-checked
#: when there is more than one maintenance recording to check it against (WP9-T5).
MOVED_DIAMETERS = 0.25


@dataclass(frozen=True, slots=True)
class BallEvidence:
    """What a burst establishes about the ball, as opposed to what one frame saw."""

    #: A ball was detected in at least one frame. Never sufficient on its own.
    seen: bool
    #: Seen in at least `MIN_FRAMES` frames of the burst.
    persisted: bool
    #: Persisted **and** moved. None when the burst is too short to tell, which is a
    #: different fact from False and is kept different.
    in_play: bool | None
    confidence: float
    frames_seen: int
    frames_checked: int
    #: The most balls detected simultaneously in any one frame. Above one, the matching
    #: below is guesswork and the trace says so.
    most_at_once: int
    #: Median distance between consecutive sightings, in ball diameters. None when fewer
    #: than two frames had one.
    travelled: float | None
    why: str

    @property
    def usable(self) -> bool:
        """Whether this burst can answer "is the ball in play" at all."""
        return self.in_play is not None


def _centre_and_size(count: PitchCount) -> tuple[tuple[float, float], float] | None:
    """The most confident ball's centre and diameter in pixels, or None."""
    if not count.balls:
        return None
    best = max(count.balls, key=lambda d: d.conf)
    x1, y1, x2, y2 = best.box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), max(float(max(x2 - x1, y2 - y1)), 1.0)


def assess(frames: Sequence[PitchCount], *, min_frames: int = MIN_FRAMES) -> BallEvidence:
    """Read a burst's ball detections as evidence about a ball in play.

    ``frames`` are the per-frame counts of one burst, in order. A single frame can establish
    that a ball was *seen* and never that it is *in play*, so ``in_play`` is None there - the
    caller is expected to skip the clause and say it skipped it, which is what `rules.decide`
    does with every cue it cannot check.
    """
    if not frames:
        raise ValueError("a burst needs at least one frame")

    sightings = [(i, _centre_and_size(c)) for i, c in enumerate(frames)]
    present = [(i, s) for i, s in sightings if s is not None]
    n_seen = len(present)
    confidence = max((c.ball_confidence for c in frames), default=0.0)
    most_at_once = max((len(c.balls) for c in frames), default=0)
    checked = len(frames)

    if n_seen == 0:
        return BallEvidence(
            seen=False, persisted=False, in_play=False, confidence=0.0, frames_seen=0,
            frames_checked=checked, most_at_once=0, travelled=None,
            why=f"no ball seen in {checked} frame(s)")

    persisted = n_seen >= min_frames
    if not persisted:
        return BallEvidence(
            seen=True, persisted=False, in_play=False, confidence=confidence,
            frames_seen=n_seen, frames_checked=checked, most_at_once=most_at_once,
            travelled=None,
            why=(f"a ball in {n_seen} of {checked} frames at {confidence:.2f} - "
                 f"below the {min_frames} it takes to be a ball rather than a flicker")
            if checked >= min_frames else
            (f"a ball at {confidence:.2f}, but {checked} frame(s) cannot show it twice"))

    if n_seen < 2:
        # Persisted by a min_frames of 1, so there is nothing to measure movement against.
        return BallEvidence(
            seen=True, persisted=True, in_play=None, confidence=confidence,
            frames_seen=n_seen, frames_checked=checked, most_at_once=most_at_once,
            travelled=None,
            why=f"a ball at {confidence:.2f} in one frame - movement not measurable")

    steps = []
    for (_, (a, size_a)), (_, (b, size_b)) in zip(present, present[1:], strict=False):
        diameter = (size_a + size_b) / 2.0
        steps.append(float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) / diameter)
    travelled = float(median(steps))
    moved = travelled >= MOVED_DIAMETERS
    crowd = (f"; {most_at_once} balls at once, so these sightings may not be one ball"
             if most_at_once > 1 else "")
    return BallEvidence(
        seen=True, persisted=True, in_play=moved, confidence=confidence,
        frames_seen=n_seen, frames_checked=checked, most_at_once=most_at_once,
        travelled=travelled,
        why=(f"a ball in {n_seen} of {checked} frames at {confidence:.2f}, "
             + (f"moving {travelled:.1f}x its own width between sightings" if moved else
                f"moving only {travelled:.1f}x its own width - lying still, not in play")
             + crowd))
