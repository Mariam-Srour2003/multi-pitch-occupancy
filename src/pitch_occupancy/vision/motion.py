"""The motion gate: a pitch where nothing moved is not a match in progress (A14).

The rule is one line - if the frame is called ACTIVE_PLAY but barely differs from the
previous frame of the same camera, call it EMPTY instead - and it earns its place on evidence
that took a video from an unseen venue to obtain:

| on a 234-second clip from a venue with no labelled frames | false-play on 13 empty frames |
|---|---|
| whole frame, no boundary | 0.74 |
| pitch boundary only | 0.38 |
| **boundary + this gate** | **0.15** |

**It is a rule rather than a feature, and that is not a stylistic choice.** Appended as a
769th dimension beside 768 backbone features the same cue changes results by exactly 0.0000:
one standardised scalar among that many already-predictive ones is shrunk to nothing by the
regulariser. A gate sits outside the probe where it cannot be shrunk.

**Why it was nearly discarded.** Measured on venue_01 camera B - the only split in the corpus
with a real class mix - the same rule scored -0.0159 and was written up as useless. That split
cannot show it: the model already scores 0.9386 there, so there is no failure for a gate to
catch. The negative result was an answer about a working model; this is the operational
question, which is about a failing one.

**What it cannot do.** A person walking across an empty pitch moves, so the gate says PLAY.
Distinguishing that from a match needs the class the corpus does not have -
`3_people_not_playing` holds six recorded frames - and no cue computed from two frames will
supply it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitch_occupancy.data.taxonomy import Class3

__all__ = ["MOTION_THRESHOLD", "WORK_SIZE", "motion_cue", "MotionGate"]

#: Frames are compared at this size, not full resolution. The cue is a scene-level "did
#: anything move", and at 160x90 it is cheap, insensitive to sensor noise, and - the part that
#: matters - the size every measurement behind `MOTION_THRESHOLD` was made at. Changing it
#: changes the scale of the cue and invalidates the threshold.
WORK_SIZE = (160, 90)

#: Mean absolute difference below which an ACTIVE_PLAY verdict is overruled.
#:
#: **Fitted on venue_01's recorded EMPTY and PLAY frames at a 15-second gap**, as the value
#: maximising (play-recall - false-play) there, and never on the footage it was then tested on.
#: It transferred: on an unseen venue it cut false-play from 0.38 to 0.15 and cost nothing the
#: boundary had already fixed.
#:
#: It is a single number on a scale that is **not calibrated across venues** - the corpus
#: median for ACTIVE_PLAY is 1.575 at venue_01 and 2.930 at the clip venues, because cameras,
#: compression and how busy the football is all differ. One threshold cannot be right
#: everywhere, and this one is right where it was measured and plausible elsewhere. A
#: per-camera threshold needs empty pitches at that camera, which is item 1 of
#: `thesis/data_requests.md`.
MOTION_THRESHOLD = 1.098

#: Gap the threshold was fitted at. The cue grows with the gap - 0.891 median for EMPTY at
#: 15s against 1.035 at 60s - so a threshold applied at a different spacing is compared
#: against a different quantity. `MotionGate` records the gap it was built for rather than
#: assuming the caller's.
FITTED_GAP_S = 15


def motion_cue(prev_bgr, cur_bgr) -> float:
    """Mean absolute greyscale difference between two frames of one camera.

    Whole frame, deliberately, and not restricted to the pitch boundary: the question is
    whether *anything in the scene* changed, and a ball leaving the pitch or a player at the
    touchline is still evidence that a game is running. The boundary is applied to the
    classifier, not to this.
    """
    import cv2

    a = cv2.resize(cv2.cvtColor(np.asarray(prev_bgr), cv2.COLOR_BGR2GRAY), WORK_SIZE)
    b = cv2.resize(cv2.cvtColor(np.asarray(cur_bgr), cv2.COLOR_BGR2GRAY), WORK_SIZE)
    return float(np.abs(b.astype(np.float32) - a.astype(np.float32)).mean())


@dataclass(frozen=True, slots=True)
class MotionGate:
    """Overrule ACTIVE_PLAY when nothing moved since the previous frame.

    Only that direction. The gate never turns EMPTY into ACTIVE_PLAY: a still frame is
    evidence against a match, but movement is not evidence for one - a groundskeeper with a
    broom moves, and so does a person crossing the pitch.
    """

    threshold: float = MOTION_THRESHOLD
    fitted_gap_s: int = FITTED_GAP_S

    def apply(self, state: Class3, prev_bgr, cur_bgr) -> tuple[Class3, float | None]:
        """Return the possibly-overruled state and the cue, or ``(state, None)``.

        ``prev_bgr`` of ``None`` means the first frame of a camera, which has no predecessor
        and therefore no cue. The gate stays silent rather than guessing: an imputed value
        would be the gate deciding on evidence it does not have, on exactly the frames it
        cannot describe.
        """
        if prev_bgr is None:
            return state, None
        cue = motion_cue(prev_bgr, cur_bgr)
        if state is Class3.ACTIVE_PLAY and cue < self.threshold:
            return Class3.EMPTY, cue
        return state, cue
