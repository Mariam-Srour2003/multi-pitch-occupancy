"""The person gate: nobody standing on the pitch is not a match in progress (A16).

Measured on 521 recorded frames from venue_01 camera B, counting inside the camera's boundary:

| class | n | median count | zero |
|---|---|---|---|
| EMPTY | 243 | 0 | **89%** |
| ACTIVE_PLAY | 278 | 6 | **0.4%** |

A count threshold alone beats the fitted probe on that split - `PLAY if >= 2` gives recall
0.9604 against false-play 0.0453, a balanced score of **+0.9152**, where the probe manages
+0.8849 on the full training set and +0.7014 pruned. No training and no venue memorisation,
which is why it is worth having beside a probe that has both.

**Two details carry it**, and the earlier attempt did neither:

- **Inside the boundary.** Spectators behind a fence and staff in a dugout are not people on
  the pitch, and on this footage they are most of what a detector finds.
- **The foot of the box, not its centre.** Someone standing at the touchline has their centre
  over the pitch and their feet outside it, and it is the feet that say where they stand.

**One direction, like `MotionGate`.** The gate turns ACTIVE_PLAY into EMPTY when it finds
nobody; it never turns EMPTY into ACTIVE_PLAY. Finding nobody is strong evidence against a
match - 0.4% of real play frames - while finding somebody is not evidence for one, since a
groundskeeper and a person crossing the pitch both count as somebody.

**What it deliberately does not do.** The three-class version of this rule fails: a third of
genuine ACTIVE_PLAY frames show four or fewer people inside the boundary, because a camera
sees part of a pitch and a detector misses distant players. "One to four people means not
playing" would be wrong on 88 real matches out of 278, so the count is *evidence toward* C3
and never a verdict of it.

**Failure is silence, not a guess.** A missing detector returns `None` from `detect_people`,
and the gate leaves the verdict alone rather than treating "not checked" as "found nobody" -
which would turn a broken install into a system that reports every pitch empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitch_occupancy.data.taxonomy import Class3

__all__ = ["DETECT_CONFIDENCE", "DETECT_IMGSZ", "count_inside", "PersonGate"]

#: Long edge the detector runs at. The default 640 shrinks a distant player in a 1080p frame
#: to a few pixels; 1280 was the setting every number above was measured at, and raising it
#: further changed nothing on the clip while costing time.
DETECT_IMGSZ = 1280

#: Confidence for a person box. 0.25 is the detector's own default and the value the
#: distribution above was measured at. Lowering it adds false positives on an empty pitch,
#: which is the error this gate exists to avoid making.
DETECT_CONFIDENCE = 0.25


def count_inside(image_bgr, polygon) -> int | None:
    """How many people are standing inside the boundary. ``None`` if not checked.

    Without a polygon every detection counts, which is the pre-boundary behaviour and is
    reported rather than refused - a camera whose outline has not been derived yet should
    still get a count, and the caller can see there was no boundary.
    """
    from pitch_occupancy.vision.explain import detect_people

    boxes = detect_people(image_bgr, confidence=DETECT_CONFIDENCE, imgsz=DETECT_IMGSZ)
    if boxes is None:
        return None
    if polygon is None:
        return len(boxes)

    import cv2

    height, width = np.asarray(image_bgr).shape[:2]
    mask = np.zeros((height, width), np.uint8)
    pts = np.array([[int(x * width), int(y * height)] for x, y in polygon], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    inside = 0
    for x1, _y1, x2, y2 in boxes:
        foot_x, foot_y = (x1 + x2) // 2, y2
        if 0 <= foot_y < height and 0 <= foot_x < width and mask[foot_y, foot_x]:
            inside += 1
    return inside


@dataclass(frozen=True, slots=True)
class PersonGate:
    """Overrule ACTIVE_PLAY when nobody is standing inside the boundary."""

    def apply(self, state: Class3, image_bgr, polygon=None) -> tuple[Class3, int | None]:
        """Return the possibly-overruled state and the count, or ``(state, None)``.

        The detector only runs when the verdict is ACTIVE_PLAY, because that is the only
        verdict this gate can change. At one frame per camera per minute a second of CPU is
        affordable; spending it on frames the answer cannot alter is not.
        """
        if state is not Class3.ACTIVE_PLAY:
            return state, None
        count = count_inside(image_bgr, polygon)
        if count is None:
            return state, None
        if count == 0:
            return Class3.EMPTY, 0
        return state, count
