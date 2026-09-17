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

**The C3 rule, and why the ball clause is what makes it possible (A18).** "One to four people
means not playing" on its own is wrong on **88 of 278** real matches at venue_01, because a
camera sees part of a pitch and a detector misses distant players. Adding the clause the rule
was actually stated with - *and no ball* - takes that to **0 of 278**, and across the nine clip
venues, 396 frames of genuine play at sites the model has never seen, the combined rule fires
on **3**, or 0.8%. That is the best-evidenced cross-venue cost in this project.

**What is not evidenced is the other half.** There are **6 recorded C3 frames in the entire
corpus**, one slot at one camera, and the rule identifies **1** of them: three show nobody
inside the boundary at all, and two show a ball. A cost measured on 396 frames and a benefit
measured on 6 is not a balanced case, and the honest reading is that this rule is *safe* rather
than *shown to work*. It is enabled because the alternative is a deployed path that cannot
return C3 at all, and because on the unseen clip it is right - the minute with one person
walking and no ball is the minute a person would call not-playing.

**Failure is silence, not a guess.** A missing detector returns `None` from `detect_people`,
and the gate leaves the verdict alone rather than treating "not checked" as "found nobody" -
which would turn a broken install into a system that reports every pitch empty.

**The ball is recorded and decides nothing (A17).** COCO's `sports ball` comes out of the same
forward pass, so it costs nothing, and at venue_01 it looks like the strongest signal in the
project: found on 278 of 278 ACTIVE_PLAY frames and 8 of 243 EMPTY ones. It is still not used
as a rule, for a reason the venue_01 number cannot show. Across the nine clip venues - every
frame genuine play - a ball is found in **40%**, ranging from 6% to 89% by venue, where the
person count finds people in 100%. At an unseen site the absence of a ball is therefore not
evidence of the absence of play, and a rule resting on it would be a rule resting on venue_01,
which is 13 distinct scenes wearing 278 frames. Its presence is informative and is written to
the record; its absence says nothing and is treated as saying nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitch_occupancy.data.taxonomy import Class3

__all__ = ["BALL_CONFIDENCE", "DETECT_CONFIDENCE", "DETECT_IMGSZ", "Counted",
           "count_inside", "detect_inside", "PersonGate"]

#: Long edge the detector runs at. The default 640 shrinks a distant player in a 1080p frame
#: to a few pixels; 1280 was the setting every number above was measured at, and raising it
#: further changed nothing on the clip while costing time.
DETECT_IMGSZ = 1280

#: Confidence for a person box. 0.25 is the detector's own default and the value the
#: distribution above was measured at. Lowering it adds false positives on an empty pitch,
#: which is the error this gate exists to avoid making.
DETECT_CONFIDENCE = 0.25

#: Confidence for a ball box, deliberately lower than for a person. A football on a CCTV frame
#: is about 400 square pixels and the detector is correspondingly unsure of it; holding it to
#: the person threshold would measure the threshold rather than the object. It can be lower
#: here precisely *because* the ball decides nothing - a false ball costs a wrong note in the
#: record, where a false person would cost a wrong verdict.
BALL_CONFIDENCE = 0.10

#: COCO class ids: person, and `sports ball`.
PERSON_CLASS, BALL_CLASS = 0, 32


@dataclass(frozen=True, slots=True)
class Counted:
    """What one detector pass found inside the boundary.

    ``ball`` is ``True`` when a ball was seen and ``False`` when the frame was checked and
    none was, which is not the same as "there is no ball" - see the module docstring.
    """

    people: int
    ball: bool
    ball_confidence: float = 0.0


def detect_inside(image_bgr, polygon) -> Counted | None:
    """People and ball inside the boundary from one forward pass. ``None`` if not checked.

    Without a polygon every detection counts, which is the pre-boundary behaviour and is
    reported rather than refused - a camera whose outline has not been derived yet should
    still get a count, and the caller can see there was no boundary.

    **A person is placed by the foot of their box and a ball by its centre.** Someone standing
    at the touchline has their centre over the pitch and their feet outside it, and it is the
    feet that say where they stand; a ball spends much of its time in the air and has no feet
    to stand on.
    """
    from pitch_occupancy.vision.explain import detect_objects

    found = detect_objects(image_bgr, confidence=min(DETECT_CONFIDENCE, BALL_CONFIDENCE),
                           imgsz=DETECT_IMGSZ, classes=(PERSON_CLASS, BALL_CLASS))
    if found is None:
        return None

    mask = None
    height, width = np.asarray(image_bgr).shape[:2]
    if polygon is not None:
        import cv2

        mask = np.zeros((height, width), np.uint8)
        pts = np.array([[int(x * width), int(y * height)] for x, y in polygon], np.int32)
        cv2.fillPoly(mask, [pts], 1)

    def inside(box, *, foot: bool) -> bool:
        if mask is None:
            return True
        x1, y1, x2, y2 = box
        x = (x1 + x2) // 2
        y = y2 if foot else (y1 + y2) // 2
        return 0 <= y < height and 0 <= x < width and bool(mask[y, x])

    people, ball_conf = 0, 0.0
    for cls, box, conf in found:
        if cls == PERSON_CLASS and conf >= DETECT_CONFIDENCE and inside(box, foot=True):
            people += 1
        elif cls == BALL_CLASS and conf >= BALL_CONFIDENCE and inside(box, foot=False):
            ball_conf = max(ball_conf, conf)
    return Counted(people=people, ball=ball_conf > 0.0, ball_confidence=ball_conf)


def count_inside(image_bgr, polygon) -> int | None:
    """How many people are standing inside the boundary. ``None`` if not checked."""
    counted = detect_inside(image_bgr, polygon)
    return None if counted is None else counted.people


@dataclass(frozen=True, slots=True)
class PersonGate:
    """Weaken an ACTIVE_PLAY verdict when the detector does not support it.

    Two overrules, both in the same direction - a play verdict can be reduced and never
    manufactured:

    * **nobody inside the boundary** becomes EMPTY (A16)
    * **a small group with no ball** becomes C3, present but not playing (A18)

    `small_group_max` is the largest group the second rule will call not-playing. Four is the
    number the rule was stated with and the number the tables in the module docstring were
    measured at. Setting it to 0 disables the C3 overrule and leaves A16's behaviour exactly
    as it was.
    """

    small_group_max: int = 4

    def inspect(self, state: Class3, image_bgr,
                polygon=None) -> tuple[Class3, Counted | None]:
        """Return the possibly-overruled state and everything the pass found.

        The detector only runs when the verdict is ACTIVE_PLAY, because that is the only
        verdict this gate can change. At one frame per camera per minute a second of CPU is
        affordable; spending it on frames the answer cannot alter is not.

        **The ball does not veto the EMPTY overrule.** A frame with nobody on the pitch and a
        ball inside the boundary is still turned to EMPTY - a ball lying on an empty pitch is
        a ball lying on an empty pitch, and 8 of venue_01's 243 recorded EMPTY frames have one.

        **It does gate the C3 overrule, and the two clauses do different jobs.** A *found*
        ball vetoes C3 outright, which is the sound direction - a found ball is a found ball,
        whatever the 40% cross-venue recall says. The rule does also require the ball to be
        absent, which is the unsound direction, and the count is what bounds the damage: at
        the nine unseen venues a real match almost never shows four or fewer people inside the
        boundary, so the unsound clause is only ever consulted on 9 frames out of 396 and
        fires wrongly on 3. It is protected by the count, not by the ball.
        """
        if state is not Class3.ACTIVE_PLAY:
            return state, None
        counted = detect_inside(image_bgr, polygon)
        if counted is None:
            return state, None
        if counted.people == 0:
            return Class3.EMPTY, counted
        if 0 < counted.people <= self.small_group_max and not counted.ball:
            return Class3.MAINTENANCE_NON_SPORTING, counted
        return state, counted

    def apply(self, state: Class3, image_bgr, polygon=None) -> tuple[Class3, int | None]:
        """:meth:`inspect` reduced to the person count, for callers that want only that."""
        state, counted = self.inspect(state, image_bgr, polygon)
        return state, None if counted is None else counted.people
