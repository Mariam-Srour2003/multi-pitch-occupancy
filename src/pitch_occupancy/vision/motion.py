"""How much a camera's picture changed between two frames, inside the pitch boundary.

A scene-level cue: the mean absolute greyscale difference at a small fixed size. The rules
read it between the frames of one burst (about a second apart, `rules.RuleConfig.motion_hi`
and `motion_play_min`); the per-person movement in `vision/movement.py` is the sharper
signal, because floodlights, auto-exposure and compression move every pixel a little too.
"""

from __future__ import annotations

import numpy as np

__all__ = ["WORK_SIZE", "motion_cue"]

#: Frames are compared at this size, not full resolution: the cue is "did anything move",
#: and at 160x90 it is cheap and insensitive to sensor noise. Changing it changes the scale
#: of the cue, so every threshold fitted on it would need fitting again.
WORK_SIZE = (160, 90)


def motion_cue(prev_bgr, cur_bgr, polygon=None) -> float:
    """Mean absolute greyscale difference between two frames of one camera.

    **Inside the boundary when one is given.** The first version measured the whole frame,
    argued for on the grounds that a ball leaving the pitch is still evidence of a game. On
    the clip this gate was built from that reasoning cost it a verdict: between minutes 12 and
    13 the only thing that moved was a person walking *behind the goal*, outside the outline,
    and the whole-frame cue rose to 1.610 - over the threshold, so the gate stayed silent and
    an empty pitch was reported as a match.

    Movement outside the pitch is movement at the venue, not play on the pitch, and a cue
    that answers "is this pitch in use" must not be driven by it.
    """
    import cv2

    a = cv2.resize(cv2.cvtColor(np.asarray(prev_bgr), cv2.COLOR_BGR2GRAY), WORK_SIZE)
    b = cv2.resize(cv2.cvtColor(np.asarray(cur_bgr), cv2.COLOR_BGR2GRAY), WORK_SIZE)
    diff = np.abs(b.astype(np.float32) - a.astype(np.float32))
    if polygon is None:
        return float(diff.mean())
    w, h = WORK_SIZE
    mask = np.zeros((h, w), np.uint8)
    pts = np.array([[int(round(x * w)), int(round(y * h))] for x, y in polygon], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    inside = mask.astype(bool)
    # A boundary that rounds away to nothing at this size would make the mean undefined;
    # falling back to the whole frame keeps the cue defined.
    return float(diff[inside].mean()) if inside.any() else float(diff.mean())
