"""What the detector saw, drawn on the frame (A36, WP9-T4).

The probe path's explanation is a heatmap, and it has to be: a logistic regression on pooled
features has no objects in it, only positions that push the score one way or the other
(`vision/explain.py`). The detector-first path has objects, so its explanation is the objects -
people in one colour, the ball in another, the pitch boundary outlined, the count and the rule
that fired written on the frame. A reader checks "five people and a ball, so PLAYING" by
looking, not by trusting a caption.

**Masks when the model has them, boxes when it does not.** A segmentation detector gives a
per-instance mask and the overlay fills it; a detection model gives a rectangle and the overlay
outlines that instead of filling it, because a filled rectangle over a person claims a precision
the box does not have. The count is the same either way - a person is placed by the foot of
their box in both (`vision/counting.py`) - so the overlay is the only thing a segmentation model
changes, and that is the trade WP9-T2 measured.

**Inside and outside the boundary are drawn differently.** A detection whose feet fall outside
the outline is drawn dimmed, because "the detector found six people and counted three" is the
single most common thing a reader needs explained on this footage, and a picture that draws
only the counted three cannot explain it.

**Redaction is separate and composes.** :func:`draw_verdict` draws; :func:`publishable`
redacts first and then draws. The overlay is computed from a verdict taken on the *original*
frame and drawn over the redacted copy, which has identical geometry - so the explanation stays
faithful while the picture stays publishable, which is the rule `experiments/make_xai_figures.py`
has followed since it was written (`thesis/ethics.md` §2). Streamed frames to an operator are
not redacted, as they are not today (`vision/walkthrough.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pitch_occupancy.vision import counting, roi
from pitch_occupancy.vision.detector import PERSON, SPORTS_BALL, VEHICLES, Detection
from pitch_occupancy.vision.rules import FrameVerdict, MinuteState

__all__ = ["COLOURS", "STATE_COLOURS", "draw_verdict", "publishable", "legend"]

#: BGR. Person and ball are deliberately far apart in hue and both far from turf green, so a
#: reader can tell them apart on a floodlit night frame and on a daylight one.
COLOURS = {
    "person": (255, 128, 0),      # azure: the counted people
    "person_outside": (140, 110, 90),  # the same, dimmed: found but not on the pitch
    "ball": (0, 200, 255),        # amber - the opposite side of the wheel from azure
    "vehicle": (200, 0, 200),     # magenta: the maintenance cue
    "boundary": (0, 255, 255),    # yellow, as `roi.outline` has always drawn it
    "text": (255, 255, 255),
    "shadow": (0, 0, 0),
}

#: The badge's background, by state. Read at a glance across a contact sheet.
STATE_COLOURS = {
    MinuteState.EMPTY: (90, 90, 90),
    MinuteState.ACTIVE_PLAY: (40, 160, 40),
    MinuteState.MAINTENANCE_NON_SPORTING: (30, 140, 200),
    MinuteState.UNCERTAIN: (60, 60, 160),
}

#: Alpha for a filled instance mask. Low enough that the player stays visible under it - the
#: point is to show what was found, not to hide it.
MASK_ALPHA = 0.45


def _scaled(image_bgr: np.ndarray) -> tuple[float, int]:
    """Font scale and line thickness for this frame's size, so a 720p and a 1080p frame
    carry the same apparent weight."""
    height, width = image_bgr.shape[:2]
    return max(0.45, min(width, height) / 900.0), max(2, min(width, height) // 400)


def _label(image_bgr, text: str, origin, *, colour, scale: float, thickness: int) -> None:
    import cv2

    x, y = origin
    cv2.putText(image_bgr, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                COLOURS["shadow"], thickness + 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thickness, cv2.LINE_AA)


def _draw_instance(canvas, det: Detection, *, colour, thickness: int, fill: bool) -> None:
    import cv2

    x1, y1, x2, y2 = det.box
    if det.mask is not None and fill:
        tint = np.zeros_like(canvas)
        tint[det.mask] = colour
        blended = cv2.addWeighted(canvas, 1.0 - MASK_ALPHA, tint, MASK_ALPHA, 0.0)
        np.copyto(canvas, np.where(det.mask[:, :, None], blended, canvas))
        contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)


def draw_verdict(
    image_bgr: np.ndarray,
    verdict: FrameVerdict,
    *,
    instances: Sequence[Detection] = (),
    polygon: Sequence[Sequence[float]] | None = None,
    trace: bool = True,
) -> np.ndarray:
    """A copy of the frame with what the detector found and what the rule made of it.

    ``instances`` are the detections to draw; when empty, the verdict's own counted people and
    balls are used. Anything whose foot point falls outside ``polygon`` is dimmed rather than
    dropped, so a reader can see what was found and not counted.
    """
    import cv2

    canvas = np.ascontiguousarray(image_bgr.copy())
    height, width = canvas.shape[:2]
    scale, thickness = _scaled(canvas)
    polygon = polygon if polygon is not None else verdict.polygon
    drawn = list(instances) or list(
        (verdict.count.people + verdict.count.balls) if verdict.count else ())

    # Which instances were counted is `vision/counting.py`'s decision, and it is read back
    # here rather than recomputed from the polygon. A second implementation of "is this inside"
    # is a second implementation that can drift from the first, which is the defect this
    # project keeps finding in itself - and it would drift silently, since a mis-drawn overlay
    # looks like a correct one. Keyed by class and box, which identify a detection; the mask
    # is a numpy array and does not compare.
    counted = {(d.cls, d.box) for d in
               ((verdict.count.people + verdict.count.balls) if verdict.count else ())}

    def inside(det: Detection) -> bool:
        if verdict.count is not None:
            return (det.cls, det.box) in counted
        # No count at all (a verdict that never reached the rule): fall back to the polygon,
        # and to "everything" when there is not even one. Through `counting`, so the drawing
        # and the counting agree about the frame edge - this had its own copy of the
        # membership test and its own copy of the bug fixed there on 2026-09-20.
        if not polygon:
            return True
        point = det.centre if det.cls == SPORTS_BALL else det.foot
        return counting.inside(counting.polygon_mask(polygon, (height, width)), point)

    for det in drawn:
        if det.cls == SPORTS_BALL:
            colour = COLOURS["ball"]
        elif det.cls in VEHICLES:
            colour = COLOURS["vehicle"]
        else:
            colour = COLOURS["person"] if inside(det) else COLOURS["person_outside"]
        _draw_instance(canvas, det, colour=colour, thickness=thickness,
                       fill=inside(det) or det.cls != PERSON)
        if det.cls == SPORTS_BALL and det.mask is None:
            # A ball is a few pixels across; a ring around it is visible where its box is not.
            x, y = det.centre
            cv2.circle(canvas, (x, y), max(12, det.height * 2), COLOURS["ball"],
                       thickness, cv2.LINE_AA)

    if polygon:
        canvas = roi.outline(canvas, list(polygon), colour=COLOURS["boundary"],
                             thickness=thickness)

    badge = STATE_COLOURS.get(verdict.state, STATE_COLOURS[MinuteState.UNCERTAIN])
    pad = int(8 * scale) + 4
    line_h = int(28 * scale) + 6
    headline = f"{verdict.state.name}  {verdict.confidence:.2f}"
    counts = []
    if verdict.people is not None:
        counts.append(f"{verdict.people} inside")
    if verdict.ball is not None:
        counts.append("ball " + (f"{verdict.ball_confidence:.2f}" if verdict.ball else "none"))
    if verdict.motion is not None:
        counts.append(f"motion {verdict.motion:.2f}")
    subtitle = " · ".join(counts)

    box_w = int(max(len(headline), len(subtitle)) * 15 * scale) + 2 * pad
    box_h = line_h * (2 if subtitle else 1) + pad
    cv2.rectangle(canvas, (0, 0), (min(box_w, width), box_h), badge, -1)
    _label(canvas, headline, (pad, line_h - pad // 2), colour=COLOURS["text"], scale=scale,
           thickness=thickness)
    if subtitle:
        _label(canvas, subtitle, (pad, 2 * line_h - pad // 2), colour=COLOURS["text"],
               scale=scale * 0.75, thickness=max(1, thickness - 1))

    if trace and verdict.trace:
        # A band behind the trace, because a white line over floodlit turf is unreadable
        # exactly where the reader most needs to check the rule against the picture.
        steps = list(verdict.trace[-4:])
        step_h = int(line_h * 0.8)
        band_top = max(0, height - pad - step_h * len(steps))
        band = canvas[band_top:height].copy()
        canvas[band_top:height] = cv2.addWeighted(band, 0.35,
                                                  np.zeros_like(band), 0.65, 0.0)
        y = height - pad
        for step in reversed(steps):
            _label(canvas, step[:110], (pad, y), colour=COLOURS["text"], scale=scale * 0.62,
                   thickness=max(1, thickness - 1))
            y -= step_h
    return canvas


def publishable(
    image_bgr: np.ndarray,
    verdict: FrameVerdict,
    *,
    instances: Sequence[Detection] = (),
    polygon: Sequence[Sequence[float]] | None = None,
    trace: bool = True,
) -> np.ndarray:
    """:func:`draw_verdict` over a redacted copy: pixelated people, then a floor of blur.

    Two layers, because one is not enough - a missed detection must still not be an
    identifiable face (`thesis/ethics.md` §2). The boxes come from the verdict rather than a
    second detector run, so the redaction and the drawing are about the same detections.

    The overlay is drawn *after* the blur so the outline, the masks and the text stay sharp;
    the frame underneath is what gets destroyed, which is the half that needs to be.
    """
    from pitch_occupancy.vision.explain import redact_frame

    drawn = list(instances) or list(
        (verdict.count.people + verdict.count.balls) if verdict.count else ())
    boxes = [d.box for d in drawn if d.cls == PERSON]
    redacted = redact_frame(image_bgr, boxes)
    return draw_verdict(redacted, verdict, instances=drawn, polygon=polygon, trace=trace)


def legend() -> list[tuple[str, tuple[int, int, int]]]:
    """What each colour means, for a figure caption or a page's key."""
    return [
        ("people counted on the pitch", COLOURS["person"]),
        ("people found outside the boundary", COLOURS["person_outside"]),
        ("ball", COLOURS["ball"]),
        ("vehicle (maintenance cue)", COLOURS["vehicle"]),
        ("pitch boundary", COLOURS["boundary"]),
    ]
