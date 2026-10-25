"""Redaction of frames that leave the machine or reach a page: people pixelated, then blurred.

**Faces are pixelated before anything is written to disk.** The footage is of identifiable
people who never agreed to be shown, and a frame image can end up somewhere permanent, so
every path that writes or shows a redacted frame takes the one rule here rather than a copy
of it (`overlay.publishable`, `overlay.detector_pane`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["FLOOR_BLUR_KERNEL", "LIGHT_BLUR_KERNEL", "pixelate_boxes", "redact_frame",
           "redact_heads"]

#: The blur applied to the whole frame regardless of what the detector found. Chosen so a face
#: at this footage's scale is unrecoverable while the pitch, the lines and the crowd/turf
#: boundary - the things an overlay is read against - stay legible.
FLOOR_BLUR_KERNEL = 21


def pixelate_boxes(image_bgr: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
                   *, blocks: int = 16) -> np.ndarray:
    """Pixelate the given regions of a copy of the frame.

    Takes boxes from a detector pass the caller already made, so the redaction is about the
    same detections as the measurement drawn beside it.
    """
    out = image_bgr.copy()
    for x1, y1, x2, y2 in boxes:
        out[y1:y2, x1:x2] = _pixelate(out[y1:y2, x1:x2], blocks)
    return out


def redact_frame(image_bgr: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
                 *, blocks: int = 16, floor_kernel: int = FLOOR_BLUR_KERNEL) -> np.ndarray:
    """Pixelate the given people, then blur the whole frame. **Both, not either.**

    One layer is not enough: a detector that misses a distant player leaves an identifiable
    face, and the floor of blur is what covers the miss. A caller with no boxes still gets
    the floor.
    """
    import cv2

    out = pixelate_boxes(image_bgr, list(boxes or []), blocks=blocks)
    kernel = floor_kernel if floor_kernel % 2 else floor_kernel + 1
    return cv2.GaussianBlur(out, (kernel, kernel), 0)


#: The lighter floor for pictures an operator has to read (saved checks, evidence, pictures
#: sent for a second opinion): bodies, kit and the ball stay clear; a face a few pixels
#: across does not.
LIGHT_BLUR_KERNEL = 5


def redact_heads(image_bgr: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
                 *, floor_kernel: int = LIGHT_BLUR_KERNEL) -> np.ndarray:
    """Pixelate the top third of each person box (the head), then blur the frame lightly.

    For pictures that must stay readable as evidence. Both layers, as `redact_frame`: the
    light floor covers the people the detector missed.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    heads = []
    for x1, y1, x2, y2 in boxes or []:
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        top, bottom = max(0, min(h, y1)), max(0, min(h, y1 + max(1, (y2 - y1) // 3)))
        if x2 > x1 and bottom > top:
            heads.append((x1, top, x2, bottom))
    out = pixelate_boxes(image_bgr, heads, blocks=3) if heads else image_bgr.copy()
    kernel = floor_kernel if floor_kernel % 2 else floor_kernel + 1
    return cv2.GaussianBlur(out, (kernel, kernel), 0) if kernel > 1 else out


def _pixelate(patch: np.ndarray, blocks: int) -> np.ndarray:
    import cv2

    h, w = patch.shape[:2]
    small = cv2.resize(patch, (max(1, min(blocks, w)), max(1, min(blocks, h))),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
