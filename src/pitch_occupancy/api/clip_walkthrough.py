"""Stream a clip's analysis step by step, showing what the detector found.

`clip_review.py` returns a finished timeline in one response. This returns the same work as
it happens - one newline-delimited JSON record per sampled frame, flushed before the next is
computed - so the page renders step *k* while the detector is still on *k+1*. What a reader
watches is the model running, not an animation replayed over an answer that was already
complete.

A verdict that arrives in one jump asks to be trusted; one that shows the frame with a box
round each person, a ring round the ball and the boundary they were counted inside can be
argued with. Each step is classified by the same pipeline as `/clip/analyse`, and an
explained step carries `overlay.detector_pane`'s drawing and the numbers behind it.

**Pacing is the client's, and the page says so.** The slow motion is a pause the browser adds
between renders, and its "continue without slow motion" button drops that pause. The detector
is working at the same speed either way. The control that changes the actual *work* is
``explain_n``: an explained frame costs a second detector pass for the drawing, so bounding
it is what finishes a long clip sooner.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from pitch_occupancy.api import clip_review, limits
from pitch_occupancy.api.clip_review import (
    MAX_UPLOAD_BYTES,
    _clock,
    _derive_boundary,
    _parse_polygon,
)
from pitch_occupancy.clip_analysis import DEFAULT_INTERVAL_S, MAX_SAMPLES

__all__ = ["router", "JPEG_WIDTH", "EXPLAIN_ALL"]

#: ``explain_n=EXPLAIN_ALL`` draws every sampled frame.
EXPLAIN_ALL: int = -1

router = APIRouter(prefix="/api/v1", tags=["clip review"])

#: The traceback goes here rather than to the browser.
LOG = logging.getLogger(__name__)

#: Frames reach the browser as JPEGs at this width. 640 keeps a sixty-step walkthrough to a
#: few megabytes while leaving the boxes legible. The detector always runs on the frame as
#: decoded - the downscale happens on the way out, never on the way in.
JPEG_WIDTH: int = 640


def _classifier():
    """`clip_review._classifier`, looked up when called rather than bound at import, so a
    test that swaps the classifier on `clip_review` swaps it here too."""
    return clip_review._classifier()


def _jpeg(image_bgr, width: int = JPEG_WIDTH) -> str:
    """A BGR array as a base64 data URI, downscaled for the wire."""
    import cv2

    height, wide = image_bgr.shape[:2]
    if wide > width:
        image_bgr = cv2.resize(image_bgr, (width, int(height * width / wide)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:  # pragma: no cover - encoder failure on a valid array
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _verdict(result) -> tuple[str, float, int | None, bool | None, float | None]:
    """``(predicted, confidence, people, ball, motion)`` from a `FrameVerdict`, or from the
    bare ``(class, confidence)`` pair a scripted classifier returns."""
    if isinstance(result, tuple):
        state, confidence = result
        return str(state), float(confidence), None, None, None
    predicted = result.class3.value if result.decided else result.state.value
    return predicted, float(result.confidence), result.people, result.ball, result.motion


def sample_frames(path: Path, interval_s: float,
                  max_samples: int = MAX_SAMPLES) -> Iterator[tuple[int, float, object]]:
    """``(index, t_s, frame_bgr)`` every ``interval_s`` seconds of a video, skipping frames
    that cannot be decoded. A long clip is sampled coarser rather than cut short, as
    `clip_analysis.analyse_clip` does; one of unknown length stops at ``max_samples``."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open {Path(path).name} as video")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration_s = frames / fps if fps > 0 and frames > 0 else 0.0
        if duration_s > 0 and duration_s / interval_s > max_samples:
            interval_s = duration_s / max_samples
        index, t = 0, 0.0
        while (duration_s <= 0 or t < duration_s) and index < max_samples:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                if duration_s <= 0:
                    break
            else:
                yield index, t, frame
                index += 1
            t += interval_s
    finally:
        capture.release()


def walk_records(path: Path, *, interval_s: float, explain_n: int,
                 redact: bool, camera: str = "", polygon: str = "",
                 whole_frame: bool = False) -> Iterator[str]:
    """One NDJSON line per step. A generator, because streaming it is the whole point.

    ``camera`` names a saved pitch boundary, applied to every sampled frame. A
    fixed camera's pitch does not move between the first frame and the last, so an outline
    drawn once on frame 0 is correct for the whole recording - which is why the editor takes
    a video and draws on its first frame rather than asking for a separate still.
    """
    from pitch_occupancy.config import settings
    from pitch_occupancy.pipeline import resolve_model_key
    from pitch_occupancy.vision import roi
    from pitch_occupancy.vision.overlay import detector_pane

    # Reported rather than assumed: asking for a boundary that does not exist and silently
    # analysing the whole frame gives back exactly the answers the boundary was meant to
    # prevent. `resolve` knows production's camera ids.
    drawn = _parse_polygon(polygon)
    derived = False
    # *Use the whole frame* arrives as its own parameter, because an absent outline is also
    # what never having drawn one looks like, and that case gets a derived outline.
    polygon = None if whole_frame else (drawn or (roi.resolve(camera) if camera else None))
    if polygon is None and drawn is None and not whole_frame:
        # The same fallback `/clip/analyse` has: an uploaded clip names no camera, and
        # without an outline the detector counts the neighbouring pitch and the car park as
        # if they were this one. Derived from the footage, which is what the page shows in
        # its drawing step - so the two tabs agree even when nobody drew anything.
        polygon = _derive_boundary(path)
        derived = polygon is not None
    classify = _classifier()
    detector = resolve_model_key(settings.default_model_key or None)
    yield json.dumps({
        "type": "meta", "detector": detector, "redacted": redact,
        "camera": camera or None, "boundary": bool(polygon),
        #: Which of the three it was. "none" and "one I measured myself" are different
        #: claims, and the page and its tests should not have to infer which happened.
        "boundary_source": "whole" if whole_frame else (
            ("drawn" if drawn else "stored" if camera and not derived
             else "derived" if derived else "stored") if polygon else "none"),
        "coverage": roi.coverage(polygon) if polygon else 1.0,
    }) + "\n"

    seen = 0
    previous = None
    for index, t_s, frame in sample_frames(path, interval_s):
        began = time.perf_counter()
        predicted, confidence, people, ball, motion = _verdict(
            classify(frame, polygon=polygon, previous=previous))
        previous = frame
        explained = explain_n == EXPLAIN_ALL or index < explain_n
        record = {
            "type": "step", "index": index, "t_s": t_s,
            "clock": _clock(t_s), "predicted": predicted,
            "confidence": confidence, "explained": explained,
            "n_inside": people, "ball": ball,
            "motion": None if motion is None else round(motion, 3),
        }
        if explained:
            # A box round each person and a ring round the ball, the boundary drawn, and
            # anything found outside it dimmed rather than dropped.
            boxes, found = detector_pane(frame, polygon=polygon, model_key=detector,
                                         redact=redact)
            record["boxes"] = _jpeg(boxes)
            record["detector"] = found
        record["elapsed_ms"] = round((time.perf_counter() - began) * 1000, 1)
        seen += 1
        yield json.dumps(record) + "\n"

    yield json.dumps({"type": "done", "n": seen}) + "\n"


@router.post("/clip/walkthrough", include_in_schema=False)
async def walkthrough(
    request: Request,
    interval_s: float = Query(DEFAULT_INTERVAL_S, gt=0.5, le=600),
    explain_n: int = Query(-1, ge=-1, le=360),
    redact: bool = Query(False),
    camera: str = Query("", max_length=120),
    #: The outline drawn on the page, as a JSON array of normalised ``[x, y]`` pairs - the
    #: same parameter `/clip/analyse` takes, because the two tabs must analyse the clip the
    #: same way. Watching the model work over the whole frame while Analyse used a boundary
    #: is the same tabs-disagree defect, one tab further along.
    polygon: str = Query("", max_length=4000),
    #: *Use the whole frame*, as an explicit choice rather than an absent outline - the same
    #: parameter `/clip/analyse` takes, because the two tabs must agree about this clip.
    whole_frame: bool = Query(False),
) -> StreamingResponse:
    """Stream the analysis, with the detector's drawing for each explained frame.

    Same deletion guarantee as `/clip/analyse`, and harder to arrange: a StreamingResponse's
    body runs *after* this function returns, so unlinking here would delete the upload before
    a single frame had been read. The removal happens in the generator's own ``finally``
    instead, which is the point at which the file has genuinely stopped being needed - and,
    since a client that disconnects leaves that generator suspended until the garbage
    collector finds it, `CleanupStreamingResponse` closes it and deletes the file itself
    however the response ends.
    """
    # Before the upload: a malformed outline is a 422 now, not an error record after 600 MB
    # have arrived. `walk_records` parses it again, which costs nothing.
    _parse_polygon(polygon)
    fd, path = limits.upload_file()
    accepted = False
    try:
        size = 0
        with os.fdopen(fd, "wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "clip exceeds the upload limit")
                handle.write(chunk)
        if size == 0:
            raise HTTPException(422, "no video in the request body")
        await limits.run_heavy(limits.check_video_pixels, path, "clip")
        accepted = True
    finally:
        if not accepted:
            path.unlink(missing_ok=True)

    def stream() -> Iterator[str]:
        try:
            yield from walk_records(path, interval_s=interval_s, explain_n=explain_n,
                                    redact=redact, camera=camera, polygon=polygon,
                                    whole_frame=whole_frame)
        except Exception as exc:  # noqa: BLE001 - a reporting boundary, not a swallow
            # Broadened from `ValueError` for the reason written out in
            # `image_walkthrough.stream`: the response is already committed as 200 by the
            # time this generator runs, so an unexpected exception would escape as an
            # unhandled ASGI error and the client would get a truncated body with no `error`
            # record. The model load is the likeliest failure here.
            LOG.exception("clip walkthrough failed")
            yield json.dumps({
                "type": "error",
                "detail": f"{type(exc).__name__}: {exc}",
                "fatal": isinstance(exc, ImportError),
            }) + "\n"
        finally:
            path.unlink(missing_ok=True)

    return limits.CleanupStreamingResponse(
        stream(), cleanup=lambda: path.unlink(missing_ok=True))
