"""Draw a pitch boundary, and see what it does to the prediction.

The storage and masking live in `vision/roi.py`; this is the surface that lets a person
create one and, more importantly, **check it before trusting it**.

The checking is the part worth building carefully. A boundary is a claim about what the
camera should be allowed to see, and the failure it exists to prevent - a match on the
neighbouring pitch counting as play on this one - is invisible in the verdict: the detector
is right about what is in frame and wrong about what was asked. So `/roi/preview` runs the
rule **twice**, counting inside the outline and over the whole frame, and returns both
verdicts with the detector's drawing of each. If the boundary is doing its job the two
disagree on exactly the frames the neighbour was in, and the people outside the outline are
drawn dimmed. If they never disagree, the boundary is either unnecessary or drawn wrong, and
the operator can see which before it goes anywhere near a night's data.
"""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pitch_occupancy.api import limits
from pitch_occupancy.api.clip_review import MAX_UPLOAD_BYTES
from pitch_occupancy.api.clip_walkthrough import _jpeg
from pitch_occupancy.vision import roi

__all__ = ["router"]

router = APIRouter(prefix="/api/v1", tags=["pitch boundary"])

#: A camera key is a label a person chooses ("venue_01/camera_A"). Bounded so the store
#: cannot be filled with a megabyte of key, and pattern-free on purpose: the deployed camera
#: ids, the dataset's venue folders and an operator's own naming all have to fit.
MAX_KEY = 120

#: A boundary is a few hundred bytes; 64 KiB leaves room for any key and `MAX_POINTS` points
#: written out longhand, and refuses a megabyte of them before it is buffered.
MAX_BOUNDARY_BODY = 64 * 1024

#: One 25 MB image as base64 (a third larger) plus the outline - the per-image ceiling
#: `/images/walkthrough` applies.
MAX_PREVIEW_BODY = 34 * 1024 * 1024

#: `roi_derive.MAX_FRAMES` stills at that ceiling would be a gigabyte of base64. A suggestion
#: needs a handful of frames, so the request is bounded well below that.
MAX_SUGGEST_BODY = 200 * 1024 * 1024


class BoundaryIn(BaseModel):
    """The camera travels in the body, not the path, and that is not a style choice.

    The natural key for a camera in this project contains a slash - `venue_01/camera_A` is
    the shape `configs/cameras.json` already uses - and a slash in a path segment cannot be
    escaped past a router: `PUT /api/v1/roi/venue_01%2Fcamera_A` returns 404, because the
    encoded separator is decoded before routing. The editor's own placeholder suggested
    exactly that format, so the first key anyone typed would have failed.

    A key chosen by a person should not be constrained by URL path semantics, so it is not.
    """

    camera: str = Field(min_length=1, max_length=MAX_KEY)
    polygon: list[list[float]] = Field(min_length=3, max_length=roi.MAX_POINTS)


class PreviewIn(BaseModel):
    image: str
    polygon: list[list[float]] | None = Field(default=None, max_length=roi.MAX_POINTS)
    fill: str = roi.DEFAULT_FILL
    #: Skips the two detector passes. The outline and the coverage figure are useful while
    #: still clicking, and running the detector on every click would make the editor unusable.
    explain: bool = True


def _decode(payload: str, what: str = "image") -> bytes:
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, f"{what} is not valid base64") from exc


def _frame(payload: str):
    import cv2
    import numpy as np

    raw = _decode(payload)
    if not raw:
        raise HTTPException(422, "no image in the request")
    # Before the decode, from the header: a 17 KiB PNG declaring 12000x12000 pixels decodes
    # to a gigabyte, and the byte caps never see the difference.
    limits.check_image_pixels(raw)
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "that file could not be read as an image")
    return frame


def _verdict(frame, polygon) -> dict:
    """Count one frame inside ``polygon`` (or over the whole frame) and apply the rule.

    The same detector and rule the walkthrough pages draw (`overlay.detector_pane`), so a
    reader moving between `/images` and `/roi` is looking at the same quantities.
    """
    from pitch_occupancy.config import settings
    from pitch_occupancy.pipeline import resolve_model_key
    from pitch_occupancy.vision.overlay import detector_pane

    drawn, found = detector_pane(
        frame, polygon=polygon, model_key=resolve_model_key(settings.default_model_key or None))
    return {
        "predicted": found["state"],
        "confidence": found["confidence"],
        "n_people": found["people_inside"],
        "n_outside": found["people_outside_boundary"],
        "ball": found["ball"],
        "checked": found["checked"],
        "image": _jpeg(drawn),
    }


@router.get("/roi")
def list_boundaries(limit: int | None = Query(None, ge=0, le=1000)) -> dict:
    """Every saved boundary, with the share of the frame each one keeps.

    ``limit`` shortens the *menu* and deletes nothing. The two stores are read by the live
    path and by every masking experiment, so a page that wants a three-item selector asks for
    three items rather than emptying `configs/roi_derived.json` - which would leave the
    experiments running unmasked and quietly disagreeing with their own logged numbers.

    Hand-drawn outlines come first when the list is trimmed, because `DERIVED_STORE` holds one
    entry per clip in the corpus and would otherwise fill a short menu with machine-generated
    ids while the outlines somebody drew on purpose fell off the end.
    """
    saved = roi.load_all()
    keys = sorted(saved)
    if limit is not None:
        drawn = roi.load_drawn()
        keys = sorted(keys, key=lambda k: (k not in drawn, k))[:limit]
        saved = {k: saved[k] for k in keys}
    return {
        "cameras": keys,
        "boundaries": {
            key: {"polygon": polygon, "coverage": roi.coverage(polygon)}
            for key, polygon in saved.items()
        },
        "fills": list(roi.FILLS),
        "default_fill": roi.DEFAULT_FILL,
        "store": str(roi.STORE),
    }


def _body_schema(model: type[BaseModel]) -> dict:
    """The OpenAPI request body for a route that reads its own body so as to bound it."""
    return {"requestBody": {"required": True, "content": {
        "application/json": {"schema": model.model_json_schema()}}}}


@router.put("/roi", openapi_extra=_body_schema(BoundaryIn))
async def save_boundary(request: Request) -> dict:
    """Store one camera's boundary. Validation refuses a mis-click rather than saving it.

    The body is read by hand, capped, rather than declared as a parameter: FastAPI buffers a
    declared body whole before anything can bound it.
    """
    body = limits.parse_model(BoundaryIn, await limits.read_json(request, MAX_BOUNDARY_BODY))
    try:
        # In a thread: the save waits on a lock and writes a file, and neither belongs on
        # the event loop.
        polygon = await limits.run_heavy(roi.save, body.camera, body.polygon)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"camera": body.camera, "polygon": polygon, "coverage": roi.coverage(polygon)}


@router.delete("/roi")
def delete_boundary(camera: str = Query(min_length=1, max_length=MAX_KEY)) -> dict:
    """Forget one camera's boundary. Deleting a camera that has none is not an error.

    A query parameter for the same reason the save takes a body: the key can contain a
    slash, and a slash cannot survive a path segment.
    """
    try:
        return {"camera": camera, "removed": roi.remove(camera)}
    except ValueError as exc:  # a reserved key, or a store that cannot be parsed
        raise HTTPException(422, str(exc)) from exc


@router.post("/roi/preview", openapi_extra=_body_schema(PreviewIn))
async def preview(request: Request) -> dict:
    """Apply a boundary to one frame and report what it changed.

    The two verdicts are the point. `inside` counts only the people standing inside the
    outline; `whole` counts the whole frame. A boundary that never changes the verdict is not
    necessarily wrong - it may simply be a frame with no neighbour in it - but one that
    changes it on *every* frame is cutting off part of the pitch.
    """
    body = limits.parse_model(PreviewIn, await limits.read_json(request, MAX_PREVIEW_BODY))
    if body.fill not in roi.FILLS:
        raise HTTPException(422, f"unknown fill {body.fill!r}; known: {', '.join(roi.FILLS)}")
    return await limits.run_heavy(_preview, body)


def _preview(body: PreviewIn) -> dict:
    """The work of `preview`, in a worker thread: a decode and two detector passes."""
    frame = _frame(body.image)
    height, width = frame.shape[:2]

    polygon = None
    if body.polygon:
        try:
            polygon = roi.validate(body.polygon)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    masked = roi.apply(frame, polygon, fill=body.fill)
    out = {
        "width": width, "height": height,
        "coverage": roi.coverage(polygon) if polygon else 1.0,
        "masked": _jpeg(masked),
        "fill": body.fill,
    }
    if body.explain:
        out["inside"] = _verdict(frame, polygon)
        # The whole-frame verdict beside it, so the reader sees what the outline changed.
        out["whole"] = _verdict(frame, None)
        out["changed"] = out["inside"]["predicted"] != out["whole"]["predicted"]
    return out


@router.post("/roi/suggest", include_in_schema=False)
async def suggest(request: Request) -> dict:
    """Measure an outline from posted stills, so the drawing step starts from something.

    The images page asks for a boundary *before* it predicts, and an operator facing an empty
    canvas and a photograph of a pitch will reasonably click "whole frame" - which is the
    thing the step exists to stop: the whole frame measured 0.74 false-play against 0.38
    with an outline. So the canvas opens with a measured suggestion on it and the
    operator corrects it, which is a much easier request than drawing one from nothing.

    The same routine `derive_roi.py` runs over the corpus: the per-pixel median of the frames
    removes anything that moved, and the largest connected excess-green region of that median
    is the turf. With several stills of one camera it is a median over them; with one it is
    that frame. The suggestion is a convex hull because the tighter contour clips players
    standing where the mask reads slightly less green, and it is a suggestion and not an
    answer because every boundary error seen so far is a colour error, and a person looking at
    the frame is still the best available check.

    Returns ``polygon: null`` when no turf is found, which is a real outcome on an indoor
    court, and the page keeps the operator drawing rather than reporting a failure.
    """
    import cv2
    import numpy as np

    from pitch_occupancy.vision import roi_derive

    payload = await limits.read_json(request, MAX_SUGGEST_BODY)
    # The shape checks a declared model would have made, made by hand: a list body, a string
    # where the images belong, or an image whose data is a number each raised AttributeError
    # here and came back as a 500.
    items = payload.get("images") if isinstance(payload, dict) else None
    if not items or not isinstance(items, list):
        raise HTTPException(422, "no images in the request")
    datas = [item["data"] for item in items[:roi_derive.MAX_FRAMES]
             if isinstance(item, dict) and isinstance(item.get("data"), str) and item["data"]]
    if not datas:
        raise HTTPException(422, "none of those could be read as an image")

    def work() -> dict:
        frames = []
        for data in datas:
            if data.startswith("data:"):
                _, _, data = data.partition(",")
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                continue
            # Up to `MAX_FRAMES` decoded frames are held at once for the median, so the
            # pixel cap matters forty times over here.
            limits.check_image_pixels(raw)
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) \
                if raw else None
            if frame is not None:
                frames.append(frame)
        if not frames:
            raise HTTPException(422, "none of those could be read as an image")

        polygon = roi_derive.derive_from_frames(frames)
        return {
            "polygon": polygon,
            "coverage": roi.coverage(polygon) if polygon else 1.0,
            "n_frames": len(frames),
        }

    # Decoding forty stills and taking their per-pixel median is seconds of CPU.
    return await limits.run_heavy(work)


@router.post("/roi/first-frame", include_in_schema=False)
async def first_frame(request: Request) -> dict:
    """Return the first readable frame of a posted video, to draw a boundary on.

    **Decoded here rather than in the browser, and that is the point of the endpoint.** A
    `<video>` element could paint frame 0 onto a canvas without any upload, which is faster
    and was the obvious design - but the browser and OpenCV do not have to agree on what
    "the first frame" is. They can differ on rotation metadata, on colour conversion, and on
    which frame a seek to 0 lands on when the stream opens on a B-frame. A boundary drawn
    against a frame the analysis never sees is off by however much they disagree, silently,
    with no error anywhere.

    So the drawing surface comes from the same `cv2.VideoCapture` that `walk_clip` reads,
    and the outline is therefore in the coordinate system it will be applied in. The
    failure this project keeps paying for is two paths that were supposed to agree.

    The upload is deleted in a `finally`, as on every other route that takes footage.
    """
    fd, path = limits.upload_file()
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
        # The decode and the derivation run in a worker thread. The derivation reads frames
        # spread over the whole clip, and on the event loop it held every other request.
        return await limits.run_heavy(_first_frame, path)
    finally:
        path.unlink(missing_ok=True)


def _first_frame(path: Path) -> dict:
    """The work of `first_frame`, in a worker thread. The caller deletes ``path``."""
    import cv2

    limits.check_video_pixels(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise HTTPException(422, "that file could not be opened as a video")
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise HTTPException(422, "that video has no readable first frame")
        height, width = frame.shape[:2]
        # Full width, not `_jpeg`'s 640: the operator is about to place points on this
        # by eye, and a downscaled drawing surface costs them precision they cannot get
        # back. The polygon is normalised, so the size it was drawn at does not matter
        # to anything downstream - only to how accurately it can be placed.
        # A suggested outline comes back with the frame, so the clip page can open its
        # drawing step on something rather than an empty canvas - the same reasoning as
        # `/roi/suggest` for stills. Measured across the *whole* clip rather than from
        # this one frame: the per-pixel median of frames spread over the file removes
        # the players, and frame 0 alone has whoever was standing there in it.
        #
        # A failure here is not a failure of the endpoint. The frame is what the caller
        # asked for; the outline is a convenience, and `null` leaves the operator drawing.
        try:
            from pitch_occupancy.vision import roi_derive

            polygon = roi_derive.derive_from_video(path)
        except Exception:  # noqa: BLE001 - a suggestion is never a precondition
            polygon = None
        return {"frame": _jpeg(frame, width=width), "width": width, "height": height,
                "polygon": polygon,
                "coverage": roi.coverage(polygon) if polygon else 1.0}
    finally:
        capture.release()


@router.get("/roi/page", response_class=HTMLResponse, include_in_schema=False)
def page() -> HTMLResponse:
    from pitch_occupancy.api.roi_page import ROI_HTML

    return HTMLResponse(ROI_HTML)
