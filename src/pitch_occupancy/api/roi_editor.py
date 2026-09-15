"""Draw a pitch boundary, and see what it does to the prediction (WP3-T1, WP6-T6).

The storage and masking live in `vision/roi.py`; this is the surface that lets a person
create one and, more importantly, **check it before trusting it**.

The checking is the part worth building carefully. A boundary is a claim about what the
camera should be allowed to see, and the failure it exists to prevent - a match on the
neighbouring pitch scoring as play on this one - is invisible in the verdict: the model is
right about what is in frame and wrong about what was asked. So `/roi/preview` runs the
classifier **twice**, on the frame as it is and on the frame with the outline applied, and
returns both predictions with both evidence maps. If the boundary is doing its job the two
disagree on exactly the frames the neighbour was in, and the evidence map moves off the edge
of the frame and onto the pitch. If they never disagree, the boundary is either unnecessary
or drawn wrong, and the operator can see which before it goes anywhere near a night's data.

The masking is not free of consequences and the preview is also how that gets noticed: a
large black region is a thing no pretraining set contains, so masking can move a prediction
on a frame that had no neighbour in it at all. That is why the fill mode is offered here
rather than fixed - see the `roi.py` docstring - and why the response reports the prediction
on the unmasked frame rather than only the masked one.
"""

from __future__ import annotations

import base64
import binascii
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pitch_occupancy.api.clip_review import MAX_UPLOAD_BYTES, _classifier
from pitch_occupancy.api.clip_walkthrough import _jpeg
from pitch_occupancy.vision import roi

__all__ = ["router"]

router = APIRouter(prefix="/api/v1", tags=["pitch boundary"])

#: A camera key is a label a person chooses ("venue_01/camera_A"). Bounded so the store
#: cannot be filled with a megabyte of key, and pattern-free on purpose: the deployed camera
#: ids, the dataset's venue folders and an operator's own naming all have to fit.
MAX_KEY = 120


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
    polygon: list[list[float]] = Field(min_length=3)


class PreviewIn(BaseModel):
    image: str
    polygon: list[list[float]] | None = None
    fill: str = roi.DEFAULT_FILL
    #: Skips the two classifier passes. The outline and the coverage figure are useful while
    #: still clicking, and running the backbone on every click would make the editor unusable.
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
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "that file could not be read as an image")
    return frame


def _verdict(frame, *, explain: bool) -> dict:
    """Classify one frame, with its evidence map when asked for.

    Returns the same vocabulary the walkthrough pages use, so a reader moving between
    `/images` and `/roi` is looking at the same quantities under the same names.
    """
    from pitch_occupancy.vision.explain import overlay_heatmap
    from pitch_occupancy.vision.walkthrough import explain_frame

    classifier = _classifier()
    if not explain:
        state, confidence = classifier(frame)
        return {"predicted": str(state), "confidence": confidence}

    shot = explain_frame(frame, classifier)
    area = shot["people_area"]
    return {
        "predicted": shot["predicted"],
        "confidence": shot["confidence"],
        "n_people": shot["n_people"],
        "score_from_map": shot["score_from_map"],
        "score_direct": shot["score_direct"],
        "focus_ratio": (shot["evidence_on_people"] or 0.0) / area if area else None,
        "heat": _jpeg(overlay_heatmap(shot["frame_bgr"], shot["evidence"])),
    }


@router.get("/roi")
def list_boundaries() -> dict:
    """Every saved boundary, with the share of the frame each one keeps."""
    saved = roi.load_all()
    return {
        "cameras": sorted(saved),
        "boundaries": {
            key: {"polygon": polygon, "coverage": roi.coverage(polygon)}
            for key, polygon in saved.items()
        },
        "fills": list(roi.FILLS),
        "default_fill": roi.DEFAULT_FILL,
        "store": str(roi.STORE),
    }


@router.put("/roi")
def save_boundary(body: BoundaryIn) -> dict:
    """Store one camera's boundary. Validation refuses a mis-click rather than saving it."""
    try:
        polygon = roi.save(body.camera, body.polygon)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"camera": body.camera, "polygon": polygon, "coverage": roi.coverage(polygon)}


@router.delete("/roi")
def delete_boundary(camera: str = Query(min_length=1, max_length=MAX_KEY)) -> dict:
    """Forget one camera's boundary. Deleting a camera that has none is not an error.

    A query parameter for the same reason the save takes a body: the key can contain a
    slash, and a slash cannot survive a path segment.
    """
    return {"camera": camera, "removed": roi.remove(camera)}


@router.post("/roi/preview")
def preview(body: PreviewIn) -> dict:
    """Apply a boundary to one frame and report what it changed.

    The two verdicts are the point. `inside` is the frame with everything outside the
    outline filled; `whole` is the frame untouched. A boundary that never changes the
    verdict is not necessarily wrong - it may simply be a frame with no neighbour in it -
    but a boundary that changes it on *every* frame is masking something the model needed.
    """
    if body.fill not in roi.FILLS:
        raise HTTPException(422, f"unknown fill {body.fill!r}; known: {', '.join(roi.FILLS)}")

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
        out["inside"] = _verdict(masked, explain=True)
        # The unmasked verdict is reported even though the boundary is the point: masking is
        # itself a distribution shift, and a reader who only sees the masked answer cannot
        # tell a neighbour being excluded from the mask having moved the prediction on its own.
        out["whole"] = _verdict(frame, explain=True)
        out["changed"] = out["inside"]["predicted"] != out["whole"]["predicted"]
    return out


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
    and the outline is therefore in the coordinate system it will be applied in. Same
    reasoning as `preprocess.py` having one entry point: the failure this project keeps
    paying for is two paths that were supposed to agree.

    The upload is deleted in a `finally`, as on every other route that takes footage.
    """
    fd, name = tempfile.mkstemp(suffix=".upload")
    path = Path(name)
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

        import cv2

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
            return {"frame": _jpeg(frame, width=width), "width": width, "height": height}
        finally:
            capture.release()
    finally:
        path.unlink(missing_ok=True)


@router.get("/roi/page", response_class=HTMLResponse, include_in_schema=False)
def page() -> HTMLResponse:
    from pitch_occupancy.api.roi_page import ROI_HTML

    return HTMLResponse(ROI_HTML)
