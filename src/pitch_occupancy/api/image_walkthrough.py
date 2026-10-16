"""Stream the analysis of one or more still images, showing where the evidence was.

The still-image counterpart to `clip_walkthrough.py`, and built to the same shape: one
newline-delimited JSON record per image, flushed before the next is computed, so the page
renders image *k* while the backbone is still on *k+1*. What a reader watches is the model
working rather than an animation over an answer that was already complete.

**Why a separate page rather than a single-frame clip.** A clip carries time, and time is
what `clip_analysis` uses to correct itself: an isolated disagreeing sample can be overruled
by the samples either side of it. A set of stills has no neighbours - two uploads may be
minutes or venues apart, and nothing in the request says which. So every prediction here
stands on its own evidence, and that is the honest thing for the page to show: the evidence
map stops being an illustration of a verdict and becomes the only basis for it.

**JSON with base64, not a multipart form.** `clip_review.py` takes raw bytes for one video
precisely to avoid the `python-multipart` dependency, and a batch of images needs names and
boundaries that raw bytes cannot carry. Base64 in a JSON body keeps the same no-new-
dependency rule at a 33% wire cost, on a request that is already local and already bounded.
The alternative - one request per image - would have worked too, and was rejected because it
moves the batch loop into the browser and leaves the server unable to report how many files
it actually managed to read.

**Nothing is kept.** Images are decoded in memory and never written to disk, so the deletion
dance `clip_review.py` performs around a temporary file has no counterpart here - there is
nothing to delete. That is a stronger guarantee than the clip route's, not a weaker one.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pitch_occupancy.api import clip_review, limits
from pitch_occupancy.api.clip_review import _gates
from pitch_occupancy.api.clip_walkthrough import _jpeg
from pitch_occupancy.api.limits import parse_model, read_json
from pitch_occupancy.vision import roi as _roi
from pitch_occupancy.vision.people import PersonGate

__all__ = ["router", "MAX_IMAGES", "MAX_IMAGE_BYTES", "MAX_TOTAL_BYTES"]

router = APIRouter(prefix="/api/v1", tags=["image review"])

#: The traceback goes here rather than to the browser: a client gets the exception
#: type and message, which is enough to act on, and the operator keeps the stack.
LOG = logging.getLogger(__name__)


def _classifier():
    """`clip_review._classifier`, looked up when called - see `clip_walkthrough._classifier`
    for why a name copied at import was the wrong seam."""
    return clip_review._classifier()

#: A batch this size takes a couple of minutes to explain, which is about as long as anyone
#: watches. Past that the page stops being a walkthrough and becomes a job.
MAX_IMAGES: int = 32

#: 25 MB per image, 200 MB per batch. A 4K JPEG off a phone is about 5 MB; the per-image cap
#: is what stops one pathological file from consuming the whole batch allowance.
MAX_IMAGE_BYTES: int = 25 * 1024 * 1024
MAX_TOTAL_BYTES: int = 200 * 1024 * 1024

#: The whole JSON body: the batch allowance as base64 (four bytes per three), plus room for
#: names and outlines. Enforced while the body arrives, before any of it is parsed.
MAX_BODY_BYTES: int = MAX_TOTAL_BYTES * 4 // 3 + 4 * 1024 * 1024


class ImageIn(BaseModel):
    name: str = Field(default="image", max_length=260)
    #: This image's own outline, normalised ``[x, y]`` pairs, or absent to use the batch's.
    #: A folder of stills is routinely a folder of different cameras and the page cannot know
    #: otherwise, so one outline across all of them is right only when they share a view.
    polygon: list[list[float]] | None = Field(default=None, max_length=_roi.MAX_POINTS)
    #: This image is to be scored over the whole frame, because somebody chose that. Distinct
    #: from `polygon` being absent, which means "not decided" and is answered by the batch
    #: outline or the stored camera - the page used to send the two the same way, so pressing
    #: *Use the whole frame* was overridden by whatever the boundary step had suggested.
    whole_frame: bool = False
    #: Base64, with or without a `data:` prefix - the browser's `FileReader.readAsDataURL`
    #: produces the prefixed form and stripping it here saves every caller doing it.
    data: str


class BatchIn(BaseModel):
    images: list[ImageIn] = Field(min_length=1)
    #: An outline drawn on the page, in normalised ``[x, y]`` pairs, used in preference to
    #: whatever `camera` resolves to.
    #:
    #: Until now a boundary could only come from the store, so an uploaded still of a camera
    #: the store has never seen was scored over the whole frame - the neighbouring pitch, the
    #: walkway and the car park counted as this pitch. That measured 0.74 false-play against
    #: 0.38 with an outline, and it is the same defect the clip route once carried. The page
    #: now asks for the outline before it predicts, so this is how it arrives.
    polygon: list[list[float]] | None = Field(default=None, max_length=_roi.MAX_POINTS)


def _decode(item: ImageIn) -> bytes:
    payload = item.data
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, f"{item.name}: not valid base64") from exc


def walk_records(paths: list[Path], *, names: list[str], explain_n: int,
                 redact: bool, camera: str = "",
                 polygon: list[list[float]] | None = None,
                 polygons: list[list[list[float]] | None] | None = None,
                 wholes: list[bool] | None = None) -> Iterator[str]:
    """One NDJSON line per image. A generator, because streaming it is the whole point.

    ``names`` carries what the user called each file. The paths on disk are prefixed with an
    index to stop two uploads of the same name from colliding, and reporting that prefixed
    name back would show the reader a filename they never chose.
    """
    from pitch_occupancy.config import settings
    from pitch_occupancy.vision import roi
    from pitch_occupancy.vision.explain import overlay_heatmap
    from pitch_occupancy.vision.overlay import detector_pane
    from pitch_occupancy.vision.rules import RuleConfig, clause_status, rule_table
    from pitch_occupancy.vision.walkthrough import walk_images

    # Whether a boundary was actually found is reported rather than assumed. Asking for one
    # that does not exist and silently analysing the whole frame would produce precisely the
    # answers the operator applied a boundary to avoid - the neighbouring pitch counted as
    # this one - and nothing on the page would say so. `resolve` knows production's camera
    # ids; `get` does not.
    # A drawn outline wins over a stored one: the operator is looking at this frame and the
    # store is not. `validate` rather than trust - a polygon arrives over the wire and a
    # malformed one should be refused here, not produce a mask nobody can explain.
    # Validated here for the same reason the batch outline is: a polygon arrives over the
    # wire, and one that could not be stored must not be analysed with either.
    if polygons:
        checked: list[list[list[float]] | None] = []
        for i, one in enumerate(polygons):
            if not one:
                checked.append(None)
                continue
            try:
                checked.append(roi.validate(one))
            except ValueError as exc:
                raise HTTPException(
                    422, f"image {i + 1}: the outline is not usable: {exc}") from exc
        polygons = checked
    # An image that opted out of a boundary carries no outline, whatever was drawn for it.
    if wholes:
        polygons = [None if (i < len(wholes) and wholes[i]) else one
                    for i, one in enumerate(polygons or [None] * len(paths))]
    drawn = bool(polygon) or any(polygons or ())
    if polygon:
        try:
            polygon = roi.validate(polygon)
        except ValueError as exc:
            raise HTTPException(422, f"the outline is not usable: {exc}") from exc
    else:
        polygon = roi.resolve(camera) if camera else None
    # A batch every one of whose images opted out gets no fallback outline either, or the
    # stored camera would quietly reinstate the boundary image by image.
    if wholes and all(wholes[i] if i < len(wholes) else False for i in range(len(paths))):
        polygon = None
    classifier = _classifier()
    # The person gate, which is the one deployed overrule a still can carry: the motion gate
    # compares a frame to the previous sample of the same camera and a folder of stills has
    # no such thing. That absence is reported in the meta record rather than left to be
    # inferred, because on this project's footage the motion gate is what catches most empty
    # frames - so a still is judged with the weaker half of the deployed path.
    _, person_gate = _gates()
    # Stills only: people found on the pitch mean it is not empty. See
    # `PersonGate.people_overrule_empty` for why this is not on for clips or the worker.
    if isinstance(person_gate, PersonGate):
        person_gate = replace(person_gate, people_overrule_empty=True)
    cfg = RuleConfig.load(settings.rules_path)
    yield json.dumps({
        "type": "meta", "backbone": getattr(classifier, "backbone", "?"),
        "n_train": getattr(classifier, "n_train", 0), "redacted": redact,
        "n_submitted": len(paths), "camera": camera or None,
        "boundary": bool(polygon) or any(polygons or ()),
        "boundary_source": ("drawn" if drawn else "stored")
                           if (polygon or any(polygons or ())) else "none",
        #: How many of the batch carry their own outline, so the page can say "3 of 8 drawn"
        #: rather than implying one boundary covered everything.
        "n_outlines": sum(1 for one in (polygons or ()) if one),
        #: How many were scored over the whole frame on purpose. "none" and "chosen" read
        #: the same on a page that only counts outlines, and they are not the same claim.
        "n_whole_frame": sum(1 for one in (wholes or ()) if one),
        "coverage": roi.coverage(polygon) if polygon else 1.0,
        "gated": person_gate is not None, "motion_gate": False,
        "detector": cfg.detector,
        # The decision table with this deployment's numbers in it, sent once rather than
        # typed into the page. `require_boundary=False` because that is what `detector_pane`
        # passes - the whole frame is counted here, so row 2 cannot fire and the page must
        # not draw it as though it could. `motion_available=False` says the same about
        # movement: a still has no predecessor, so rows and clauses that rest on the motion
        # cue are marked unreachable rather than shown as passed.
        "rules": rule_table(cfg, motion_available=False, require_boundary=False),
        "clauses": clause_status(cfg, motion_available=False),
    }) + "\n"

    seen = 0
    gated = 0
    for shot in walk_images(paths, classifier, explain_n=explain_n, redact=redact,
                            polygon=polygon, polygons=polygons, wholes=wholes,
                            person_gate=person_gate):
        seen += 1
        gated += shot.probed is not None
        record = {
            "type": "shot", "index": shot.index,
            "name": names[shot.index] if shot.index < len(names) else shot.name,
            "predicted": shot.predicted, "confidence": shot.confidence,
            "elapsed_ms": round(shot.elapsed_ms, 1), "explained": shot.explained,
            "width": shot.width, "height": shot.height,
            "probed": shot.probed, "n_inside": shot.n_inside, "ball": shot.ball,
        }
        if shot.explained:
            record.update({
                "frame": _jpeg(shot.frame_bgr),
                "heat": _jpeg(overlay_heatmap(shot.frame_bgr, shot.evidence,
                                             polygon=shot.polygon)),
                "grid": list(shot.grid or ()),
                "score_from_map": shot.score_from_map,
                "score_direct": shot.score_direct,
                "reconstruction_error": shot.reconstruction_error,
                "n_people": shot.n_people,
                "evidence_on_people": shot.evidence_on_people,
                "people_area": shot.people_area,
                "focus_ratio": shot.focus_ratio,
                # Zero whenever a boundary is in force, and on the page for exactly
                # that reason: the doubt this answers is that the evidence map showed
                # the model reading past the outline.
                "evidence_outside": shot.evidence_outside,
            })
            # The other model, on the same frame. The probe explains itself as a heatmap
            # because a logistic fit on pooled features has no objects in it; the detector
            # explains itself as a box round each person and a ring round the ball. Both
            # go on the page, because "it is reading the floodlights" and "it found six
            # people and a ball" are only distinguishable side by side.
            boxes, found = detector_pane(shot.raw_bgr, polygon=shot.polygon,
                                         redact=redact)
            record["boxes"] = _jpeg(boxes)
            record["detector"] = found
        yield json.dumps(record) + "\n"

    # `n` and `n_submitted` are both reported because they can differ: a file OpenCV cannot
    # decode yields nothing rather than a guess, and a page that showed only the count it
    # managed would silently drop the difference.
    yield json.dumps({"type": "done", "n": seen, "n_unreadable": len(paths) - seen,
                      "n_gated": gated}) + "\n"


def _stage(batch: BatchIn) -> tuple[Path, list[Path]]:
    """Decode and check every image, then write them to a new workspace: ``(dir, paths)``.

    Synchronous, for a worker thread - base64 over 200 MB is seconds of CPU. Every check runs
    before anything is written, and the workspace is removed if writing fails part-way, so a
    refused batch leaves nothing on disk. It once did: a 255-character name or a NUL byte
    failed the write after earlier images were already there, and they stayed.
    """
    blobs: list[bytes] = []
    total = 0
    for item in batch.images:
        raw = _decode(item)
        if not raw:
            raise HTTPException(422, f"{item.name}: empty image")
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(
                413, f"{item.name} exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB"
            )
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise HTTPException(
                413, f"batch exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB"
            )
        # From the header, before anything decodes it: the byte caps above bound the file,
        # not the gigabyte a small PNG can declare.
        limits.check_image_pixels(raw, item.name)
        blobs.append(raw)

    workspace = limits.upload_workspace()
    try:
        paths = []
        for i, raw in enumerate(blobs):
            # The index alone, never the caller's name: any string at all can arrive there,
            # and the name only travels back in the records, through `names`. `cv2.imread`
            # decides what a file is from its bytes, so no extension is needed either.
            target = workspace / f"{i:03d}.img"
            target.write_bytes(raw)
            paths.append(target)
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    return workspace, paths


@router.post("/images/walkthrough", include_in_schema=False)
async def walkthrough(
    request: Request,
    explain_n: int = Query(-1, ge=-1, le=MAX_IMAGES),
    redact: bool = Query(False),
    camera: str = Query("", max_length=120),
) -> StreamingResponse:
    """Stream the analysis of a batch of stills, with an evidence map for each.

    Decoded to a temporary directory rather than held as arrays, so `walk_images` reads them
    the same way every other caller in this project reads a frame - through `cv2.imread`,
    which is also what decides whether a file is an image at all. The directory is removed
    when the response ends, however it ends - `CleanupStreamingResponse` closes the generator
    and removes it even when the client disconnects, rather than leaving that to whenever
    the garbage collector finds the suspended generator.

    The body is read by hand, capped at what the image caps allow, rather than declared as a
    `BatchIn` parameter: FastAPI buffers a declared body whole before any cap applies.
    """
    batch = parse_model(BatchIn, await read_json(request, MAX_BODY_BYTES))
    if len(batch.images) > MAX_IMAGES:
        raise HTTPException(
            413, f"{len(batch.images)} images; the walkthrough takes at most {MAX_IMAGES}"
        )
    # The outlines are checked here as well as in `walk_records`, so a malformed one is a 422
    # before anything is decoded or written, not an error record inside a 200 stream.
    for label, one in [("the outline", batch.polygon)] + [
            (f"image {i + 1}: the outline", item.polygon)
            for i, item in enumerate(batch.images)]:
        if one:
            try:
                _roi.validate(one)
            except ValueError as exc:
                raise HTTPException(422, f"{label} is not usable: {exc}") from exc

    workspace, paths = await limits.run_heavy(_stage, batch)
    names = [item.name for item in batch.images]

    def stream() -> Iterator[str]:
        try:
            yield from walk_records(paths, names=names,
                                    explain_n=explain_n, redact=redact, camera=camera,
                                    polygon=batch.polygon,
                                    polygons=[i.polygon for i in batch.images],
                                    wholes=[i.whole_frame for i in batch.images])
        except Exception as exc:  # noqa: BLE001 - see below; this is a reporting boundary
            # **Every** failure has to become a record, not just the expected ones.
            #
            # This caught `ValueError` alone, which is the failure a bad upload produces -
            # and the response has already been committed as 200 by the time the generator
            # runs, so anything else escaped as an unhandled ASGI error *mid-stream*. The
            # client then saw a truncated NDJSON body with no `error` record in it: the page
            # spun forever, and the only account of what went wrong was in the server
            # console. Observed for real when Windows Application Control blocked torch's
            # `_C` DLL and the model failed to load - an `ImportError`, on the very first
            # record, before a single image had been processed.
            #
            # The model load is the most likely thing to fail here and the least likely to
            # raise `ValueError`, so the narrow catch was wrong in exactly the case that
            # matters most.
            LOG.exception("image walkthrough failed")
            yield json.dumps({
                "type": "error",
                "detail": f"{type(exc).__name__}: {exc}",
                # The page cannot usefully retry a missing model, and telling someone to
                # re-upload when the backbone will not load wastes their time.
                "fatal": isinstance(exc, ImportError),
            }) + "\n"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    return limits.CleanupStreamingResponse(
        stream(), cleanup=lambda: shutil.rmtree(workspace, ignore_errors=True))
