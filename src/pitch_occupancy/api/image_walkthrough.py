"""Stream the analysis of one or more still images, showing where the evidence was (WP6-T6).

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
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pitch_occupancy.api.clip_review import _classifier, _gates
from pitch_occupancy.api.clip_walkthrough import _jpeg

__all__ = ["router", "MAX_IMAGES", "MAX_IMAGE_BYTES", "MAX_TOTAL_BYTES"]

router = APIRouter(prefix="/api/v1", tags=["image review"])

#: The traceback goes here rather than to the browser: a client gets the exception
#: type and message, which is enough to act on, and the operator keeps the stack.
LOG = logging.getLogger(__name__)

#: A batch this size takes a couple of minutes to explain, which is about as long as anyone
#: watches. Past that the page stops being a walkthrough and becomes a job.
MAX_IMAGES: int = 32

#: 25 MB per image, 200 MB per batch. A 4K JPEG off a phone is about 5 MB; the per-image cap
#: is what stops one pathological file from consuming the whole batch allowance.
MAX_IMAGE_BYTES: int = 25 * 1024 * 1024
MAX_TOTAL_BYTES: int = 200 * 1024 * 1024


class ImageIn(BaseModel):
    name: str = Field(default="image", max_length=260)
    #: Base64, with or without a `data:` prefix - the browser's `FileReader.readAsDataURL`
    #: produces the prefixed form and stripping it here saves every caller doing it.
    data: str


class BatchIn(BaseModel):
    images: list[ImageIn] = Field(min_length=1)


def _decode(item: ImageIn) -> bytes:
    payload = item.data
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, f"{item.name}: not valid base64") from exc


def walk_records(paths: list[Path], *, names: list[str], explain_n: int,
                 redact: bool, camera: str = "") -> Iterator[str]:
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
    # ids; `get` did not (A36).
    polygon = roi.resolve(camera) if camera else None
    classifier = _classifier()
    # The person gate, which is the one deployed overrule a still can carry: the motion gate
    # compares a frame to the previous sample of the same camera and a folder of stills has
    # no such thing. That absence is reported in the meta record rather than left to be
    # inferred, because on this project's footage the motion gate is what catches most empty
    # frames - so a still is judged with the weaker half of the deployed path.
    _, person_gate = _gates()
    cfg = RuleConfig.load(settings.rules_path)
    yield json.dumps({
        "type": "meta", "backbone": getattr(classifier, "backbone", "?"),
        "n_train": getattr(classifier, "n_train", 0), "redacted": redact,
        "n_submitted": len(paths), "camera": camera or None,
        "boundary": bool(polygon),
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
                            polygon=polygon, person_gate=person_gate):
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
            # people and a ball" are only distinguishable side by side (A35, WP9-T4a).
            boxes, found = detector_pane(shot.frame_bgr, polygon=shot.polygon,
                                         redact=redact)
            record["boxes"] = _jpeg(boxes)
            record["detector"] = found
        yield json.dumps(record) + "\n"

    # `n` and `n_submitted` are both reported because they can differ: a file OpenCV cannot
    # decode yields nothing rather than a guess, and a page that showed only the count it
    # managed would silently drop the difference.
    yield json.dumps({"type": "done", "n": seen, "n_unreadable": len(paths) - seen,
                      "n_gated": gated}) + "\n"


@router.post("/images/walkthrough", include_in_schema=False)
async def walkthrough(
    batch: BatchIn,
    explain_n: int = Query(-1, ge=-1, le=MAX_IMAGES),
    redact: bool = Query(False),
    camera: str = Query("", max_length=120),
) -> StreamingResponse:
    """Stream the analysis of a batch of stills, with an evidence map for each.

    Decoded to a temporary directory rather than held as arrays, so `walk_images` reads them
    the same way every other caller in this project reads a frame - through `cv2.imread`,
    which is also what decides whether a file is an image at all. The directory is removed in
    the generator's ``finally``, which - as in `clip_walkthrough` - is the point at which the
    files have genuinely stopped being needed, because a StreamingResponse's body runs after
    this function has returned.
    """
    if len(batch.images) > MAX_IMAGES:
        raise HTTPException(
            413, f"{len(batch.images)} images; the walkthrough takes at most {MAX_IMAGES}"
        )

    blobs: list[tuple[str, bytes]] = []
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
        blobs.append((item.name, raw))

    workspace = Path(tempfile.mkdtemp(prefix="shots-"))
    paths = []
    for i, (name, raw) in enumerate(blobs):
        # Indexed, and the original name kept only as a suffix: two uploads can share a
        # name, and a caller can send any string at all. `Path(name).name` alone would still
        # collide; this cannot.
        target = workspace / f"{i:03d}-{Path(name).name}"
        target.write_bytes(raw)
        paths.append(target)

    def stream() -> Iterator[str]:
        try:
            yield from walk_records(paths, names=[n for n, _ in blobs],
                                    explain_n=explain_n, redact=redact, camera=camera)
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
            for path in paths:
                path.unlink(missing_ok=True)
            workspace.rmdir()

    return StreamingResponse(stream(), media_type="application/x-ndjson")
