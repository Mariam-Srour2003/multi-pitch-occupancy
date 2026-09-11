"""Stream a clip's analysis step by step, showing where the evidence was (WP4-T5, WP6-T6).

`clip_review.py` returns a finished timeline in one response. This returns the same work as
it happens - one newline-delimited JSON record per sampled frame, flushed before the next is
computed - so the page renders step *k* while the backbone is still on *k+1*. What a reader
watches is the model running, not an animation replayed over an answer that was already
complete.

That distinction is the reason this exists. A dashboard that produces a verdict in one jump
asks to be trusted; one that shows the frame, the region the evidence came from, and the
score those regions add up to, can be argued with. `vision/explain.py` has had the machinery
since WP4-T5 and nothing in the operator surface had ever surfaced it - it existed only in
figures generated for the thesis.

**Pacing is the client's, and the page says so.** The slow motion is a pause the browser adds
between renders, and its "continue without slow motion" button drops that pause. The backbone
is working at the same speed either way. The control that changes the actual *work* is
``explain_n``: an explained frame costs roughly twice a bare prediction, so bounding it is
what finishes a long clip sooner. Labelling the button honestly matters more than making it
sound powerful - a reader who thinks they sped the model up has learnt something false about
the system.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from pitch_occupancy.api.clip_review import MAX_UPLOAD_BYTES, _classifier, _clock
from pitch_occupancy.clip_analysis import DEFAULT_INTERVAL_S

__all__ = ["router", "JPEG_WIDTH"]

router = APIRouter(prefix="/api/v1", tags=["clip review"])

#: Frames reach the browser as JPEGs at this width. 640 keeps a sixty-step walkthrough to a
#: few megabytes while leaving the heatmap legible. The explanation is always computed on the
#: frame as decoded - the downscale happens on the way out, never on the way in.
JPEG_WIDTH: int = 640


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


def walk_records(path: Path, *, interval_s: float, explain_n: int,
                 redact: bool) -> Iterator[str]:
    """One NDJSON line per step. A generator, because streaming it is the whole point."""
    from pitch_occupancy.vision.explain import overlay_heatmap
    from pitch_occupancy.vision.walkthrough import walk_clip

    classifier = _classifier()
    yield json.dumps({
        "type": "meta", "backbone": getattr(classifier, "backbone", "?"),
        "n_train": getattr(classifier, "n_train", 0), "redacted": redact,
    }) + "\n"

    seen = 0
    for step in walk_clip(path, classifier, interval_s=interval_s,
                          explain_n=explain_n, redact=redact):
        seen += 1
        record = {
            "type": "step", "index": step.index, "t_s": step.t_s,
            "clock": _clock(step.t_s), "predicted": step.predicted,
            "confidence": step.confidence, "elapsed_ms": round(step.elapsed_ms, 1),
            "explained": step.explained,
        }
        if step.explained:
            record.update({
                "frame": _jpeg(step.frame_bgr),
                "heat": _jpeg(overlay_heatmap(step.frame_bgr, step.evidence)),
                "grid": list(step.grid or ()),
                "score_from_map": step.score_from_map,
                "score_direct": step.score_direct,
                "reconstruction_error": step.reconstruction_error,
                "n_people": step.n_people,
                "evidence_on_people": step.evidence_on_people,
                "people_area": step.people_area,
                "focus_ratio": step.focus_ratio,
            })
        yield json.dumps(record) + "\n"

    yield json.dumps({"type": "done", "n": seen}) + "\n"


@router.post("/clip/walkthrough", include_in_schema=False)
async def walkthrough(
    request: Request,
    interval_s: float = Query(DEFAULT_INTERVAL_S, gt=0.5, le=600),
    explain_n: int = Query(-1, ge=-1, le=360),
    redact: bool = Query(False),
) -> StreamingResponse:
    """Stream the analysis, with an evidence map for each explained frame.

    Same deletion guarantee as `/clip/analyse`, and harder to arrange: a StreamingResponse's
    body runs *after* this function returns, so unlinking here would delete the upload before
    a single frame had been read. The removal happens in the generator's own ``finally``
    instead, which is the point at which the file has genuinely stopped being needed.
    """
    fd, name = tempfile.mkstemp(suffix=".upload")
    path = Path(name)
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
        accepted = True
    finally:
        if not accepted:
            path.unlink(missing_ok=True)

    def stream() -> Iterator[str]:
        try:
            yield from walk_records(path, interval_s=interval_s, explain_n=explain_n,
                                    redact=redact)
        except ValueError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
        finally:
            path.unlink(missing_ok=True)

    return StreamingResponse(stream(), media_type="application/x-ndjson")
