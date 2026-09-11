"""Upload a clip, get a timeline: the review tool for footage with no slot behind it.

The dashboard answers "was this booked hour used?" from the database. This answers "what is
happening in this video?" for a file somebody has in their hand - a disputed slot exported
after the fact, a camera being checked before it goes into the schedule, a clip an operator
wants a second opinion on. There is no schedule, no booking and no slot id involved.

Three things about it are deliberate.

**The upload is never kept.** It is streamed to a temporary file, read, and deleted in a
``finally`` - so a crash mid-analysis does not leave footage of identifiable people on the
server. Nothing is written to the database either; this route has no side effects at all,
which is what lets it sit beside `/schedule/validate` in the mutating-route allowlist rather
than beside the override.

**The body is the video.** Raw bytes, not a multipart form, so this needs no extra
dependency and the browser can post a `File` object directly. The length is capped while
streaming rather than after, because a cap you enforce once the file has arrived has not
capped anything.

**Corrections are reported, not hidden.** `clip_analysis` flags every sample its neighbours
overruled; this route passes all of them through and the page renders them struck through
beside the raw prediction. On the ten-minute clip this was built against, smoothing correctly
absorbed a one-sample flicker and *also* erased a genuine 15-second maintenance event that
the labels confirm was real. Both happen, the reviewer is the one who can tell them apart,
and they can only do that if they are shown what changed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pitch_occupancy.clip_analysis import (
    DEFAULT_INTERVAL_S,
    DEFAULT_WINDOW,
    MAX_SAMPLES,
    analyse_clip,
)

__all__ = ["router", "MAX_UPLOAD_BYTES"]

router = APIRouter(prefix="/api/v1", tags=["clip review"])

#: 600 MB. A ten-minute 960x540 clip is about 140 MB, so this leaves room for a full slot
#: without letting an unbounded body fill the disk. Enforced while streaming.
MAX_UPLOAD_BYTES: int = 600 * 1024 * 1024

#: Loaded once and reused. The first request pays ~13 s to build the probe from the feature
#: cache; every later one pays nothing. Module-level rather than a FastAPI dependency because
#: it is genuinely process-wide state - there is one model, not one per request.
_CLASSIFIER = None


def _classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        from pitch_occupancy.vision.classifier import load_classifier

        _CLASSIFIER = load_classifier()
    return _CLASSIFIER


class SampleOut(BaseModel):
    index: int
    t_s: float
    clock: str
    raw: str
    smoothed: str
    confidence: float
    corrected: bool


class SegmentOut(BaseModel):
    state: str
    label: str
    starts_after: float
    starts_by: float
    ends_by: float
    duration_s: float
    n_samples: int
    n_corrected: int
    describe: str


class ClipOut(BaseModel):
    duration_s: float
    interval_s: float
    window: int
    n_samples: int
    n_corrected: int
    n_unreadable: int
    interval_widened: bool
    summary: str
    dominant: str | None
    samples: list[SampleOut]
    segments: list[SegmentOut] = Field(default_factory=list)


def _clock(t_s: float) -> str:
    total = int(round(t_s))
    return f"{total // 60}:{total % 60:02d}"


@router.post("/clip/analyse", response_model=ClipOut)
async def analyse(
    request: Request,
    interval_s: float = Query(DEFAULT_INTERVAL_S, gt=0.5, le=600),
    window: int = Query(DEFAULT_WINDOW, ge=1, le=31),
) -> ClipOut:
    """Sample the posted video every ``interval_s`` seconds and return its timeline.

    The video is deleted before this returns, whatever happens. ``window`` must be odd: an
    even window has no centre, so "the neighbours overruled it" stops being symmetric.
    """
    if window % 2 == 0:
        raise HTTPException(422, "window must be odd, so each sample has equal neighbours")

    # Delete on the way out rather than relying on the OS: this is footage of people.
    fd, name = tempfile.mkstemp(suffix=".upload")
    path = Path(name)
    try:
        size = 0
        with os.fdopen(fd, "wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"clip exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Trim it, or "
                        f"run the worker over the full slot instead",
                    )
                handle.write(chunk)
        if size == 0:
            raise HTTPException(422, "no video in the request body")

        try:
            result = analyse_clip(
                path, _classifier(), interval_s=interval_s, window=window,
                max_samples=MAX_SAMPLES,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)

    return ClipOut(
        duration_s=result.duration_s,
        interval_s=result.interval_s,
        window=result.window,
        n_samples=len(result.samples),
        n_corrected=result.n_corrected,
        n_unreadable=len(result.unreadable),
        interval_widened=result.interval_widened,
        summary=result.summary(),
        dominant=result.dominant.value if result.dominant else None,
        samples=[
            SampleOut(
                index=s.index, t_s=s.t_s, clock=_clock(s.t_s), raw=s.raw.value,
                smoothed=s.smoothed.value, confidence=s.confidence, corrected=s.corrected,
            )
            for s in result.samples
        ],
        segments=[
            SegmentOut(
                state=g.state.value,
                label=g.state.name.lower().replace("_", " "),
                starts_after=g.starts_after, starts_by=g.starts_by, ends_by=g.ends_by,
                duration_s=g.duration_s, n_samples=g.n_samples, n_corrected=g.n_corrected,
                describe=g.describe(),
            )
            for g in result.segments
        ],
    )


@router.get("/clip/page", response_class=HTMLResponse, include_in_schema=False)
def page() -> HTMLResponse:
    from pitch_occupancy.api.clip_page import CLIP_HTML

    return HTMLResponse(CLIP_HTML)
