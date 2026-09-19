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
from pitch_occupancy.vision import roi

__all__ = ["router", "MAX_UPLOAD_BYTES"]

router = APIRouter(prefix="/api/v1", tags=["clip review"])

#: 600 MB. A ten-minute 960x540 clip is about 140 MB, so this leaves room for a full slot
#: without letting an unbounded body fill the disk. Enforced while streaming.
MAX_UPLOAD_BYTES: int = 600 * 1024 * 1024

def _classifier():
    """The deployed classifier, from the process-wide pipeline (`pipeline.shared`, A36).

    Until 2026-09-19 this loaded the probe itself, which is how the page came to run a
    different system from the worker (A35): each surface assembled its own. Now every surface
    asks the same seam. Kept as a function so a test can swap the classifier for a script
    without assembling a pipeline; the first real call pays the probe fit once.
    """
    from pitch_occupancy.pipeline import shared

    return shared().classify


class SampleOut(BaseModel):
    index: int
    t_s: float
    clock: str
    raw: str
    smoothed: str
    confidence: float
    corrected: bool
    #: What the probe said before a gate overruled it, or None when none did. `raw` is the
    #: verdict after the gates because that is what the system answers; this is why.
    probed: str | None = None
    gated: bool = False
    #: People inside the boundary, and whether a ball was there. None means the detector did
    #: not run - the gate skips a verdict it cannot change - which is a different fact from
    #: zero and is why the page shows a dash.
    people: int | None = None
    ball: bool | None = None


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
    #: How many verdicts a gate weakened. Separate from `n_corrected`, which counts the
    #: neighbour smoothing: they are different mechanisms and a reviewer chasing one should
    #: not be handed the other.
    n_gated: int = 0
    n_unreadable: int
    interval_widened: bool
    summary: str
    dominant: str | None
    samples: list[SampleOut]
    segments: list[SegmentOut] = Field(default_factory=list)
    #: Which boundary was asked for, and whether one was found. Reported rather than
    #: assumed: a camera named with no boundary saved means the whole frame was analysed,
    #: which is the case the boundary exists to avoid.
    camera: str | None = None
    boundary: bool = False
    #: True when the outline was measured from this clip rather than read from the store.
    #: A derived boundary is a fallback and weaker than a hand-drawn one, so the page should
    #: say which it had - "no boundary" and "one I guessed" are different claims.
    boundary_derived: bool = False


def _clock(t_s: float) -> str:
    total = int(round(t_s))
    return f"{total // 60}:{total % 60:02d}"


def _gates():
    """The gates the route applies, as a factory for the same reason `_classifier` is one.

    A test about neighbour smoothing wants to script a classifier and see the smoothing; with
    the gates hard-wired it instead sees the person gate turn every frame of a synthetic video
    EMPTY, because a generated test frame contains no people. Patching this returns the route
    to the probe alone, which is what such a test means by "the model said".
    """
    from pitch_occupancy.pipeline import default_gates

    return default_gates()


def _derive_boundary(path: Path) -> list[list[float]] | None:
    """Measure a pitch outline from the clip itself, or None if it cannot be found.

    `roi.derive_from_video` - the routine `scripts/derive_roi.py` runs over the corpus, now
    in the package (A36) rather than reached through `sys.path`. It is a fallback and not a
    replacement: a hand-drawn or stored outline is better and is preferred above. What it
    replaces is *no boundary at all*, which A19 measured on this exact clip at 0.74
    false-play against 0.38 with one.

    Failure returns None rather than raising. A clip this cannot find turf in is still worth
    classifying without an outline, and the response says which happened.
    """
    return roi.derive_from_video(path)


@router.post("/clip/analyse", response_model=ClipOut)
async def analyse(
    request: Request,
    interval_s: float = Query(DEFAULT_INTERVAL_S, gt=0.5, le=600),
    window: int = Query(DEFAULT_WINDOW, ge=1, le=31),
    camera: str = Query("", max_length=120),
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

        # A boundary is applied by wrapping the classifier rather than by teaching
        # `analyse_clip` about polygons. It takes a `classify` callable precisely so that
        # what a frame means is the caller's business, and a masked frame is still just a
        # frame - so the timeline, the smoothing and the segmentation stay unaware of ROI,
        # and there is one masking call rather than a second code path through them.
        classify = _classifier()
        # `resolve`, not `get`: the camera is named the way production names it, and the
        # store's own keys are not that (A36).
        polygon = roi.resolve(camera) if camera else None
        derived_boundary = False
        if polygon is None:
            # An uploaded clip usually names no camera, or names one the store has never
            # seen, and the boundary was simply skipped - which is how a 234-second clip of
            # an empty floodlit pitch came back 24 of 24 ACTIVE_PLAY at confidence 1.000.
            # Without an outline the probe scores the neighbouring pitch, the walkway and the
            # car park as if they were this pitch. `derive_roi` measures one from the footage
            # itself, which is the same thing `scripts/run_slot_on_video.py` does and the
            # reason its numbers on this clip were right.
            polygon = _derive_boundary(path)
            derived_boundary = polygon is not None
        if polygon:
            inner = classify

            def classify(frame):  # noqa: F811 - deliberately shadows, one frame at a time
                # Both halves, and both are needed. `apply` fills the outside so the pixels
                # are not the neighbouring pitch; `polygon=` drops those positions from the
                # pooling so the fill is not scored either. Masking alone left the model
                # reading a large black region and calling it evidence.
                return inner(roi.apply(frame, polygon), polygon=polygon)

        try:
            # The same two gates `worker.run_slot` applies, for the same reason: the probe
            # alone is not the system, and a review page showing the probe alone is showing
            # something nobody deploys.
            motion_gate, person_gate = _gates()
            result = analyse_clip(
                path, classify, interval_s=interval_s, window=window,
                max_samples=MAX_SAMPLES, polygon=polygon,
                motion_gate=motion_gate, person_gate=person_gate,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)

    return ClipOut(
        camera=camera or None,
        boundary=bool(polygon),
        boundary_derived=derived_boundary,
        duration_s=result.duration_s,
        interval_s=result.interval_s,
        window=result.window,
        n_samples=len(result.samples),
        n_corrected=result.n_corrected,
        n_gated=sum(1 for s in result.samples if s.gated),
        n_unreadable=len(result.unreadable),
        interval_widened=result.interval_widened,
        summary=result.summary(),
        dominant=result.dominant.value if result.dominant else None,
        samples=[
            SampleOut(
                index=s.index, t_s=s.t_s, clock=_clock(s.t_s), raw=s.raw.value,
                probed=s.probed.value if s.probed else None, gated=s.gated,
                people=s.people, ball=s.ball,
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
