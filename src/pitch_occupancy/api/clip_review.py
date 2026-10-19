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

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pitch_occupancy.api import limits
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
    """The deployed classifier for one frame, from the process-wide pipeline.

    Returns ``classify(frame, *, polygon=None, previous=None) -> FrameVerdict`` over
    `pipeline.shared`, so this page runs the same detector and rule as the worker. Kept as a
    function so a test can swap the classifier for a script without loading the detector.
    """
    from pitch_occupancy.pipeline import shared

    pipeline = shared()

    def classify(frame, *, polygon=None, previous=None):
        return pipeline.classify_frame(frame, polygon=polygon, previous=previous)

    return classify


def _play_min() -> int:
    """The rule's ``play_min``, read from `configs/rules.json` without loading a detector."""
    from pitch_occupancy.config import settings
    from pitch_occupancy.vision.rules import RuleConfig

    return RuleConfig.load(settings.rules_path).play_min


class SampleOut(BaseModel):
    index: int
    t_s: float
    clock: str
    raw: str
    smoothed: str
    confidence: float
    corrected: bool
    #: People inside the boundary, and whether a ball was there. None means the detector did
    #: not run, which is a different fact from zero and is why the page shows a dash.
    people: int | None = None
    ball: bool | None = None
    #: The motion cue against the previous sample; None on the first, which has none.
    motion: float | None = None


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
    #: Frames that were read but not decided (the detector did not run).
    n_undecided: int = 0
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
    #: True when a person drew the outline on the page for this clip. The strongest of the
    #: three: it beats the store and the derivation, because the operator is looking at the
    #: footage and neither of the others is.
    boundary_drawn: bool = False
    #: True when the outline was measured from this clip rather than read from the store.
    #: A derived boundary is a fallback and weaker than a hand-drawn one, so the page should
    #: say which it had - "no boundary" and "one I guessed" are different claims.
    boundary_derived: bool = False

    #: The whole frame was analysed because the operator chose that, not because no outline
    #: could be found. The page distinguishes them; so should anything reading this.
    whole_frame: bool = False


def _clock(t_s: float) -> str:
    total = int(round(t_s))
    return f"{total // 60}:{total % 60:02d}"


def _parse_polygon(raw: str) -> list[list[float]] | None:
    """A drawn outline from the query string, or None. Refuses rather than repairs.

    `roi.validate` is the same check `PUT /roi` applies before saving one, so an outline that
    could not be stored cannot be analysed either - a polygon arriving over the wire is the
    caller's claim, and this is where it stops being taken on trust.
    """
    if not raw.strip():
        return None
    try:
        points = json.loads(raw)
    except (ValueError, RecursionError) as exc:  # 2000 nested brackets fit in the query
        raise HTTPException(422, f"polygon is not valid JSON: {exc}") from exc
    try:
        return roi.validate(points)
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(422, f"the outline is not usable: {exc}") from exc


def _derive_boundary(path: Path) -> list[list[float]] | None:
    """Measure a pitch outline from the clip itself, or None if it cannot be found.

    `roi.derive_from_video` - the routine `scripts/derive_roi.py` runs over recordings. It is
    a fallback and not a replacement: a hand-drawn or stored outline is better and is
    preferred above. What it replaces is *no boundary at all*, where the detector counts the
    neighbouring pitch as this one.

    Failure returns None rather than raising. A clip this cannot find turf in is still worth
    classifying without an outline, and the response says which happened.
    """
    return roi.derive_from_video(path)


def _analyse_file(path: Path, *, interval_s: float, window: int, camera: str,
                  drawn: list[list[float]] | None, whole_frame: bool):
    """The work of `/clip/analyse` once the upload is on disk: ``(result, polygon, derived)``.

    Synchronous on purpose, so the route can hand the whole of it to a worker thread.
    """
    limits.check_video_pixels(path, "clip")
    classify = _classifier()
    # `resolve`, not `get`: the camera is named the way production names it, and the
    # store's own keys are not that.
    # `whole_frame` beats every source, including a drawn outline and a stored one: it is
    # the one value here that is unambiguously a decision somebody made about this clip.
    polygon = None if whole_frame else (drawn or (roi.resolve(camera) if camera else None))
    derived_boundary = False
    if polygon is None and not whole_frame:
        # An uploaded clip usually names no camera, or names one the store has never
        # seen. Without an outline the detector counts the neighbouring pitch, the walkway
        # and the car park as if they were this pitch, so one is measured from the footage
        # itself - the same thing `scripts/run_slot_on_video.py` does.
        polygon = _derive_boundary(path)
        derived_boundary = polygon is not None

    try:
        # The boundary is passed, not painted in: the detector counts people whose feet are
        # inside it, exactly as `worker.run_slot` does.
        result = analyse_clip(
            path, classify, interval_s=interval_s, window=window,
            max_samples=MAX_SAMPLES, polygon=polygon, play_min=_play_min(),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result, polygon, derived_boundary


@router.post("/clip/analyse", response_model=ClipOut)
async def analyse(
    request: Request,
    interval_s: float = Query(DEFAULT_INTERVAL_S, gt=0.5, le=600),
    window: int = Query(DEFAULT_WINDOW, ge=1, le=31),
    camera: str = Query("", max_length=120),
    #: An outline drawn on the page, as a JSON array of normalised ``[x, y]`` pairs. A query
    #: parameter because the body is the video; eight points is about 120 characters, which a
    #: URL carries comfortably. It wins over both the stored boundary and the derived one -
    #: the operator is looking at this clip and neither of the others is.
    polygon: str = Query("", max_length=4000),
    #: The operator pressed *Use the whole frame*. This has to be its own parameter: an
    #: omitted `polygon` is what "I have not drawn one yet" looks like, and the fallback
    #: below answers that case by measuring one from the footage - so declining a boundary
    #: and never being offered one were indistinguishable on the wire, and the clip was
    #: analysed with the outline the operator had just rejected. Reported from use.
    whole_frame: bool = Query(False),
) -> ClipOut:
    """Sample the posted video every ``interval_s`` seconds and return its timeline.

    The video is deleted before this returns, whatever happens. ``window`` must be odd: an
    even window has no centre, so "the neighbours overruled it" stops being symmetric.
    """
    if window % 2 == 0:
        raise HTTPException(422, "window must be odd, so each sample has equal neighbours")
    # Before the upload and before the model: a malformed outline is the caller's mistake and
    # is answered as one, without receiving 600 MB or loading the detector first.
    drawn = _parse_polygon(polygon)

    # Delete on the way out rather than relying on the OS: this is footage of people.
    fd, path = limits.upload_file()
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
        # Everything from here is decoding and model passes - up to 360 of each - so it runs
        # in a worker thread. On the event loop it froze the dashboard, the API and every
        # live stream for as long as the clip took.
        result, polygon, derived_boundary = await limits.run_heavy(
            lambda: _analyse_file(path, interval_s=interval_s, window=window, camera=camera,
                                  drawn=drawn, whole_frame=whole_frame))
    finally:
        path.unlink(missing_ok=True)

    return ClipOut(
        camera=camera or None,
        boundary=bool(polygon),
        boundary_derived=derived_boundary,
        boundary_drawn=drawn is not None and not whole_frame,
        whole_frame=whole_frame,
        duration_s=result.duration_s,
        interval_s=result.interval_s,
        window=result.window,
        n_samples=len(result.samples),
        n_corrected=result.n_corrected,
        n_unreadable=len(result.unreadable),
        n_undecided=len(result.undecided),
        interval_widened=result.interval_widened,
        summary=result.summary(),
        dominant=result.dominant.value if result.dominant else None,
        samples=[
            SampleOut(
                index=s.index, t_s=s.t_s, clock=_clock(s.t_s), raw=s.raw.value,
                people=s.people, ball=s.ball,
                motion=None if s.motion is None else round(s.motion, 3),
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
