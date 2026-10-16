"""Upload two videos of one pitch, get one timeline: the two-camera reviewer's route.

`clip_review.py` is the single-camera reviewer and is left exactly as it was. This is its
two-camera counterpart over `pair_analysis.analyse_pair`, and it keeps the same three promises
- nothing kept, the body is the video, corrections reported - with one change each needs.

**The body is both videos, back to back.** Still raw bytes and still no multipart dependency:
the page posts ``new Blob([fileA, fileB])`` and says where A ends with ``size_a``. Two requests
would have meant holding the first file on the server while waiting for the second, which is
exactly the retention this route exists not to have.

**It always runs the detector-first path.** The point of two cameras is adding what they see,
and only the detector path counts. `pipeline.shared()` follows `settings.default_model_key`,
which is still the probe; so this assembles its own with the detector `configs/rules.json`
names, once per process, and says which in the response.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pitch_occupancy.api import limits
from pitch_occupancy.api.clip_review import (
    MAX_UPLOAD_BYTES,
    SegmentOut,
    _clock,
    _derive_boundary,
    _parse_polygon,
)
from pitch_occupancy.clip_analysis import DEFAULT_INTERVAL_S, DEFAULT_WINDOW, MAX_SAMPLES
from pitch_occupancy.pair_analysis import CAMERAS, analyse_pair, tally
from pitch_occupancy.vision import roi

__all__ = ["router"]

router = APIRouter(prefix="/api/v1", tags=["clip review"])

_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()


def _pipeline():
    """The detector-first pipeline, assembled once. A function so a test can swap it.

    Locked because it is called from worker threads now, and two first requests arriving
    together would otherwise each load the detector.
    """
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            from pitch_occupancy.config import settings
            from pitch_occupancy.pipeline import assemble
            from pitch_occupancy.vision.rules import RuleConfig

            cfg = RuleConfig.load(settings.rules_path)
            _PIPELINE = assemble(cfg.detector, require_boundary=False, rules=cfg)
    return _PIPELINE


class SideOut(BaseModel):
    camera: str
    readable: bool
    state: str | None = None
    confidence: float = 0.0
    people: int | None = None
    ball: bool | None = None
    motion: float | None = None
    rule: int = 0


class PairSampleOut(BaseModel):
    index: int
    t_s: float
    clock: str
    #: Where camera B was read, on B's own clock.
    clock_b: str
    sides: list[SideOut]
    raw: str
    smoothed: str
    confidence: float
    corrected: bool
    #: Both cameras together: people added, ball from either.
    people: int | None = None
    ball: bool | None = None
    ball_sides: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    if_overlapping: str | None = None
    depends_on_sum: bool = False
    ball_one_side: bool = False
    split_view: bool = False
    one_side_blind: bool = False


class BoundaryOut(BaseModel):
    camera: str
    #: "drawn", "stored", "derived", "whole frame" or "none".
    source: str


class PairOut(BaseModel):
    model: str
    rules: dict[str, object]
    duration_s: float
    interval_s: float
    window: int
    offset_s: float
    start_s: float
    end_s: float
    interval_widened: bool
    n_samples: int
    n_corrected: int
    n_depends_on_sum: int
    n_ball_one_side: int
    n_split_view: int
    n_one_side_blind: int
    agrees_with_pitch: dict[str, int]
    shares: dict[str, float]
    summary: str
    dominant: str | None
    boundaries: list[BoundaryOut]
    samples: list[PairSampleOut]
    segments: list[SegmentOut] = Field(default_factory=list)


async def _receive(request: Request, size_a: int, dirs: tuple[Path, Path], *,
                   max_each: int = MAX_UPLOAD_BYTES) -> None:
    """Stream the body into the two files, switching at ``size_a``. Capped while streaming.

    ``max_each`` is the limit per video. A file over it is refused as too large (413) rather
    than as a malformed request: an hour of real recording is about 630 MB, so "too large" is
    the answer a person with a real file most needs to read correctly.
    """
    mb = 1024 * 1024
    if size_a <= 0:
        raise HTTPException(422, "size_a must be the byte length of camera A's video")
    if size_a > max_each:
        raise HTTPException(
            413, f"camera A's video is {size_a // mb} MB; the limit is {max_each // mb} MB "
                 f"per video")
    cap = size_a + max_each
    written = 0
    handles = [open(p, "wb") for p in dirs]  # noqa: SIM115 - closed in finally
    try:
        async for chunk in request.stream():
            if written + len(chunk) > cap:
                raise HTTPException(
                    413, f"camera B's video is over the limit of {max_each // mb} MB per video")
            split = max(0, min(len(chunk), size_a - written))
            handles[0].write(chunk[:split])
            handles[1].write(chunk[split:])
            written += len(chunk)
    finally:
        for h in handles:
            h.close()
    if written <= size_a:
        raise HTTPException(422, "the body holds no second video after size_a bytes")


def _analyse_files(paths: list[Path], *, cameras: tuple[str, str],
                   drawn: tuple[roi.Polygon | None, roi.Polygon | None], whole_frame: bool,
                   offset_s: float, interval_s: float, window: int):
    """The work of `/clip/pair/analyse` once both files are on disk, for a worker thread.

    Returns ``(result, sources, pipeline)``.
    """
    polygons = []
    sources = []
    for path, camera, drawn_one in zip(paths, cameras, drawn, strict=True):
        limits.check_video_pixels(path, f"camera {camera or len(sources) + 1}'s video")
        if whole_frame:
            polygon, source = None, "whole frame"
        elif drawn_one:
            polygon, source = drawn_one, "drawn"
        elif camera and (stored := roi.resolve(camera)):
            polygon, source = stored, "stored"
        elif (derived := _derive_boundary(path)) is not None:
            polygon, source = derived, "derived"
        else:
            polygon, source = None, "none"
        polygons.append(polygon)
        sources.append(source)

    pipeline = _pipeline()
    try:
        result = analyse_pair(paths[0], paths[1], pipeline, offset_s=offset_s,
                              interval_s=interval_s, window=window,
                              max_samples=MAX_SAMPLES,
                              polygons=(polygons[0], polygons[1]))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result, sources, pipeline


@router.post("/clip/pair/analyse", response_model=PairOut)
async def analyse_two(
    request: Request,
    size_a: int = Query(..., ge=1),
    offset_s: float = Query(0.0, ge=-3600, le=3600),
    interval_s: float = Query(DEFAULT_INTERVAL_S, gt=0.5, le=600),
    window: int = Query(DEFAULT_WINDOW, ge=1, le=31),
    camera_a: str = Query("", max_length=120),
    camera_b: str = Query("", max_length=120),
    polygon_a: str = Query("", max_length=4000),
    polygon_b: str = Query("", max_length=4000),
    whole_frame: bool = Query(False),
) -> PairOut:
    """Both videos in one body, A first; the pitch's timeline from the two together.

    Each camera's boundary is chosen the way `/clip/analyse` chooses one - drawn, else
    stored under the camera's name, else measured from that camera's own footage - and
    the response says which each camera got. Both files are deleted before this returns.
    """
    if window % 2 == 0:
        raise HTTPException(422, "window must be odd, so each sample has equal neighbours")
    # Before either upload and before the model: a malformed outline is a 422 that costs
    # nothing, not one discovered after two videos have arrived.
    drawn_pair = (_parse_polygon(polygon_a), _parse_polygon(polygon_b))
    paths: list[Path] = []
    try:
        for _ in CAMERAS:
            fd, path = limits.upload_file()
            os.close(fd)
            paths.append(path)
        await _receive(request, size_a, (paths[0], paths[1]))
        # Decoding two videos and running the detector on both is minutes of CPU, so it runs
        # in a worker thread rather than on the event loop every other request shares.
        result, sources, pipeline = await limits.run_heavy(
            lambda: _analyse_files(paths, cameras=(camera_a, camera_b), drawn=drawn_pair,
                                   whole_frame=whole_frame, offset_s=offset_s,
                                   interval_s=interval_s, window=window))
    finally:
        for path in paths:
            path.unlink(missing_ok=True)

    cfg = pipeline.rules
    from pitch_occupancy.vision.rules import MinuteState

    return PairOut(
        model=result.model,
        rules={} if cfg is None else {
            "play_min": cfg.play_min, "small_group_max": cfg.small_group_max,
            "require_ball": cfg.require_ball, "require_motion": cfg.require_motion,
            "burst_frames": cfg.burst_frames, "frozen": cfg.frozen},
        duration_s=result.duration_s, interval_s=result.interval_s, window=result.window,
        offset_s=result.offset_s, start_s=result.start_s, end_s=result.end_s,
        interval_widened=result.interval_widened,
        n_samples=len(result.samples), n_corrected=result.n_corrected,
        n_depends_on_sum=result.count("depends_on_sum"),
        n_ball_one_side=result.count("ball_one_side"),
        n_split_view=result.count("split_view"),
        n_one_side_blind=result.count("one_side_blind"),
        agrees_with_pitch={c: tally(result.samples).get(c, 0) for c in CAMERAS},
        shares={s.value: round(result.share(s), 4) for s in MinuteState},
        summary=result.summary(),
        dominant=result.dominant.value if result.dominant else None,
        boundaries=[BoundaryOut(camera=c, source=s) for c, s in zip(CAMERAS, sources,
                                                                     strict=True)],
        samples=[
            PairSampleOut(
                index=s.index, t_s=s.t_s, clock=_clock(s.t_s),
                clock_b=_clock(max(0.0, s.t_s - result.offset_s)),
                sides=[SideOut(camera=d.camera, readable=d.readable,
                               state=d.state.value if d.state else None,
                               confidence=d.confidence, people=d.people, ball=d.ball,
                               motion=None if d.motion is None else round(d.motion, 3),
                               rule=d.rule) for d in s.sides],
                raw=s.raw.value, smoothed=s.smoothed.value, confidence=s.confidence,
                corrected=s.corrected, people=s.people, ball=s.ball,
                ball_sides=list(s.ball_sides), trace=list(s.rule_trace),
                if_overlapping=s.if_overlapping.value if s.if_overlapping else None,
                depends_on_sum=s.depends_on_sum, ball_one_side=s.ball_one_side,
                split_view=s.split_view, one_side_blind=s.one_side_blind,
            )
            for s in result.samples
        ],
        segments=[
            SegmentOut(
                state=g.state.value, label=g.state.name.lower().replace("_", " "),
                starts_after=g.starts_after, starts_by=g.starts_by, ends_by=g.ends_by,
                duration_s=g.duration_s, n_samples=g.n_samples, n_corrected=g.n_corrected,
                describe=g.describe(),
            )
            for g in result.segments
        ],
    )


@router.get("/clip/pair/page", response_class=HTMLResponse, include_in_schema=False)
def page() -> HTMLResponse:
    from pitch_occupancy.api.clip_pair_page import PAIR_HTML

    return HTMLResponse(PAIR_HTML)
