"""The Testing section: one hub for every way to try the system, and the booking-day test.

The single-video and two-camera reviewers answer *"what happened in this clip?"* on the
clip's own clock. A client testing the product asks a different question of a whole morning
of footage: *"which of my rental slots were played in?"*. The booking-day test answers that.
The client says what clock time the video starts and how long a slot is; the footage is
classified exactly as the reviewers classify it, and `timetable` adds the minutes up per slot.

Four things about it are deliberate.

**It runs in the background.** Hours of footage is minutes of CPU, longer than a request
should be held open. The upload is streamed to temporary files under the same size cap as the
reviewers, then handed to `api.jobs` - one worker, one analysis at a time - and the page polls
for progress.

**The same system as the reviewers, not a copy of it.** One video goes through the classifier
and boundary choice of `/api/v1/clip/analyse` (`clip_review._classifier`,
`_derive_boundary`, `roi.resolve`); two go through the detector pipeline of
`/api/v1/clip/pair/analyse` (`clip_pair._pipeline`). They are looked up on those modules at
call time, so a test that scripts the reviewers scripts this too.

**Footage is never kept; results are.** The temporary video files are deleted in the job's
``finally`` - or at once, if the job never gets submitted. What is saved under
``data_dir/test_runs`` is the per-slot table and a compact per-moment timeline: numbers and
labels, nothing a person could be recognised from.

**Comparing with bookings is a second step, on a saved run.** The footage is not needed again,
so a client can try one booking export, then another, without re-running the analysis. A slot
the footage did not settle is reported as "review" whatever the booking says.

**The rules are the client's to set, and are saved with the run.** How much of a slot must be
seen, and which share of play is a close call (`timetable.Rules`), can be given in percent with
the upload or applied afterwards to a saved run (``POST /api/v1/testing/runs/{id}/rejudge``).
Rejudging is exact without the footage: the run keeps every slot's unrounded seconds, which is
all the rules act on. A stored comparison with bookings is redone with the new results, from
the booking facts it kept per slot.

**A day of video is many GB, so it need not be uploaded.** An upload is capped per video by
``settings.testing_max_upload_mb`` (16 GB by default; an hour of real recording is ~630 MB per
camera) and refused up front with 507 when the temp folder's disk could not hold it. A
recording already on this machine, under ``settings.recordings_dir``, can be analysed in place
(``POST /api/v1/testing/day/local``): the name is resolved inside that folder and nowhere else,
and the file - the client's own - is never deleted, unlike an upload's temporary copy.

**A test can be stopped.** ``GET /api/v1/testing/jobs`` lists the recent tests, running and
queued first on the hub page, and ``POST /api/v1/testing/jobs/{id}/cancel`` stops one
(`api.jobs.cancel`): a queued test never starts, a running one stops at its next sample. An
upload's temporary video is deleted either way - by the job's ``finally`` when it was running,
by the clean-up handed to `jobs.submit` when it never started. A recording on this machine is
never deleted, cancelled or not.

Mutating routes: ``POST /api/v1/testing/day``, ``POST /api/v1/testing/day/local``,
``POST /api/v1/testing/jobs/{id}/cancel``, ``POST /api/v1/testing/runs/{id}/compare`` and
``POST /api/v1/testing/runs/{id}/rejudge``.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from pitch_occupancy import retention, timetable
from pitch_occupancy.api import clip_pair, clip_review, jobs, limits
from pitch_occupancy.api.layout import page
from pitch_occupancy.api.templating import template
from pitch_occupancy.clip_analysis import DEFAULT_WINDOW, analyse_clip
from pitch_occupancy.pair_analysis import analyse_pair
from pitch_occupancy.vision import roi

__all__ = ["router", "MAX_DAY_SAMPLES", "runs_dir", "recordings_dir", "VIDEO_SUFFIXES"]

router = APIRouter(tags=["testing"])

#: A ceiling on moments per day test. A full day at one a minute is 1,440, so the default
#: interval always fits; a finer interval over a long video is widened, and the run says so.
MAX_DAY_SAMPLES: int = 2000

_RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_CODES = {"C2_ACTIVE_PLAY": "P", "C1_EMPTY": "E", "C3_MAINTENANCE_NON_SPORTING": "M"}
_MB = 1024 * 1024
#: Room kept free on the temp folder's disk beyond the upload itself, so a large upload cannot
#: fill the disk the database and the live worker also write to.
_DISK_HEADROOM = 200 * _MB
#: How often, in bytes received, a streaming upload re-checks the free space.
_DISK_CHECK_EVERY = 256 * _MB
#: A booking CSV for one day is kilobytes; this is the most that is read. It used to be the
#: point the body was *truncated* at, after all of it had been buffered - so a gigabyte body
#: was held in memory first, and a CSV over the line was silently cut mid-row.
_MAX_BOOKING_BYTES = 20 * _MB

#: File types offered from (and accepted in) the recordings folder.
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts", ".mts", ".webm",
                            ".wmv", ".mpg", ".mpeg", ".dav", ".h264", ".264"})


def runs_dir() -> Path:
    """Where finished day tests are saved. Read at call time so a test can point it away."""
    from pitch_occupancy.config import settings

    return Path(settings.data_dir) / "test_runs"


def recordings_dir() -> Path:
    """Where recordings already on this machine are read from. Read at call time."""
    from pitch_occupancy.config import settings

    return Path(settings.recordings_dir)


def _max_upload_bytes() -> int:
    from pitch_occupancy.config import settings

    return int(settings.testing_max_upload_mb) * _MB


def _parse_hhmm(value: str, name: str) -> time:
    match = _HHMM.match(value.strip())
    if not match:
        raise HTTPException(422, f"{name} must be a clock time like 08:00, not {value!r}")
    return time(int(match.group(1)), int(match.group(2)))


def _run_path(run_id: str) -> Path:
    if not _RUN_ID.match(run_id):
        raise HTTPException(404, "no such test run")
    return runs_dir() / f"{run_id}.json"


def _load_run(run_id: str) -> dict[str, Any]:
    path = _run_path(run_id)
    if not path.is_file():
        raise HTTPException(404, "no such test run")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_run(run: dict[str, Any]) -> None:
    folder = runs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{run['id']}.json"
    scratch = target.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(run, indent=1), encoding="utf-8")
    os.replace(scratch, target)


def _new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def _rules(min_seen_pct: float | None, close_low_pct: float | None,
           close_high_pct: float | None) -> timetable.Rules:
    """The judging rules from the optional percentages, defaults for any left out.

    The 0-100 range is checked by the query parameters; the band's order is checked here,
    so the client is told in words rather than by a stack trace.
    """
    try:
        return timetable.Rules.from_percent(min_seen_pct, close_low_pct, close_high_pct)
    except ValueError as exc:
        d = timetable.DEFAULT_RULES
        low = d.close_low * 100 if close_low_pct is None else close_low_pct
        high = d.close_high * 100 if close_high_pct is None else close_high_pct
        raise HTTPException(422, f"The close-call band must start below where it ends "
                                 f"(you gave {low:g}% to {high:g}%).") from exc


def _rules_fields(rules: timetable.Rules) -> dict[str, Any]:
    """What a saved run keeps about its rules: the numbers, and the sentence shown with them."""
    return {"rules": rules.to_dict(), "rules_text": rules.describe()}


def _run_rules(run: dict[str, Any]) -> timetable.Rules:
    """The rules a saved run was judged with; a run from before rules were saved used the
    defaults."""
    try:
        return timetable.Rules.from_dict(run.get("rules"))
    except (KeyError, TypeError, ValueError):
        return timetable.DEFAULT_RULES


_PCT = {"ge": 0, "le": 100}


# --- the analysis, run in the background ----------------------------------------------------


def _single(path: Path, *, interval_s: float, camera: str, whole_frame: bool, progress):
    """One video, classified exactly as `/api/v1/clip/analyse` does it."""
    classify = clip_review._classifier()
    if whole_frame:
        polygon, source = None, "whole frame"
    elif camera and (stored := roi.resolve(camera)):
        polygon, source = stored, "stored"
    elif (derived := clip_review._derive_boundary(path)) is not None:
        polygon, source = derived, "derived"
    else:
        polygon, source = None, "none"
    result = analyse_clip(path, classify, interval_s=interval_s, window=DEFAULT_WINDOW,
                          max_samples=MAX_DAY_SAMPLES, polygon=polygon,
                          play_min=clip_review._play_min(), progress=progress)
    samples = [(s.t_s, s.smoothed) for s in result.samples]
    return samples, result.duration_s, result.interval_s, result.interval_widened, [source]


def _pair(paths: list[Path], *, interval_s: float, offset_s: float, cameras: tuple[str, str],
          whole_frame: bool, progress):
    """Two cameras, through the detector pipeline of `/api/v1/clip/pair/analyse`."""
    polygons, sources = [], []
    for path, camera in zip(paths, cameras, strict=True):
        if whole_frame:
            polygon, source = None, "whole frame"
        elif camera and (stored := roi.resolve(camera)):
            polygon, source = stored, "stored"
        elif (derived := clip_review._derive_boundary(path)) is not None:
            polygon, source = derived, "derived"
        else:
            polygon, source = None, "none"
        polygons.append(polygon)
        sources.append(source)
    result = analyse_pair(paths[0], paths[1], clip_pair._pipeline(), offset_s=offset_s,
                          interval_s=interval_s, window=DEFAULT_WINDOW,
                          max_samples=MAX_DAY_SAMPLES, polygons=(polygons[0], polygons[1]),
                          progress=progress)
    samples = [(s.t_s, s.smoothed) for s in result.samples]
    # Camera A's clock: the video start time the client typed is camera A's.
    return samples, result.end_s, result.interval_s, result.interval_widened, sources


def _build_run(meta: dict[str, Any], samples, duration_s: float, used_interval: float,
               widened: bool, boundaries: list[str]) -> dict[str, Any]:
    windows = timetable.slot_windows(meta["video_start"], duration_s, meta["slot_minutes"],
                                     meta["first_slot"])
    rules = meta.get("rules") or timetable.DEFAULT_RULES
    results = timetable.summarise(samples, windows, used_interval, rules=rules)
    note = ""
    if widened:
        note = (f"The video is long, so it was checked every {used_interval:.0f} seconds "
                f"instead of every {meta['interval_s']:g} seconds to fit "
                f"{MAX_DAY_SAMPLES} checks.")
    return {
        "id": meta["id"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "cameras": 2 if meta["two"] else 1,
        "video_name": meta["name"], "video_name_b": meta["name_b"],
        "start": f"{meta['video_start']:%H:%M}",
        "first_slot": f"{meta['first_slot']:%H:%M}" if meta["first_slot"] else None,
        "slot_minutes": meta["slot_minutes"],
        "interval_s": meta["interval_s"], "interval_used_s": round(used_interval, 3),
        "interval_widened": widened, "note": note,
        "offset_s": meta["offset_s"],
        "duration_s": round(duration_s, 2),
        "field_id": meta["field_id"] or None,
        "day": meta["day"].isoformat() if meta["day"] else None,
        "boundaries": boundaries,
        "n_samples": len(samples),
        "counts": _counts(results),
        **_rules_fields(rules),
        "slots": [r.to_dict() for r in results],
        # Per moment, as (seconds, letter): P playing, E empty, M maintenance, U undecided.
        "timeline": [[round(t, 1), _CODES.get(getattr(s, "value", s), "U")]
                     for t, s in samples],
        "comparison": None,
    }


def _counts(results: list[timetable.SlotResult]) -> dict[str, int]:
    counts = Counter(r.status.value for r in results)
    return {"slots": len(results), "USED": counts.get("USED", 0),
            "NOT_USED": counts.get("NOT_USED", 0), "REVIEW": counts.get("REVIEW", 0)}


def _make_work(paths: list[Path], meta: dict[str, Any], *, delete: bool = True):
    """The background job. ``delete`` removes the videos afterwards - right for an upload's
    temporary copies, wrong for a recording that belongs to the client."""

    def work(progress) -> dict[str, Any]:
        try:
            if meta["two"]:
                out = _pair(paths, interval_s=meta["interval_s"], offset_s=meta["offset_s"],
                            cameras=meta["cameras"], whole_frame=meta["whole_frame"],
                            progress=progress)
            else:
                out = _single(paths[0], interval_s=meta["interval_s"],
                              camera=meta["cameras"][0], whole_frame=meta["whole_frame"],
                              progress=progress)
        finally:
            if delete:
                for path in paths:
                    path.unlink(missing_ok=True)
        run = _build_run(meta, *out)
        _save_run(run)
        return run

    return work


def _label(meta: dict[str, Any]) -> str:
    """What the list of running tests calls this one: the video name(s)."""
    return meta["name"] + (f" + {meta['name_b']}" if meta["two"] else "")


def _deleter(paths: list[Path]):
    """The clean-up for an upload cancelled before its job started."""

    def delete() -> None:
        for path in paths:
            path.unlink(missing_ok=True)

    return delete


async def _receive_one(request: Request, path: Path, *, max_bytes: int) -> None:
    """Stream the body to ``path``, capped at ``max_bytes`` while streaming."""
    declared = _content_length(request)
    if declared is not None and declared > max_bytes:
        raise HTTPException(413, f"The video is {declared // _MB} MB; the limit is "
                                 f"{max_bytes // _MB} MB per video.")
    size = 0
    checked = 0
    with open(path, "wb") as handle:
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"The video is over {size // _MB} MB so far; the limit "
                                         f"is {max_bytes // _MB} MB per video.")
            # A body sent without a length was never checked against the free space, so it
            # is checked as it arrives: a full disk breaks the database and the snapshots.
            if size - checked >= _DISK_CHECK_EVERY:
                checked = size
                free = retention.free_bytes(path.parent)
                if free is not None and free < _DISK_HEADROOM:
                    raise HTTPException(507, "The disk is nearly full; the upload was stopped.")
            handle.write(chunk)
    if size == 0:
        raise HTTPException(422, "no video in the request body")


def _content_length(request: Request) -> int | None:
    try:
        value = int(request.headers.get("content-length", ""))
    except ValueError:
        return None
    return value if value >= 0 else None


def _check_disk(request: Request) -> None:
    """Refuse (507) an upload the temp folder's disk could not hold, before reading it.

    Only when the request says how big it is and the free space can be measured; otherwise
    the upload goes ahead and the per-video cap is the only limit.
    """
    need = _content_length(request)
    if not need:
        return
    folder = Path(tempfile.gettempdir())
    free = retention.free_bytes(folder)
    if free is not None and free < need + _DISK_HEADROOM:
        raise HTTPException(507, f"There is not enough free disk space on this computer for "
                                 f"this upload: it is {need // _MB} MB and only "
                                 f"{free // _MB} MB is free. Free some space, or put the "
                                 f"recording in {recordings_dir()} and use \"a recording on "
                                 f"this computer\" instead.")


def _meta(*, two: bool, start: str, slot_minutes: int, first_slot: str, interval_s: float,
          offset_s: float, field_id: str, day: date | None, whole_frame: bool, camera: str,
          camera_a: str, camera_b: str, name: str, name_b: str | None,
          rules: timetable.Rules) -> dict[str, Any]:
    """Everything the job needs besides the files; shared by the upload and local routes."""
    video_start = _parse_hhmm(start, "start")
    first = _parse_hhmm(first_slot, "first_slot") if first_slot.strip() else None
    return {
        "id": _new_run_id(), "two": two, "video_start": video_start,
        "first_slot": first, "slot_minutes": slot_minutes, "interval_s": interval_s,
        "offset_s": offset_s if two else 0.0, "field_id": field_id.strip(), "day": day,
        "whole_frame": whole_frame,
        "cameras": (camera_a or camera, camera_b) if two else (camera or camera_a, ""),
        "name": name or "video", "name_b": (name_b or "second video") if two else None,
        "rules": rules,
    }


@router.post("/api/v1/testing/day")
async def start_day_test(
    request: Request,
    start: str = Query(..., max_length=8, description="clock time the video starts, HH:MM"),
    slot_minutes: int = Query(90, ge=1, le=600),
    first_slot: str = Query("", max_length=8),
    interval_s: float = Query(60.0, ge=5, le=3600),
    size_a: int | None = Query(None, ge=1),
    offset_s: float = Query(0.0, ge=-86400, le=86400),
    field_id: str = Query("", max_length=120),
    day: date | None = None,
    whole_frame: bool = Query(False),
    camera: str = Query("", max_length=120),
    camera_a: str = Query("", max_length=120),
    camera_b: str = Query("", max_length=120),
    name: str = Query("", max_length=200),
    name_b: str = Query("", max_length=200),
    min_seen_pct: float | None = Query(None, **_PCT),
    close_low_pct: float | None = Query(None, **_PCT),
    close_high_pct: float | None = Query(None, **_PCT),
) -> dict[str, str]:
    """Receive one video (or A then B with ``size_a``) and queue the day test. Returns a job.

    ``min_seen_pct``, ``close_low_pct`` and ``close_high_pct`` (0-100, optional) set the
    judging rules; any left out keeps its default (50, 40, 60).
    """
    rules = _rules(min_seen_pct, close_low_pct, close_high_pct)
    two = size_a is not None
    meta = _meta(two=two, start=start, slot_minutes=slot_minutes, first_slot=first_slot,
                 interval_s=interval_s, offset_s=offset_s, field_id=field_id, day=day,
                 whole_frame=whole_frame, camera=camera, camera_a=camera_a, camera_b=camera_b,
                 name=name, name_b=name_b, rules=rules)
    if not jobs.has_room():
        raise HTTPException(429, f"{jobs.MAX_WAITING} tests are already waiting; try again when "
                                 f"one has finished.")
    _check_disk(request)
    max_each = _max_upload_bytes()
    paths: list[Path] = []
    for _ in range(2 if two else 1):
        # The shared prefix lets `limits.sweep_stale` remove what a restart orphaned.
        fd, tmp = tempfile.mkstemp(prefix=limits.UPLOAD_PREFIX, suffix=".upload")
        os.close(fd)
        paths.append(Path(tmp))
    submitted = False
    try:
        if two:
            await clip_pair._receive(request, size_a, (paths[0], paths[1]), max_each=max_each)
        else:
            await _receive_one(request, paths[0], max_bytes=max_each)
        # Refused now, as a 413 the page can show, rather than as a job that fails - or
        # exhausts memory - an hour into the queue.
        for path in paths:
            await anyio.to_thread.run_sync(limits.check_video_pixels, path)
        job_id = jobs.submit("day", _make_work(paths, meta), label=_label(meta),
                             on_cancel=_deleter(paths))
        submitted = True
    finally:
        if not submitted:
            for path in paths:
                path.unlink(missing_ok=True)
    return {"job_id": job_id, "run_id": meta["id"]}


# --- recordings already on this machine ---------------------------------------------------


def _video_info(path: Path) -> float | None:
    """Duration in seconds from the container's metadata, or None. Reads no frames."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
    except Exception:  # noqa: BLE001 - a listing must not fail on one odd file
        return None
    if not frames or not fps or frames <= 0 or fps <= 0 or fps > 1000:
        return None
    return round(frames / fps, 1)


def _resolve_recording(name: str) -> Path:
    """``name`` (relative to the recordings folder) as a real video file inside it, or 422.

    Resolved with symlinks followed and then checked to be inside the folder, so neither
    ``..`` nor a link pointing elsewhere reaches any other file on the machine.
    """
    root = recordings_dir().resolve()
    clean = (name or "").strip().replace("\\", "/")
    if not clean or clean.startswith("/") or ":" in clean or ".." in clean.split("/"):
        raise HTTPException(422, f"{name!r} is not a recording name: give the name as listed, "
                                 f"inside {root}")
    try:
        target = (root / clean).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(404, f"There is no recording called {clean!r} in {root}.") from exc
    if not target.is_relative_to(root) or target == root:
        raise HTTPException(422, f"{clean!r} is not inside the recordings folder {root}.")
    if not target.is_file():
        raise HTTPException(404, f"There is no recording called {clean!r} in {root}.")
    if target.suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(422, f"{clean!r} is not a video file this test reads "
                                 f"({', '.join(sorted(VIDEO_SUFFIXES))}).")
    return target


@router.get("/api/v1/testing/recordings")
def list_recordings() -> dict[str, Any]:
    """Video files in the recordings folder and the folders directly inside it.

    Names are relative to the folder (what ``/day/local`` takes). Duration comes from the
    file's own metadata and may be missing. Links that point outside the folder are left out.
    """
    root = recordings_dir()
    out: list[dict[str, Any]] = []
    if root.is_dir():
        real = root.resolve()
        candidates = [*root.glob("*"), *root.glob("*/*")]
        for path in candidates:
            if path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            try:
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(real) or not resolved.is_file():
                    continue
                stat = resolved.stat()
            except (OSError, RuntimeError):
                continue
            out.append({
                "name": path.relative_to(root).as_posix(),
                "size_mb": round(stat.st_size / _MB, 1),
                "duration_s": _video_info(resolved),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            })
    out.sort(key=lambda r: r["name"].lower())
    return {"folder": str(root), "exists": root.is_dir(), "recordings": out}


class LocalDayIn(BaseModel):
    """The upload route's options, as JSON, plus which recording(s) to read."""

    file_a: str = Field(max_length=500)
    file_b: str | None = Field(default=None, max_length=500)
    start: str = Field(max_length=8)
    slot_minutes: int = Field(default=90, ge=1, le=600)
    first_slot: str = Field(default="", max_length=8)
    interval_s: float = Field(default=60.0, ge=5, le=3600)
    offset_s: float = Field(default=0.0, ge=-86400, le=86400)
    field_id: str = Field(default="", max_length=120)
    day: date | None = None
    whole_frame: bool = False
    camera: str = Field(default="", max_length=120)
    camera_a: str = Field(default="", max_length=120)
    camera_b: str = Field(default="", max_length=120)
    min_seen_pct: float | None = Field(default=None, **_PCT)
    close_low_pct: float | None = Field(default=None, **_PCT)
    close_high_pct: float | None = Field(default=None, **_PCT)


@router.post("/api/v1/testing/day/local")
def start_local_day_test(body: LocalDayIn) -> dict[str, str]:
    """Queue the day test on recording(s) already in the recordings folder. Returns a job.

    The files are read where they are and **not deleted** afterwards: they are the client's.
    """
    rules = _rules(body.min_seen_pct, body.close_low_pct, body.close_high_pct)
    paths = [_resolve_recording(body.file_a)]
    two = bool(body.file_b and body.file_b.strip())
    if two:
        paths.append(_resolve_recording(body.file_b or ""))
        if paths[0] == paths[1]:
            raise HTTPException(422, "Choose two different recordings for the two cameras.")
    root = recordings_dir().resolve()
    names = [p.relative_to(root).as_posix() for p in paths]
    meta = _meta(two=two, start=body.start, slot_minutes=body.slot_minutes,
                 first_slot=body.first_slot, interval_s=body.interval_s,
                 offset_s=body.offset_s, field_id=body.field_id, day=body.day,
                 whole_frame=body.whole_frame, camera=body.camera, camera_a=body.camera_a,
                 camera_b=body.camera_b, name=names[0], name_b=names[1] if two else None,
                 rules=rules)
    meta["source"] = "recording"
    # No clean-up on cancel, and ``delete=False``: these files are the client's.
    job_id = jobs.submit("day", _make_work(paths, meta, delete=False), label=_label(meta))
    return {"job_id": job_id, "run_id": meta["id"]}


@router.get("/api/v1/testing/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    snap = jobs.get(job_id)
    if snap is None:
        raise HTTPException(404, "no such job; it may have finished long ago")
    return snap


@router.get("/api/v1/testing/jobs")
def list_jobs(limit: int = Query(20, ge=1, le=50)) -> list[dict[str, Any]]:
    """Recent tests, newest first: status, progress, when and the video name - no results."""
    return jobs.list_jobs(limit)


@router.post("/api/v1/testing/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    """Stop a queued or running test. 404 for an unknown job, 409 for one already finished.

    A running test stops at its next sample, so the answer may still say ``running`` with
    ``cancel_requested``; poll the job to see it become ``cancelled``.
    """
    snap = jobs.get(job_id)
    if snap is None:
        raise HTTPException(404, "no such job; it may have finished long ago")
    if not jobs.cancel(job_id):
        status = (jobs.get(job_id) or snap)["status"]
        raise HTTPException(409, f"This test has already {_FINISHED_WORD.get(status, 'ended')}.")
    return jobs.get(job_id) or snap


_FINISHED_WORD = {"done": "finished", "failed": "failed", "cancelled": "been cancelled"}


# --- saved runs -----------------------------------------------------------------------------


@router.get("/api/v1/testing/runs")
def list_runs(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    folder = runs_dir()
    if not folder.is_dir():
        return []
    out = []
    for path in folder.glob("*.json"):
        if not _RUN_ID.match(path.stem):
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({
            "id": run.get("id"), "created": run.get("created"),
            "video_name": run.get("video_name"), "cameras": run.get("cameras", 1),
            "start": run.get("start"), "slot_minutes": run.get("slot_minutes"),
            "field_id": run.get("field_id"), "day": run.get("day"),
            "counts": run.get("counts", {}),
        })
    out.sort(key=lambda r: r["created"] or "", reverse=True)
    return out[:limit]


@router.get("/api/v1/testing/runs/{run_id}.csv")
def run_csv(run_id: str) -> PlainTextResponse:
    run = _load_run(run_id)
    checks = (run.get("comparison") or {}).get("checks")
    body = timetable.to_csv(run["slots"], checks, rules=_run_rules(run))
    return PlainTextResponse(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="booking-day-{run_id}.csv"'})


@router.get("/api/v1/testing/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return _load_run(run_id)


@router.post("/api/v1/testing/runs/{run_id}/compare")
async def compare_run(
    run_id: str,
    request: Request,
    field_id: str = Query("", max_length=120),
    day: date | None = None,
    use_configured: bool = Query(False),
) -> dict[str, Any]:
    """Set a saved run's slots against a booking CSV (the body) or the configured source."""
    run = _load_run(run_id)
    field = field_id.strip() or (run.get("field_id") or "")
    when = day or (date.fromisoformat(run["day"]) if run.get("day") else None)
    if not field or when is None:
        raise HTTPException(422, "Give the pitch's field id and the date of the video to "
                                 "compare it with bookings.")
    if use_configured:
        try:
            from pitch_occupancy.booking_sources import load_source
        except ImportError as exc:
            raise HTTPException(503, "Booking sources are not available on this server "
                                     "yet. Upload a booking file instead.") from exc
        src = load_source()
        if src is None:
            raise HTTPException(404, "No booking source is configured. Upload a booking "
                                     "file (CSV) instead, or ask the administrator to set "
                                     "one up under Bookings.")
        try:
            records = list(src.read())
        except (OSError, ValueError) as exc:
            raise HTTPException(422, f"The configured bookings could not be read: {exc}") \
                from exc
        source = getattr(src, "source", None) or "configured"
    else:
        body = await limits.read_body(request, _MAX_BOOKING_BYTES, what="the booking file")
        text = body.decode("utf-8-sig", errors="replace")
        if not text.strip():
            raise HTTPException(422, "Upload a booking file (CSV), or use the configured "
                                     "bookings.")
        try:
            from pitch_occupancy.booking_sources import parse_csv_text
        except ImportError as exc:
            raise HTTPException(503, "Reading uploaded booking files is not available on "
                                     "this server yet.") from exc
        try:
            # A 20 MB sheet takes a while to parse; off the event loop, which every other
            # request shares.
            records = await anyio.to_thread.run_sync(
                lambda: parse_csv_text(text, source="upload"))
        except ValueError as exc:
            raise HTTPException(422, f"The booking file could not be read: {exc}") from exc
        source = "upload"

    results = [timetable.SlotResult.from_dict(d) for d in run["slots"]]
    checks = timetable.compare(results, records, field_id=field, day=when)
    days = {when + timedelta(days=r.window.day_offset) for r in results}
    n_bookings = sum(1 for r in records if r.field_id.strip() == field and r.date in days)
    comparison = {
        "field_id": field, "day": when.isoformat(), "source": str(source),
        "n_bookings": n_bookings,
        "counts": dict(Counter(c.outcome for c in checks)),
        "checks": [c.to_dict() for c in checks],
    }
    run["comparison"] = comparison
    _save_run(run)
    return {"run_id": run_id, **comparison}


@router.post("/api/v1/testing/runs/{run_id}/rejudge")
def rejudge_run(
    run_id: str,
    min_seen_pct: float | None = Query(None, **_PCT),
    close_low_pct: float | None = Query(None, **_PCT),
    close_high_pct: float | None = Query(None, **_PCT),
) -> dict[str, Any]:
    """Judge a saved run's slots again under new rules, without the video. Returns the run.

    Exact, from each slot's stored seconds (`timetable.rejudge`) - the per-moment timeline is
    not needed. A stored comparison with bookings is redone from the booking facts it kept
    per slot (`timetable.recheck`), so the saved run never shows results judged one way and
    a comparison judged another.
    """
    run = _load_run(run_id)
    rules = _rules(min_seen_pct, close_low_pct, close_high_pct)
    before = _run_rules(run)
    results = timetable.rejudge((timetable.SlotResult.from_dict(d) for d in run["slots"]),
                                rules)
    changed = sum(1 for old, new in zip(run["slots"], results, strict=True)
                  if old["status"] != new.status.value)
    run["slots"] = [r.to_dict() for r in results]
    run["counts"] = _counts(results)
    run.update(_rules_fields(rules))
    run["rejudged"] = {"at": datetime.now().isoformat(timespec="seconds"),
                       "previous_rules_text": before.describe(), "slots_changed": changed}
    if run.get("comparison"):
        checks = timetable.recheck(results, run["comparison"].get("checks") or [])
        run["comparison"]["checks"] = [c.to_dict() for c in checks]
        run["comparison"]["counts"] = dict(Counter(c.outcome for c in checks))
        run["comparison"]["rejudged"] = True
    _save_run(run)
    return run


# --- pages ----------------------------------------------------------------------------------


@router.get("/testing/video", include_in_schema=False)
def _alias_video() -> RedirectResponse:
    return RedirectResponse("/clip", status_code=307)


@router.get("/testing/two-cameras", include_in_schema=False)
def _alias_pair() -> RedirectResponse:
    return RedirectResponse("/clip/pair", status_code=307)


@router.get("/testing/images", include_in_schema=False)
def _alias_images() -> RedirectResponse:
    return RedirectResponse("/images", status_code=307)


_HUB_BODY = template("testing_hub_body.html")

_HUB_SCRIPT = template("testing_hub_script.js")

_HUB_CSS = template("testing_hub_css.css")


@router.get("/testing", response_class=HTMLResponse, include_in_schema=False)
def hub() -> HTMLResponse:
    return HTMLResponse(page("Pitch Occupancy - Testing", "Testing", _HUB_BODY,
                             active="testing", css=_HUB_CSS, script=_HUB_SCRIPT))


_DAY_CSS = template("testing_day_css.css")

_DAY_BODY = template("testing_day_body.html")

_DAY_SCRIPT = template("testing_day_script.js")


@router.get("/testing/day", response_class=HTMLResponse, include_in_schema=False)
def day_page() -> HTMLResponse:
    return HTMLResponse(page("Pitch Occupancy - Booking-day test", "Booking-day test",
                             _DAY_BODY, active="testing", css=_DAY_CSS, script=_DAY_SCRIPT))
