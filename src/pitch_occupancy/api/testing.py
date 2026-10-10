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

**The same system as the reviewers, not a copy of it.** One video goes through the classifier,
gates and boundary choice of `/api/v1/clip/analyse` (`clip_review._classifier`, `_gates`,
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

Mutating routes: ``POST /api/v1/testing/day``, ``POST /api/v1/testing/day/local``,
``POST /api/v1/testing/runs/{id}/compare`` and ``POST /api/v1/testing/runs/{id}/rejudge``.
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

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from pitch_occupancy import retention, timetable
from pitch_occupancy.api import clip_pair, clip_review, jobs
from pitch_occupancy.api.layout import page
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
    """One video, chosen and gated exactly as `/api/v1/clip/analyse` does it."""
    classify = clip_review._classifier()
    if whole_frame:
        polygon, source = None, "whole frame"
    elif camera and (stored := roi.resolve(camera)):
        polygon, source = stored, "stored"
    elif (derived := clip_review._derive_boundary(path)) is not None:
        polygon, source = derived, "derived"
    else:
        polygon, source = None, "none"
    if polygon:
        inner = classify

        def classify(frame):  # noqa: F811 - the same masking wrapper as clip_review
            return inner(roi.apply(frame, polygon), polygon=polygon)

    motion_gate, person_gate = clip_review._gates()
    result = analyse_clip(path, classify, interval_s=interval_s, window=DEFAULT_WINDOW,
                          max_samples=MAX_DAY_SAMPLES, polygon=polygon,
                          motion_gate=motion_gate, person_gate=person_gate,
                          progress=progress)
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


async def _receive_one(request: Request, path: Path, *, max_bytes: int) -> None:
    """Stream the body to ``path``, capped at ``max_bytes`` while streaming."""
    declared = _content_length(request)
    if declared is not None and declared > max_bytes:
        raise HTTPException(413, f"The video is {declared // _MB} MB; the limit is "
                                 f"{max_bytes // _MB} MB per video.")
    size = 0
    with open(path, "wb") as handle:
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"The video is over {size // _MB} MB so far; the limit "
                                         f"is {max_bytes // _MB} MB per video.")
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
    _check_disk(request)
    max_each = _max_upload_bytes()
    paths: list[Path] = []
    for _ in range(2 if two else 1):
        fd, tmp = tempfile.mkstemp(suffix=".upload")
        os.close(fd)
        paths.append(Path(tmp))
    submitted = False
    try:
        if two:
            await clip_pair._receive(request, size_a, (paths[0], paths[1]), max_each=max_each)
        else:
            await _receive_one(request, paths[0], max_bytes=max_each)
        job_id = jobs.submit("day", _make_work(paths, meta))
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
    job_id = jobs.submit("day", _make_work(paths, meta, delete=False))
    return {"job_id": job_id, "run_id": meta["id"]}


@router.get("/api/v1/testing/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    snap = jobs.get(job_id)
    if snap is None:
        raise HTTPException(404, "no such job; it may have finished long ago")
    return snap


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
        body = (await request.body())[: 20 * 1024 * 1024]
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
            records = parse_csv_text(text, source="upload")
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


_HUB_BODY = """
<h2>Testing</h2>
<p class="sub2">Try the system on your own footage or photos. Nothing you upload is kept -
only the results.</p>
<div class="grid2">
  <a class="card" href="/clip"><h3>Test a video</h3>
    <p>Upload one video of a pitch and see, moment by moment, whether it was empty, being
    played on, or had people on it who were not playing.</p></a>
  <a class="card" href="/clip/pair"><h3>Two cameras</h3>
    <p>Upload the videos from both cameras of one pitch, recorded at the same time, and see
    the answer for the whole pitch.</p></a>
  <a class="card" href="/images"><h3>Test images</h3>
    <p>Upload one or more photos of a pitch and see what the system makes of each.</p></a>
  <a class="card" href="/testing/day"><h3>Booking-day test</h3>
    <p>Upload a long video, say when it starts and how long your booking slots are, and see
    which slots were used - and, if you like, compare them with your bookings.</p></a>
</div>
<h2>Previous booking-day tests</h2>
<div id="runs" class="card"><p class="muted">Loading...</p></div>
"""

_HUB_SCRIPT = r"""
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
fetch('/api/v1/testing/runs').then(r=>r.json()).then(rows=>{
  const box=document.getElementById('runs');
  if(!rows.length){box.innerHTML='<p class="muted">No tests yet.</p>';return;}
  box.innerHTML='<ul class="runs">'+rows.map(r=>{
    const c=r.counts||{};
    return '<li><a href="/testing/day?run='+encodeURIComponent(r.id)+'">'+
      esc(r.video_name||'video')+'</a> <span class="muted">'+esc((r.created||'').replace('T',' '))+
      ' &middot; starts '+esc(r.start)+' &middot; '+esc(r.slot_minutes)+'-minute slots'+
      (r.cameras==2?' &middot; two cameras':'')+'</span><br><span class="mono">'+
      (c.slots||0)+' slots: '+(c.USED||0)+' used, '+(c.NOT_USED||0)+' not used, '+
      (c.REVIEW||0)+' to review</span></li>';}).join('')+'</ul>';
}).catch(()=>{document.getElementById('runs').innerHTML=
  '<p class="muted">Could not load previous tests.</p>';});
"""

_HUB_CSS = """
ul.runs{list-style:none;margin:0;padding:0}
ul.runs li{padding:9px 0;border-bottom:1px solid var(--line);font-size:13.5px}
ul.runs li:last-child{border-bottom:none}
ul.runs a{color:var(--accent);font-weight:600;text-decoration:none}
"""


@router.get("/testing", response_class=HTMLResponse, include_in_schema=False)
def hub() -> HTMLResponse:
    return HTMLResponse(page("Pitch Occupancy - Testing", "Testing", _HUB_BODY,
                             active="testing", css=_HUB_CSS, script=_HUB_SCRIPT))


_DAY_CSS = """
.pair{display:grid;gap:14px;grid-template-columns:1fr 1fr}
@media (max-width:700px){.pair{grid-template-columns:1fr}}
.drop .cam{font:700 12px 'JetBrains Mono',monospace;letter-spacing:.1em;color:var(--accent)}
.drop.has{border-style:solid;border-color:var(--accent)}
.form{display:grid;gap:4px 16px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  margin-top:16px}
.form .field{margin:0 0 10px}
details{margin:6px 0 0;font-size:13px;color:var(--ink-2)}
details summary{cursor:pointer;color:var(--accent)}
.field input[type=checkbox]{width:auto}
.slots{display:flex;gap:3px;flex-wrap:wrap;margin:6px 0 4px}
.slots div{flex:1 1 70px;min-width:62px;border-radius:6px;padding:7px 6px;font:600 11px
  'JetBrains Mono',monospace;text-align:center;border:1px solid var(--line)}
.slots .USED{background:var(--play);color:#fff;border-color:var(--play)}
.slots .NOT_USED{background:var(--empty-soft);color:var(--empty)}
.slots .REVIEW{background:var(--flag-soft);color:var(--flag)}
.slots small{display:block;font-weight:500;opacity:.85}
.pill.USED{background:var(--play-soft);color:var(--play)}
.pill.NOT_USED{background:var(--empty-soft);color:var(--empty)}
.pill.REVIEW{background:var(--flag-soft);color:var(--flag)}
.pill.ok{background:var(--play-soft);color:var(--play)}
.pill.bad{background:var(--flag-soft);color:var(--flag)}
.pill.rev{background:var(--surface-2);color:var(--ink-2)}
.tl i.U{background:repeating-linear-gradient(45deg,var(--surface-2) 0 4px,var(--line) 4px 8px)}
.tl i.P{background:var(--play)}.tl i.E{background:var(--empty-soft)}.tl i.M{background:var(--maint)}
.btnrow{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0 0}
a.act{font:600 13px Archivo,sans-serif;padding:9px 18px;border-radius:6px;
  border:1px solid var(--accent);color:var(--accent);text-decoration:none;background:var(--surface)}
td.reason{font-size:12.5px;color:var(--ink-2);min-width:200px}
#prog{display:none}#prog.on{display:block}
#res{display:none}#res.on{display:block}
.bk{display:none}.bk.on{display:table-cell}
details.rejudge{margin:14px 0 0}
.srcpick{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 12px;font-size:14px}
.srcpick label{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
.srcpick input{width:auto}
@media print{
  :root{--ink:#000;--ink-2:#000;--ink-3:#333;--ground:#fff;--surface:#fff;--surface-2:#fff;
    --line:#999}
  body,main{background:#fff!important;color:#000!important}
  header,nav,#setup,#prog,#localBox,.btnrow,button,.drop,details.rejudge,#cBox .form,
    #cBox .btnrow,#cBox .sub2,#note:empty{display:none!important}
  main{padding:0!important;max-width:none!important}
  #res{display:block!important}
  .scroll{overflow:visible!important}
  table{min-width:0!important;width:100%}
  tr,.slots div,.tile{break-inside:avoid;page-break-inside:avoid}
  thead{display:table-header-group}
  h2{break-after:avoid;page-break-after:avoid}
  .slots,.tl,.pill,.sw,.tile{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .card,.tile{box-shadow:none!important;border-color:#999!important}
  a{color:#000;text-decoration:none}
}
"""

_DAY_BODY = """
<h2>Booking-day test</h2>
<p class="sub2">Upload a video of a pitch, tell us what time it starts and how long your
booking slots are. For every slot you get how long the pitch was played on, empty or used for
maintenance, and whether the slot was used. Nothing you upload is kept - only the results.</p>

<div id="setup">
<div class="srcpick" role="radiogroup" aria-label="Where the video comes from">
  <label><input type="radio" name="src" value="upload" checked> Upload a video</label>
  <label><input type="radio" name="src" value="local"> Use a recording on this computer</label>
</div>
<div id="localBox" class="card" hidden>
  <div class="form" style="margin-top:0">
    <div class="field"><label for="recA">Recording</label><select id="recA"></select></div>
    <div class="field"><label for="recB">Second camera's recording (optional)</label>
      <select id="recB"></select></div>
  </div>
  <p class="sub2" id="recHint" style="margin:0"></p>
</div>
<div class="pair" id="uploadBox">
  <label class="drop" id="dropA"><span class="cam">VIDEO</span><br>
    <strong id="nameA">Drop or choose a video</strong><p>the pitch, recorded over the day</p>
    <input type="file" accept="video/*" id="fileA" hidden></label>
  <label class="drop" id="dropB"><span class="cam">SECOND CAMERA (OPTIONAL)</span><br>
    <strong id="nameB">Drop or choose a video</strong>
    <p>only if the pitch has two cameras: the other half, same time</p>
    <input type="file" accept="video/*" id="fileB" hidden></label>
</div>

<div class="form">
  <div class="field"><label for="start">Video starts at</label>
    <input id="start" type="time" value="08:00" required></div>
  <div class="field"><label for="slot">Slot length (minutes)</label>
    <input id="slot" type="number" min="1" max="600" value="90"></div>
  <div class="field"><label for="first"
      title="Leave empty if the first slot starts when the video starts">First slot starts at
      (optional)</label><input id="first" type="time"></div>
  <div class="field"><label for="interval"
      title="How often the video is checked. Smaller is more detailed but slower.">Check every
      (seconds)</label><input id="interval" type="number" min="5" step="1" value="60"></div>
  <div class="field" id="offsetBox" style="display:none"><label for="offset"
      title="If the second camera started recording later than the first, how many seconds
      later. Negative if it started earlier.">Second camera starts after the first
      (seconds)</label><input id="offset" type="number" step="1" value="0"></div>
  <div class="field"><label for="field">Pitch / field id (optional)</label>
    <input id="field" placeholder="e.g. field_01"></div>
  <div class="field"><label for="day">Date of the video (optional)</label>
    <input id="day" type="date"></div>
</div>
<details><summary>More options</summary>
  <div class="form">
    <div class="field"><label for="camA" title="A camera name with a saved pitch outline">
      Camera name (optional)</label><input id="camA" placeholder="optional"></div>
    <div class="field"><label for="camB" title="The second camera's name">
      Second camera name (optional)</label><input id="camB" placeholder="optional"></div>
    <div class="field"><label for="whole">Use the whole picture</label>
      <input id="whole" type="checkbox"></div>
  </div>
  <p class="sub2" style="margin:4px 0 0">How a slot is judged. The defaults suit most
  clubs; change them only if you want a different line.</p>
  <div class="form" style="margin-top:6px">
    <div class="field"><label for="minSeen"
        title="A slot where less than this was seen is marked please review, never not used">
        Minimum part of a slot that must be seen (%) - default 50</label>
      <input id="minSeen" type="number" min="0" max="100" step="1" value="50"></div>
    <div class="field"><label for="closeLow"
        title="Playing for between these two shares of the time seen is a close call">
        Close call from (% playing) - default 40</label>
      <input id="closeLow" type="number" min="0" max="100" step="1" value="40"></div>
    <div class="field"><label for="closeHigh">Close call to (% playing) - default 60</label>
      <input id="closeHigh" type="number" min="0" max="100" step="1" value="60"></div>
  </div>
</details>
<div class="btnrow"><button class="act" id="go" disabled>Start</button></div>
<p class="note" id="status"></p>
</div>

<section id="prog" class="card" style="margin-top:18px">
  <p id="progText" style="margin:0 0 8px">Uploading...</p>
  <div class="bar"><i id="progBar"></i></div>
</section>

<section id="res">
  <h2>Result</h2>
  <p class="sub2" id="summary"></p>
  <p class="sub2" id="rulesLine"></p>
  <p class="note warn" id="note"></p>
  <div class="tiles" id="tiles"></div>

  <h2>Slots</h2>
  <div class="slots" id="slotbar"></div>
  <div class="legend">
    <span><i class="sw" style="background:var(--play)"></i>used</span>
    <span><i class="sw" style="background:var(--empty-soft)"></i>not used</span>
    <span><i class="sw" style="background:var(--flag-soft)"></i>please review</span>
  </div>

  <h2>Over the whole video</h2>
  <div class="tl" id="timeline"></div>
  <div class="axis" id="axis"></div>
  <div class="legend">
    <span><i class="sw" style="background:var(--play)"></i>playing</span>
    <span><i class="sw" style="background:var(--empty-soft)"></i>empty</span>
    <span><i class="sw" style="background:var(--maint)"></i>maintenance / not playing</span>
    <span><i class="sw" style="background:var(--line)"></i>could not tell</span>
  </div>

  <h2>Slot by slot</h2>
  <div class="scroll"><table>
    <thead><tr><th>Slot</th><th>Main activity</th><th class="num">Playing</th>
      <th class="num">Empty</th><th class="num">Maintenance</th><th class="num">Seen</th>
      <th>Result</th><th>Why</th><th class="bk" id="bkHead">Bookings</th></tr></thead>
    <tbody id="rows"></tbody></table></div>
  <div class="btnrow"><a class="act" id="csv" href="#">Download CSV</a>
    <button class="ghost" id="print">Print / save as PDF</button>
    <button class="ghost" id="again">Test another video</button></div>

  <details class="rejudge"><summary>Judge again with other rules</summary>
    <p class="sub2" style="margin:6px 0 0">Uses the saved results - the video is not needed
    again. A comparison with bookings is redone too.</p>
    <div class="form" style="margin-top:6px">
      <div class="field"><label for="rMinSeen">Minimum part seen (%)</label>
        <input id="rMinSeen" type="number" min="0" max="100" step="1" value="50"></div>
      <div class="field"><label for="rLow">Close call from (%)</label>
        <input id="rLow" type="number" min="0" max="100" step="1" value="40"></div>
      <div class="field"><label for="rHigh">Close call to (%)</label>
        <input id="rHigh" type="number" min="0" max="100" step="1" value="60"></div>
    </div>
    <div class="btnrow" style="margin-top:4px"><button class="act" id="rGo">Judge
      again</button></div>
    <p class="note" id="rStatus"></p>
  </details>

  <div id="cBox">
  <h2>Compare with bookings</h2>
  <p class="sub2">Check the slots against your booking records: a booked slot nobody played
  in may be a no-show, and play in a slot nobody booked may be unbooked use. Slots marked
  "please review" are never counted against anyone.</p>
  <div class="card">
    <div class="form">
      <div class="field"><label for="cField">Pitch / field id</label>
        <input id="cField" placeholder="e.g. field_01"></div>
      <div class="field"><label for="cDay">Date of the video</label>
        <input id="cDay" type="date"></div>
      <div class="field"><label for="cFile"
          title="Columns: field_id, date, start, end, customer_ref, status, entered_by">
          Booking file (CSV)</label><input id="cFile" type="file" accept=".csv,text/csv"></div>
    </div>
    <div class="btnrow">
      <button class="act" id="cUpload">Compare with this file</button>
      <button class="ghost" id="cConfigured">Use our booking system</button>
    </div>
    <p class="note" id="cStatus"></p>
    <div class="tiles" id="cTiles" style="margin-top:14px"></div>
  </div>
  </div>
</section>
"""

_DAY_SCRIPT = r"""
const $=id=>document.getElementById(id);
const files={A:null,B:null};
let RUN=null;
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const RESULT={USED:'used',NOT_USED:'not used',REVIEW:'please review'};
const OUTCOME={booked_and_used:['booked and used','ok'],
  booked_not_used:['booked, not used','bad'],used_not_booked:['used, not booked','bad'],
  not_booked_not_used:['not booked, not used','ok'],
  off_sale_but_used:['closed, but used','bad'],review:['please review','rev']};
const pct=x=>Math.round(100*(x||0))+'%';
const hm=s=>{s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
  return h+':'+String(m).padStart(2,'0');};

const local=()=>document.querySelector('input[name=src]:checked').value==='local';
function ready(){
  const two=local()?!!$('recB').value:!!files.B;
  $('offsetBox').style.display=two?'block':'none';
  $('go').disabled=local()?!$('recA').value:!files.A;
}
function choose(cam,file){
  if(!file)return;files[cam]=file;
  $('name'+cam).textContent=file.name+' ('+(file.size/1048576).toFixed(1)+' MB)';
  $('drop'+cam).classList.add('has');
  ready();
}
let RECS=null;
async function loadRecordings(){
  $('recHint').textContent='Looking for recordings...';
  try{
    const r=await fetch('/api/v1/testing/recordings');
    if(!r.ok)throw new Error(await detail(r));
    RECS=await r.json();
  }catch(err){$('recHint').textContent='Could not list recordings: '+err.message;return;}
  const opt=x=>'<option value="'+esc(x.name)+'">'+esc(x.name)+' ('+
    (x.size_mb>=1024?(x.size_mb/1024).toFixed(1)+' GB':x.size_mb+' MB')+
    (x.duration_s?', '+hm(x.duration_s):'')+')</option>';
  $('recA').innerHTML=RECS.recordings.map(opt).join('');
  $('recB').innerHTML='<option value="">none - one camera</option>'+
    RECS.recordings.map(opt).join('');
  $('recHint').innerHTML=RECS.recordings.length?'Recordings are read from <code>'+
    esc(RECS.folder)+'</code> and are not deleted afterwards.':'No recordings yet. Copy the '+
    'video files into <code>'+esc(RECS.folder)+'</code> on this computer (or a folder '+
    'inside it), then choose this option again.';
  ready();
}
for(const radio of document.querySelectorAll('input[name=src]'))
  radio.addEventListener('change',()=>{
    $('localBox').hidden=!local();$('uploadBox').hidden=local();
    if(local()&&RECS===null)loadRecordings();ready();});
$('recA').addEventListener('change',ready);$('recB').addEventListener('change',ready);
for(const cam of ['A','B']){
  const drop=$('drop'+cam),input=$('file'+cam);
  input.addEventListener('change',()=>choose(cam,input.files[0]));
  drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over');});
  drop.addEventListener('dragleave',()=>drop.classList.remove('over'));
  drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');
    choose(cam,e.dataTransfer.files[0]);});
}

function setStatus(el,msg,bad){$(el).className='note'+(bad?' err':'');$(el).textContent=msg;}
async function detail(r){
  try{const d=await r.json();
    return typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||d);}
  catch(e){return 'The server answered '+r.status+'.';}
}

$('go').addEventListener('click',async()=>{
  if(!$('start').value){setStatus('status','Enter the time the video starts.',true);return;}
  if(local())return startLocal();
  $('go').disabled=true;setStatus('status','');
  $('prog').classList.add('on');$('res').classList.remove('on');
  $('progText').textContent='Uploading the video...';$('progBar').style.width='0';
  const q=new URLSearchParams({start:$('start').value,slot_minutes:$('slot').value||90,
    interval_s:$('interval').value||60,whole_frame:$('whole').checked,
    name:files.A.name});
  if($('first').value)q.set('first_slot',$('first').value);
  if($('field').value.trim())q.set('field_id',$('field').value.trim());
  if($('day').value)q.set('day',$('day').value);
  const rules=ruleParams('minSeen','closeLow','closeHigh');
  if(typeof rules==='string'){setStatus('status',rules,true);ready();
    $('prog').classList.remove('on');return;}
  for(const [k,v] of Object.entries(rules))q.set(k,v);
  let body=files.A;
  if(files.B){
    q.set('size_a',files.A.size);q.set('offset_s',$('offset').value||0);
    q.set('camera_a',$('camA').value);q.set('camera_b',$('camB').value);
    q.set('name_b',files.B.name);body=new Blob([files.A,files.B]);
  }else if($('camA').value){q.set('camera',$('camA').value);}
  try{
    const r=await fetch('/api/v1/testing/day?'+q,{method:'POST',body,
      headers:{'Content-Type':'application/octet-stream'}});
    if(!r.ok)throw new Error(await detail(r));
    const d=await r.json();
    poll(d.job_id);
  }catch(err){$('prog').classList.remove('on');setStatus('status',err.message,true);
    ready();}
});

async function startLocal(){
  const rules=ruleParams('minSeen','closeLow','closeHigh');
  if(typeof rules==='string'){setStatus('status',rules,true);return;}
  const b={file_a:$('recA').value,start:$('start').value,slot_minutes:+($('slot').value||90),
    interval_s:+($('interval').value||60),whole_frame:$('whole').checked,...rules};
  if($('recB').value){b.file_b=$('recB').value;b.offset_s=+($('offset').value||0);
    b.camera_a=$('camA').value;b.camera_b=$('camB').value;}
  else if($('camA').value)b.camera=$('camA').value;
  if($('first').value)b.first_slot=$('first').value;
  if($('field').value.trim())b.field_id=$('field').value.trim();
  if($('day').value)b.day=$('day').value;
  $('go').disabled=true;setStatus('status','');
  $('prog').classList.add('on');$('res').classList.remove('on');
  $('progText').textContent='Starting...';$('progBar').style.width='0';
  try{
    const r=await fetch('/api/v1/testing/day/local',{method:'POST',body:JSON.stringify(b),
      headers:{'Content-Type':'application/json'}});
    if(!r.ok)throw new Error(await detail(r));
    poll((await r.json()).job_id);
  }catch(err){$('prog').classList.remove('on');setStatus('status',err.message,true);ready();}
}

async function poll(id){
  try{
    const r=await fetch('/api/v1/testing/jobs/'+encodeURIComponent(id));
    if(!r.ok)throw new Error(await detail(r));
    const j=await r.json();
    const p=j.progress||{};
    if(j.status==='queued'){$('progText').textContent='Waiting for another test to finish...';}
    else if(j.status==='running'){
      $('progText').textContent=p.total?('Checking the video... '+p.done+' of '+p.total+
        ' moments'):'Preparing the video...';
      $('progBar').style.width=(p.total?100*p.done/p.total:2)+'%';
    }
    if(j.status==='done'){
      $('prog').classList.remove('on');ready();
      history.replaceState(null,'','/testing/day?run='+encodeURIComponent(j.result.id));
      render(j.result);return;
    }
    if(j.status==='failed')throw new Error('The test could not be completed: '+j.error);
    setTimeout(()=>poll(id),1500);
  }catch(err){$('prog').classList.remove('on');setStatus('status',err.message,true);
    ready();}
}

// The judging rules from three percentage inputs, or a sentence saying what is wrong.
function ruleParams(seen,low,high){
  const out={},names={min_seen_pct:seen,close_low_pct:low,close_high_pct:high};
  for(const [k,id] of Object.entries(names)){
    const raw=$(id).value.trim();if(raw==='')continue;
    const v=Number(raw);
    if(!(v>=0&&v<=100))return 'Each rule must be a percentage from 0 to 100.';
    out[k]=v;
  }
  const lo=out.close_low_pct??40,hi=out.close_high_pct??60;
  if(lo>=hi)return 'The close-call band must start below where it ends.';
  return out;
}
const RULES0={min_seen_share:0.5,close_low:0.4,close_high:0.6};
const pc=x=>String(Math.round(1000*x)/10);

function render(run){
  RUN=run;$('res').classList.add('on');
  const rl=run.rules||RULES0;
  $('rulesLine').textContent='Rules: '+(run.rules_text||'at least 50% seen; 40-60% is a '+
    'close call')+'.'+(run.rejudged?' Judged again on '+run.rejudged.at.replace('T',' ')+
    ' (before: '+run.rejudged.previous_rules_text+').':'');
  $('rMinSeen').value=pc(rl.min_seen_share);$('rLow').value=pc(rl.close_low);
  $('rHigh').value=pc(rl.close_high);
  const c=run.counts;
  $('summary').textContent=(run.video_name||'Video')+(run.cameras==2?' + '+run.video_name_b:'')+
    ' - starts '+run.start+', '+run.slot_minutes+'-minute slots'+
    (run.first_slot?' from '+run.first_slot:'')+', checked every '+
    Math.round(run.interval_used_s)+' seconds ('+run.n_samples+' checks, '+
    hm(run.duration_s)+' of video).';
  $('note').textContent=run.note||'';
  $('tiles').innerHTML=[['slots',c.slots],['used',c.USED],['not used',c.NOT_USED],
    ['please review',c.REVIEW,c.REVIEW>0]].map(([k,v,hot])=>'<div class="tile'+
    (hot?' flagged':'')+'"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>').join('');
  $('slotbar').innerHTML=run.slots.map(s=>'<div class="'+s.status+'" title="'+
    esc(s.label+': '+RESULT[s.status]+'. '+s.reason)+'">'+esc(s.start)+
    '<small>'+RESULT[s.status]+'</small></div>').join('');
  const total=Math.max(run.duration_s,1e-6),step=run.interval_used_s;
  $('timeline').innerHTML=run.timeline.map(([t,k])=>'<i class="'+k+'" style="width:'+
    (Math.max(0,Math.min(step,total-t))/total*100)+'%"></i>').join('');
  $('axis').innerHTML='<span>'+run.start+'</span><span>+'+hm(total/2)+'</span><span>+'+
    hm(total)+'</span>';
  $('csv').href='/api/v1/testing/runs/'+encodeURIComponent(run.id)+'.csv';
  $('cField').value=run.field_id||$('cField').value;
  $('cDay').value=run.day||$('cDay').value;
  renderRows(run.comparison);
  $('res').scrollIntoView({behavior:'smooth',block:'start'});
}

function renderRows(cmp){
  const byIdx={};
  if(cmp)for(const k of cmp.checks)byIdx[k.index]=k;
  for(const el of document.querySelectorAll('.bk'))el.classList.toggle('on',!!cmp);
  $('rows').innerHTML=RUN.slots.map(s=>{
    let bk='';
    if(cmp){const k=byIdx[s.index],o=k?OUTCOME[k.outcome]:null;
      bk='<td class="bk on">'+(o?'<span class="pill '+o[1]+'">'+o[0]+'</span><br><span '+
        'class="muted" style="font-size:12px">'+esc(k.message)+'</span>':'')+'</td>';}
    return '<tr><td class="mono">'+esc(s.label)+(s.coverage<1?'<br><span class="muted">'+
      pct(s.coverage)+' in video</span>':'')+'</td><td>'+esc(s.main_label)+'</td>'+
      '<td class="num">'+pct(s.playing_share)+'<br><span class="muted">'+s.playing_min+
      ' min</span></td><td class="num">'+pct(s.empty_share)+'</td><td class="num">'+
      pct(s.maintenance_share)+'</td><td class="num">'+pct(s.seen_share)+'</td><td>'+
      '<span class="pill '+s.status+'">'+RESULT[s.status]+'</span></td><td class="reason">'+
      esc(s.reason)+'</td>'+bk+'</tr>';}).join('');
  if(cmp){
    const n=cmp.counts||{};
    $('cTiles').innerHTML=Object.entries(OUTCOME).filter(([k])=>n[k]).map(([k,[lab,cls]])=>
      '<div class="tile'+(cls==='bad'?' flagged':'')+'"><div class="k">'+lab+
      '</div><div class="v">'+n[k]+'</div></div>').join('');
    setStatus('cStatus','Compared with '+(cmp.source==='upload'?'the uploaded file':
      'your booking system')+' for '+cmp.field_id+' on '+cmp.day+'.'+
      (cmp.n_bookings?'':' No bookings were found for this pitch on this date, so every '+
       'slot counts as not booked.'));
  }
}

async function compare(useConfigured){
  if(!RUN)return;
  const q=new URLSearchParams({use_configured:useConfigured});
  if($('cField').value.trim())q.set('field_id',$('cField').value.trim());
  if($('cDay').value)q.set('day',$('cDay').value);
  let body='';
  if(!useConfigured){
    const f=$('cFile').files[0];
    if(!f){setStatus('cStatus','Choose a booking file first.',true);return;}
    body=await f.text();
  }
  setStatus('cStatus','Comparing...');
  try{
    const r=await fetch('/api/v1/testing/runs/'+encodeURIComponent(RUN.id)+'/compare?'+q,
      {method:'POST',body,headers:{'Content-Type':'text/csv'}});
    if(!r.ok)throw new Error(await detail(r));
    const d=await r.json();RUN.comparison=d;renderRows(d);
  }catch(err){setStatus('cStatus',err.message,true);}
}
$('cUpload').addEventListener('click',()=>compare(false));
$('cConfigured').addEventListener('click',()=>compare(true));
$('print').addEventListener('click',()=>window.print());
$('rGo').addEventListener('click',async()=>{
  if(!RUN)return;
  const rules=ruleParams('rMinSeen','rLow','rHigh');
  if(typeof rules==='string'){setStatus('rStatus',rules,true);return;}
  setStatus('rStatus','Judging again...');
  try{
    const r=await fetch('/api/v1/testing/runs/'+encodeURIComponent(RUN.id)+'/rejudge?'+
      new URLSearchParams(rules),{method:'POST'});
    if(!r.ok)throw new Error(await detail(r));
    const run=await r.json();render(run);
    const n=run.rejudged.slots_changed;
    setStatus('rStatus',n?(n+' slot'+(n==1?'':'s')+' changed.'+(run.comparison?
      ' The comparison with bookings was redone.':'')):'No slot changed.');
  }catch(err){setStatus('rStatus',err.message,true);}
});
$('again').addEventListener('click',()=>{$('res').classList.remove('on');
  history.replaceState(null,'','/testing/day');window.scrollTo(0,0);});

const wanted=new URLSearchParams(location.search).get('run');
if(wanted){fetch('/api/v1/testing/runs/'+encodeURIComponent(wanted)).then(async r=>{
  if(!r.ok)throw new Error(await detail(r));return r.json();}).then(render)
  .catch(err=>setStatus('status','Could not open that test: '+err.message,true));}
"""


@router.get("/testing/day", response_class=HTMLResponse, include_in_schema=False)
def day_page() -> HTMLResponse:
    return HTMLResponse(page("Pitch Occupancy - Booking-day test", "Booking-day test",
                             _DAY_BODY, active="testing", css=_DAY_CSS, script=_DAY_SCRIPT))
