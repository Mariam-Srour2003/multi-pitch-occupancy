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

from pitch_occupancy import timetable
from pitch_occupancy.api import clip_pair, clip_review, jobs
from pitch_occupancy.api.clip_review import MAX_UPLOAD_BYTES
from pitch_occupancy.api.layout import page
from pitch_occupancy.clip_analysis import DEFAULT_WINDOW, analyse_clip
from pitch_occupancy.pair_analysis import analyse_pair
from pitch_occupancy.vision import roi

__all__ = ["router", "MAX_DAY_SAMPLES", "runs_dir"]

router = APIRouter(tags=["testing"])

#: A ceiling on moments per day test. A full day at one a minute is 1,440, so the default
#: interval always fits; a finer interval over a long video is widened, and the run says so.
MAX_DAY_SAMPLES: int = 2000

_RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_CODES = {"C2_ACTIVE_PLAY": "P", "C1_EMPTY": "E", "C3_MAINTENANCE_NON_SPORTING": "M"}


def runs_dir() -> Path:
    """Where finished day tests are saved. Read at call time so a test can point it away."""
    from pitch_occupancy.config import settings

    return Path(settings.data_dir) / "test_runs"


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
    results = timetable.summarise(samples, windows, used_interval)
    counts = Counter(r.status.value for r in results)
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
        "counts": {"slots": len(results), "USED": counts.get("USED", 0),
                   "NOT_USED": counts.get("NOT_USED", 0), "REVIEW": counts.get("REVIEW", 0)},
        "slots": [r.to_dict() for r in results],
        # Per moment, as (seconds, letter): P playing, E empty, M maintenance, U undecided.
        "timeline": [[round(t, 1), _CODES.get(getattr(s, "value", s), "U")]
                     for t, s in samples],
        "comparison": None,
    }


def _make_work(paths: list[Path], meta: dict[str, Any]):
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
            for path in paths:
                path.unlink(missing_ok=True)
        run = _build_run(meta, *out)
        _save_run(run)
        return run

    return work


async def _receive_one(request: Request, path: Path) -> None:
    """Stream the body to ``path``, capped while streaming as `/api/v1/clip/analyse` caps it."""
    size = 0
    with open(path, "wb") as handle:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    413, f"the video exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            handle.write(chunk)
    if size == 0:
        raise HTTPException(422, "no video in the request body")


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
) -> dict[str, str]:
    """Receive one video (or A then B with ``size_a``) and queue the day test. Returns a job."""
    video_start = _parse_hhmm(start, "start")
    first = _parse_hhmm(first_slot, "first_slot") if first_slot.strip() else None
    two = size_a is not None
    paths: list[Path] = []
    for _ in range(2 if two else 1):
        fd, tmp = tempfile.mkstemp(suffix=".upload")
        os.close(fd)
        paths.append(Path(tmp))
    submitted = False
    try:
        if two:
            await clip_pair._receive(request, size_a, (paths[0], paths[1]))
        else:
            await _receive_one(request, paths[0])
        meta = {
            "id": _new_run_id(), "two": two, "video_start": video_start,
            "first_slot": first, "slot_minutes": slot_minutes, "interval_s": interval_s,
            "offset_s": offset_s if two else 0.0, "field_id": field_id.strip(), "day": day,
            "whole_frame": whole_frame,
            "cameras": (camera_a or camera, camera_b) if two else (camera or camera_a, ""),
            "name": name or "video", "name_b": (name_b or "second video") if two else None,
        }
        job_id = jobs.submit("day", _make_work(paths, meta))
        submitted = True
    finally:
        if not submitted:
            for path in paths:
                path.unlink(missing_ok=True)
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
    body = timetable.to_csv(run["slots"], checks)
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
"""

_DAY_BODY = """
<h2>Booking-day test</h2>
<p class="sub2">Upload a video of a pitch, tell us what time it starts and how long your
booking slots are. For every slot you get how long the pitch was played on, empty or used for
maintenance, and whether the slot was used. Nothing you upload is kept - only the results.</p>

<div id="setup">
<div class="pair">
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
    <button class="ghost" id="again">Test another video</button></div>

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

function choose(cam,file){
  if(!file)return;files[cam]=file;
  $('name'+cam).textContent=file.name+' ('+(file.size/1048576).toFixed(1)+' MB)';
  $('drop'+cam).classList.add('has');
  $('offsetBox').style.display=files.B?'block':'none';
  $('go').disabled=!files.A;
}
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
  $('go').disabled=true;setStatus('status','');
  $('prog').classList.add('on');$('res').classList.remove('on');
  $('progText').textContent='Uploading the video...';$('progBar').style.width='0';
  const q=new URLSearchParams({start:$('start').value,slot_minutes:$('slot').value||90,
    interval_s:$('interval').value||60,whole_frame:$('whole').checked,
    name:files.A.name});
  if($('first').value)q.set('first_slot',$('first').value);
  if($('field').value.trim())q.set('field_id',$('field').value.trim());
  if($('day').value)q.set('day',$('day').value);
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
    $('go').disabled=!files.A;}
});

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
      $('prog').classList.remove('on');$('go').disabled=!files.A;
      history.replaceState(null,'','/testing/day?run='+encodeURIComponent(j.result.id));
      render(j.result);return;
    }
    if(j.status==='failed')throw new Error('The test could not be completed: '+j.error);
    setTimeout(()=>poll(id),1500);
  }catch(err){$('prog').classList.remove('on');setStatus('status',err.message,true);
    $('go').disabled=!files.A;}
}

function render(run){
  RUN=run;$('res').classList.add('on');
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
