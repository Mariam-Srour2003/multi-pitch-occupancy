"""Editing the capture schedule (WP6-T6).

**This is the only write path in the API, and it is worth saying why that is acceptable when
`bookings.py` refuses to have one at all.** The booking sheet is the *facility's* financial
record and this project is an instrument pointed at it; the schedule is this system's own
configuration, and an operator who adds a pitch needs to say so somewhere. The asymmetry is
deliberate rather than an inconsistency.

That still leaves it the most dangerous endpoint here, because **the schedule is the one input
that decides whether an hour is observed at all**. A dropped entry is not a wrong verdict, it
is no verdict and no footage, discovered weeks later when someone disputes a booking, and
nothing recovers it. So three protections, none of them optional:

* **Validated by the reader, not by a second copy of its rules.** A proposed schedule is
  written to a temporary file and put through
  :func:`~pitch_occupancy.scheduler.load_schedule` itself. Re-implementing the checks here
  would give two validators that agree until they do not, and the one that matters is the one
  the scheduler actually runs at 10:00.
* **Nothing is ever partially written.** The new file is written beside the old one and moved
  into place with :func:`os.replace`, which is atomic on both platforms this runs on. A crash
  mid-write leaves the previous schedule intact rather than a truncated JSON file that
  `load_schedule` will refuse at the next slot boundary.
* **The previous version is kept.** Every write copies the current file to
  ``configs/schedule_history/slots_schedule.<timestamp>.json`` first. Disk is cheap and an
  operator who deletes the wrong row at 09:55 has a way back.

:func:`validate` writes nothing and is exposed separately, so the page can tell an operator
what is wrong *before* they commit to it - the same shape as `pitch retention` and
`pitch schedule`, where deciding and doing are separate calls.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pitch_occupancy.scheduler import DEFAULT_SCHEDULE, load_schedule

__all__ = ["router", "history_dir", "validate", "write_schedule"]

router = APIRouter(prefix="/api/v1", tags=["schedule"])

#: How many previous versions to keep. Not unbounded: an editor used daily for a year would
#: otherwise leave 365 files nobody prunes, and the ones worth having are the recent ones.
KEEP_VERSIONS = 20


class ScheduleEntry(BaseModel):
    """One slot, in the shape `load_schedule` reads.

    Loosely typed on purpose - the authoritative validation is `load_schedule`, and a stricter
    model here would reject some inputs with a *different* error than the scheduler would,
    which is how an operator comes to fix a problem the real reader does not have.
    """

    venue_id: str
    start: str
    duration_minutes: int
    cameras: list[str]
    field_id: str = ""
    days: list[str] = Field(default_factory=list)


class ScheduleProposal(BaseModel):
    entries: list[ScheduleEntry]


def history_dir(path: Path | None = None) -> Path:
    return (path or DEFAULT_SCHEDULE).parent / "schedule_history"


def validate(entries: list[dict], path: Path | None = None) -> str | None:
    """Return the reason a proposed schedule would be rejected, or ``None`` if it is fine.

    Runs the real reader over a temporary copy rather than duplicating its rules. That is the
    whole point: the validator and the loader cannot disagree because they are the same code.
    """
    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "slots_schedule.json"
        candidate.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        try:
            load_schedule(candidate)
        except (ValueError, KeyError, TypeError) as exc:
            return str(exc).replace(candidate.name, "the proposed schedule")
        except FileNotFoundError:  # pragma: no cover - just written above
            return "the proposed schedule could not be written for checking"
    return None


def write_schedule(entries: list[dict], path: Path | None = None) -> Path:
    """Validate, back up, then replace the schedule atomically. Returns the backup's path.

    Raises:
        ValueError: if the proposal does not load. Nothing is written in that case - the
            check happens before the backup, so a rejected edit leaves no trace at all.
    """
    target = path or DEFAULT_SCHEDULE
    problem = validate(entries, target)
    if problem is not None:
        raise ValueError(problem)

    backups = history_dir(target)
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = backups / f"{target.stem}.{stamp}.json"
    if target.exists():
        shutil.copy2(target, backup)

    # Written beside the target so the replace is same-filesystem and therefore atomic; a
    # temp file in the system temp directory can be on another volume, where os.replace is
    # not atomic and can leave the schedule half-written.
    staging = target.with_suffix(".json.new")
    staging.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(staging, target)

    for old in sorted(backups.glob(f"{target.stem}.*.json"))[:-KEEP_VERSIONS]:
        old.unlink(missing_ok=True)
    return backup


@router.get("/schedule")
def read_schedule() -> dict:
    """The current schedule as the editor should show it, with whether it currently loads.

    A schedule that no longer loads is still returned, because an operator cannot repair a
    file the editor refuses to display. ``problem`` says what is wrong.
    """
    if not DEFAULT_SCHEDULE.exists():
        return {"entries": [], "problem": f"no schedule file at {DEFAULT_SCHEDULE.name}"}
    try:
        raw = json.loads(DEFAULT_SCHEDULE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"entries": [], "problem": f"the schedule file is not valid JSON: {exc}"}
    entries = raw if isinstance(raw, list) else []
    return {"entries": entries, "problem": validate(entries)}


@router.post("/schedule/validate")
def validate_schedule(proposal: ScheduleProposal) -> dict:
    """Check a proposal. **Writes nothing** - deciding and doing are separate calls."""
    problem = validate([e.model_dump() for e in proposal.entries])
    return {"ok": problem is None, "problem": problem}


@router.put("/schedule")
def replace_schedule(proposal: ScheduleProposal) -> dict:
    """Replace the schedule, after validation, keeping the previous version."""
    try:
        backup = write_schedule([e.model_dump() for e in proposal.entries])
    except ValueError as exc:
        # 422, not 500: the request was understood and refused. An operator needs to see the
        # scheduler's own words, because those are what they have to satisfy.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "ok": True,
        "slots": len(proposal.entries),
        "previous_version": backup.name,
        "note": (
            "The running scheduler reads this file when it next checks what is due; a slot "
            "already under way is not affected."
        ),
    }
