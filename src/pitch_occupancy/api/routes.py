"""Dashboard and audit endpoints (WP6-T6).

The web layer is deliberately thin: it reads what the worker wrote and accepts operator
corrections. **No inference happens in a request handler** - a dashboard restart must never
drop a sample, and a slow model must never make the UI slow.

The operator flow the endpoints are shaped around is the one the blueprint asks for: land
on the anomalies list, open a slot's evidence, agree or correct in one click. Anything that
does not serve that flow is not here.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pitch_occupancy.config import settings
from pitch_occupancy.data.taxonomy import SlotStatus
from pitch_occupancy.db.schema import connect, initialise
from pitch_occupancy.db.store import load_verdict, override_verdict

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


def get_conn() -> sqlite3.Connection:
    conn = connect(settings.db_path)
    initialise(conn)
    try:
        yield conn
    finally:
        conn.close()


# --- models -----------------------------------------------------------------


class SlotSummary(BaseModel):
    slot_id: str
    status: SlotStatus
    reason: str
    play_ratio: float
    empty_ratio: float
    n_samples: int
    mean_confidence: float
    is_overridden: bool
    override_status: SlotStatus | None = None


class Evidence(BaseModel):
    slot_id: str
    status: SlotStatus
    reason: str
    evidence_paths: list[str]
    samples: list[dict]


class OverrideRequest(BaseModel):
    status: SlotStatus
    operator: str = Field(min_length=1, description="Who is making the correction.")
    note: str | None = None


class Meters(BaseModel):
    slots_evaluated: int
    used: int
    notused: int
    review: int
    overridden: int
    open_anomalies: int
    review_rate: float


# --- endpoints --------------------------------------------------------------


@router.get("/meters", response_model=Meters)
def meters(conn: sqlite3.Connection = Depends(get_conn)) -> Meters:
    """Global counters for the top of the dashboard."""
    row = conn.execute(
        """SELECT COUNT(*) total,
                  SUM(status = 'USED')    used,
                  SUM(status = 'NOTUSED') notused,
                  SUM(status = 'REVIEW')  review,
                  SUM(is_overridden)      overridden
           FROM slot_evaluations"""
    ).fetchone()
    open_anom = conn.execute(
        """SELECT COUNT(*) c FROM reconciliations
           WHERE resolved_at IS NULL AND anomaly NOT IN ('CONSISTENT', 'NEEDS_REVIEW')"""
    ).fetchone()["c"]
    total = row["total"] or 0
    return Meters(
        slots_evaluated=total,
        used=row["used"] or 0,
        notused=row["notused"] or 0,
        review=row["review"] or 0,
        overridden=row["overridden"] or 0,
        open_anomalies=open_anom,
        # the number that decides how much manual work the facility is buying
        review_rate=round((row["review"] or 0) / total, 4) if total else 0.0,
    )


@router.get("/slots", response_model=list[SlotSummary])
def list_slots(
    status: SlotStatus | None = Query(None, description="Filter by verdict."),
    limit: int = Query(50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[SlotSummary]:
    sql = "SELECT * FROM slot_evaluations"
    params: list[object] = []
    if status is not None:
        sql += " WHERE status = ?"
        params.append(status.value)
    sql += " ORDER BY evaluated_at DESC LIMIT ?"
    params.append(limit)
    return [
        SlotSummary(
            slot_id=r["slot_id"], status=SlotStatus(r["status"]), reason=r["reason"],
            play_ratio=r["play_ratio"], empty_ratio=r["empty_ratio"],
            n_samples=r["n_samples"], mean_confidence=r["mean_confidence"],
            is_overridden=bool(r["is_overridden"]),
            override_status=SlotStatus(r["override_status"]) if r["override_status"] else None,
        )
        for r in conn.execute(sql, params).fetchall()
    ]


@router.get("/slots/{slot_id}/evidence", response_model=Evidence)
def slot_evidence(slot_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> Evidence:
    """Everything an operator needs to agree or disagree with one verdict."""
    row = load_verdict(conn, slot_id)
    if row is None:
        raise HTTPException(404, f"no evaluation for slot {slot_id!r}")
    samples = conn.execute(
        """SELECT camera_id, minute_index, predicted, confidence, image_path
           FROM frame_samples WHERE slot_id = ? ORDER BY minute_index, camera_id""",
        (slot_id,),
    ).fetchall()
    return Evidence(
        slot_id=slot_id,
        status=SlotStatus(row["status"]),
        reason=row["reason"],
        evidence_paths=json.loads(row["evidence_paths"] or "[]"),
        samples=[dict(s) for s in samples],
    )


#: Roots an evidence image may live under. Checked after resolving, so a symlink or a
#: `..` inside a stored path cannot reach outside them.
def _evidence_roots() -> tuple[Path, ...]:
    return (
        (settings.interim_dir / "evidence").resolve(),
        settings.dataset_dir.resolve(),
    )


@router.get("/slots/{slot_id}/evidence/{index}", include_in_schema=False)
def evidence_image(
    slot_id: str, index: int, conn: sqlite3.Connection = Depends(get_conn)
) -> FileResponse:
    """One evidence image, addressed by its position rather than by its path.

    The URL carries a slot id and an integer and never a filename, so path traversal is not
    something to sanitise here - it is not expressible. The stored path is still resolved and
    confined to the evidence roots afterwards, because the database is not a trust boundary
    either: a path could have been written there by an older version or an edited row.
    """
    row = load_verdict(conn, slot_id)
    if row is None:
        raise HTTPException(404, f"no evaluation for slot {slot_id!r}")
    paths = json.loads(row["evidence_paths"] or "[]")
    if not 0 <= index < len(paths):
        raise HTTPException(404, f"slot {slot_id!r} has {len(paths)} evidence image(s)")

    path = Path(paths[index])
    if not path.is_absolute():
        path = (settings.dataset_dir / path)
    path = path.resolve()
    if not any(path.is_relative_to(root) for root in _evidence_roots()):
        raise HTTPException(404, "evidence image is outside the permitted directories")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not path.is_file():
        # Retention deletes evidence on a schedule, so a missing file is normal operation
        # rather than a fault - and it must read as "gone", never as a placeholder image.
        raise HTTPException(404, "evidence image is not on disk (retention may have removed it)")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/slots/{slot_id}/override", response_model=SlotSummary)
def override(
    slot_id: str, body: OverrideRequest, conn: sqlite3.Connection = Depends(get_conn)
) -> SlotSummary:
    """Record a human correction. The model's original verdict is retained."""
    try:
        override_verdict(conn, slot_id, body.status, operator=body.operator, note=body.note)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    r = load_verdict(conn, slot_id)
    assert r is not None
    return SlotSummary(
        slot_id=r["slot_id"], status=SlotStatus(r["status"]), reason=r["reason"],
        play_ratio=r["play_ratio"], empty_ratio=r["empty_ratio"],
        n_samples=r["n_samples"], mean_confidence=r["mean_confidence"],
        is_overridden=True, override_status=SlotStatus(r["override_status"]),
    )


@router.get("/anomalies")
def anomalies(
    include_resolved: bool = Query(False),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[dict]:
    """Reconciliation findings, most serious first.

    CONSISTENT and NEEDS_REVIEW are excluded: this list is for things that need acting on,
    and padding it with agreements is how an operator learns to stop reading it.
    """
    sql = """SELECT * FROM reconciliations
             WHERE anomaly NOT IN ('CONSISTENT', 'NEEDS_REVIEW')"""
    if not include_resolved:
        sql += " AND resolved_at IS NULL"
    sql += """ ORDER BY CASE severity
                 WHEN 'serious' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, created_at DESC"""
    return [dict(r) for r in conn.execute(sql).fetchall()]


@router.get("/fields/{field_id}/day/{day}")
def field_day(
    field_id: str, day: date, conn: sqlite3.Connection = Depends(get_conn)
) -> list[dict]:
    """One field's slots for one day - the dashboard's field-by-slot matrix row."""
    return [
        dict(r)
        for r in conn.execute(
            """SELECT rs.slot_id, rs.start_time, rs.end_time,
                      se.status, se.mean_confidence, se.is_overridden, se.override_status
               FROM rental_slots rs
               LEFT JOIN slot_evaluations se ON se.slot_id = rs.slot_id
               WHERE rs.field_id = ? AND rs.slot_date = ?
               ORDER BY rs.start_time""",
            (field_id, day.isoformat()),
        ).fetchall()
    ]
