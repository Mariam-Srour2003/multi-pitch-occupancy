"""Reading and writing slot evidence (WP6).

Thin functions over :mod:`pitch_occupancy.db.schema`. The one rule worth stating: a slot
verdict and the samples behind it are written together or not at all, so no verdict can
exist without the evidence that produced it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from pitch_occupancy.data.taxonomy import SlotStatus
from pitch_occupancy.db.schema import transaction
from pitch_occupancy.slots.aggregate import SlotVerdict

__all__ = ["Sample", "record_slot", "load_verdict", "override_verdict", "overridden_slots"]


@dataclass(frozen=True, slots=True)
class Sample:
    camera_id: str
    minute_index: int
    predicted: str
    confidence: float
    captured_at: str
    image_path: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_slot(
    conn: sqlite3.Connection,
    slot_id: str,
    verdict: SlotVerdict,
    samples: Sequence[Sample],
    *,
    model_key: str,
    evidence_paths: Sequence[str] = (),
) -> None:
    """Persist a verdict together with every sample behind it, atomically."""
    with transaction(conn):
        conn.executemany(
            """INSERT OR REPLACE INTO frame_samples
               (slot_id, camera_id, captured_at, minute_index, predicted, confidence, image_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (slot_id, s.camera_id, s.captured_at, s.minute_index,
                 s.predicted, s.confidence, s.image_path)
                for s in samples
            ],
        )
        conn.execute(
            """INSERT OR REPLACE INTO slot_evaluations
               (slot_id, status, reason, play_ratio, empty_ratio, maintenance_ratio,
                n_samples, mean_confidence, evidence_paths, evaluated_at, model_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                slot_id, verdict.status.value, verdict.reason, verdict.play_ratio,
                verdict.empty_ratio, verdict.maintenance_ratio, verdict.n_samples,
                verdict.mean_confidence, json.dumps(list(evidence_paths)), _now(), model_key,
            ),
        )


def load_verdict(conn: sqlite3.Connection, slot_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM slot_evaluations WHERE slot_id = ?", (slot_id,)
    ).fetchone()


def override_verdict(
    conn: sqlite3.Connection,
    slot_id: str,
    status: SlotStatus,
    *,
    operator: str,
    note: str | None = None,
) -> None:
    """Record a human correction.

    The original verdict is never overwritten - both are kept, because the disagreement is
    the signal. Overridden slots are the cheapest source of real slot labels the project
    has (WP6-T7).
    """
    if load_verdict(conn, slot_id) is None:
        raise KeyError(f"no evaluation exists for slot {slot_id!r} to override")
    with transaction(conn):
        conn.execute(
            """UPDATE slot_evaluations
               SET is_overridden = 1, override_status = ?, override_by = ?,
                   override_at = ?, override_note = ?
               WHERE slot_id = ?""",
            (status.value, operator, _now(), note, slot_id),
        )


def overridden_slots(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Slots a human corrected - i.e. slots with a trustworthy label attached."""
    return conn.execute(
        """SELECT slot_id, status, override_status, override_by, override_at, override_note
           FROM slot_evaluations WHERE is_overridden = 1 ORDER BY override_at"""
    ).fetchall()
