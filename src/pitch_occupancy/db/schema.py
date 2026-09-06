"""SQLite schema and access (WP6).

Stdlib ``sqlite3`` rather than an ORM: the schema is eight tables, the queries are
straightforward, and the target is a Mini-PC that should not carry a dependency it does
not need.

Two structural decisions carry over from the blueprint correction:

* **``cameras`` is its own table, two rows per field.** The original specification had
  ``primary_camera_rtsp`` / ``secondary_camera_rtsp`` columns on the field, which cannot
  express per-camera ROI polygons or a third camera, and makes "which half did this frame
  come from" unanswerable.
* **``frame_samples`` stores the per-camera prediction, and ``slot_evaluations`` the fused
  verdict.** Keeping raw observations alongside the decision is what makes an evidence
  trail auditable - an operator disputing a verdict can see the minutes behind it.

Override fields on ``slot_evaluations`` are the human-in-the-loop record. They double as
free ground truth: a corrected slot is a labelled slot, which feeds the periodic head
re-fit (WP6-T7) and, eventually, the real slots STAN needs.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["SCHEMA", "connect", "initialise", "transaction"]

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS venues (
    venue_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fields (
    field_id    TEXT PRIMARY KEY,
    venue_id    TEXT NOT NULL REFERENCES venues(venue_id),
    name        TEXT NOT NULL
);

-- exactly two rows per field in the deployed system, one per half
CREATE TABLE IF NOT EXISTS cameras (
    camera_id    TEXT PRIMARY KEY,
    field_id     TEXT NOT NULL REFERENCES fields(field_id),
    side         TEXT NOT NULL CHECK (side IN ('A', 'B')),
    rtsp_url     TEXT,
    roi_polygon  TEXT,               -- JSON, normalised 0-1 coordinates
    UNIQUE (field_id, side)
);

CREATE TABLE IF NOT EXISTS rental_slots (
    slot_id     TEXT PRIMARY KEY,
    field_id    TEXT NOT NULL REFERENCES fields(field_id),
    slot_date   TEXT NOT NULL,       -- ISO date
    start_time  TEXT NOT NULL,       -- HH:MM
    end_time    TEXT NOT NULL,
    UNIQUE (field_id, slot_date, start_time)
);

-- one row per camera per sampled minute: the raw observation behind a verdict
CREATE TABLE IF NOT EXISTS frame_samples (
    sample_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id       TEXT NOT NULL REFERENCES rental_slots(slot_id),
    camera_id     TEXT NOT NULL REFERENCES cameras(camera_id),
    captured_at   TEXT NOT NULL,     -- ISO timestamp
    minute_index  INTEGER NOT NULL,
    predicted     TEXT NOT NULL,     -- Class3 value
    confidence    REAL NOT NULL,
    image_path    TEXT,
    UNIQUE (slot_id, camera_id, minute_index)
);

CREATE INDEX IF NOT EXISTS idx_samples_slot ON frame_samples(slot_id);

CREATE TABLE IF NOT EXISTS slot_evaluations (
    slot_id            TEXT PRIMARY KEY REFERENCES rental_slots(slot_id),
    status             TEXT NOT NULL CHECK (status IN ('USED', 'NOTUSED', 'REVIEW')),
    reason             TEXT NOT NULL,
    play_ratio         REAL NOT NULL,
    empty_ratio        REAL NOT NULL,
    maintenance_ratio  REAL NOT NULL,
    n_samples          INTEGER NOT NULL,
    mean_confidence    REAL NOT NULL,
    evidence_paths     TEXT,          -- JSON list
    evaluated_at       TEXT NOT NULL,
    model_key          TEXT NOT NULL,
    -- human-in-the-loop; a correction here is also a free slot label
    is_overridden      INTEGER NOT NULL DEFAULT 0,
    override_status    TEXT CHECK (override_status IN ('USED', 'NOTUSED', 'REVIEW')),
    override_by        TEXT,
    override_at        TEXT,
    override_note      TEXT
);

CREATE TABLE IF NOT EXISTS bookings (
    booking_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id              TEXT NOT NULL REFERENCES fields(field_id),
    slot_date             TEXT NOT NULL,
    start_time            TEXT NOT NULL,
    customer_ref          TEXT,
    booked                INTEGER NOT NULL DEFAULT 1,
    staff_recorded_used   INTEGER,    -- NULL == no record exists
    maintenance_window    INTEGER NOT NULL DEFAULT 0,
    entered_by            TEXT,       -- retained for the operator; never aggregated
    source                TEXT NOT NULL DEFAULT 'csv',
    UNIQUE (field_id, slot_date, start_time)
);

CREATE TABLE IF NOT EXISTS reconciliations (
    reconciliation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_id            TEXT NOT NULL REFERENCES rental_slots(slot_id),
    anomaly            TEXT NOT NULL,
    severity           TEXT NOT NULL,
    explanation        TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    resolved_at        TEXT,
    resolved_by        TEXT,
    UNIQUE (slot_id)
);

CREATE INDEX IF NOT EXISTS idx_recon_anomaly ON reconciliations(anomaly);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with foreign keys on and rows accessible by name."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialise(conn: sqlite3.Connection) -> None:
    """Create the schema. Idempotent - safe on an existing database."""
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on any exception.

    A slot evaluation writes several rows; a partial write would leave a verdict with no
    evidence behind it, which is worse than no verdict at all.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
