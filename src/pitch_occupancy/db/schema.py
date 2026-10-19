"""SQLite schema and access.

Stdlib ``sqlite3`` rather than an ORM: the schema is eight tables, the queries are
straightforward, and the target is a Mini-PC that should not carry a dependency it does
not need.

Two structural decisions shape it:

* **``cameras`` is its own table, two rows per field.** An earlier design had
  ``primary_camera_rtsp`` / ``secondary_camera_rtsp`` columns on the field, which cannot
  express per-camera ROI polygons or a third camera, and makes "which half did this frame
  come from" unanswerable.
* **``frame_samples`` stores the per-camera prediction, and ``slot_evaluations`` the fused
  verdict.** Keeping raw observations alongside the decision is what makes an evidence
  trail auditable - an operator disputing a verdict can see the minutes behind it.

Override fields on ``slot_evaluations`` are the human-in-the-loop record. They double as
free ground truth: a corrected slot is a labelled slot, which feeds the periodic head
re-fit.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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

-- one row per camera per automatic check (`playgrounds.check`): were there people, and how many.
-- Separate from the slot tables on purpose: a check belongs to a camera and a moment, not to
-- a booked slot, so it needs no schedule and no field. `playground_check_id` ties the row to
-- the playground check it was part of; the timings feed the computer-speed estimate.
CREATE TABLE IF NOT EXISTS camera_checks (
    check_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    venue_id      TEXT NOT NULL,
    camera_id     TEXT NOT NULL,
    checked_at    TEXT NOT NULL,     -- local ISO timestamp with offset
    readable      INTEGER NOT NULL,  -- 0 = the camera gave no picture
    people        INTEGER,           -- NULL = not counted (unreadable or no detector)
    state         TEXT,              -- MinuteState value, NULL when not decided
    label         TEXT,              -- the plain-words verdict the pages show
    confidence    REAL,
    picture_path  TEXT,              -- only when saving pictures is switched on
    problem       TEXT
);

CREATE INDEX IF NOT EXISTS idx_checks_time ON camera_checks(checked_at);
CREATE INDEX IF NOT EXISTS idx_checks_camera ON camera_checks(venue_id, camera_id);

-- one row per playground per check: its cameras combined into one count (added when they
-- watch different parts, the largest when they watch the same area - see `playgrounds`).
CREATE TABLE IF NOT EXISTS playground_checks (
    check_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    playground_id  TEXT NOT NULL,
    checked_at     TEXT NOT NULL,     -- local ISO timestamp with offset
    people         INTEGER,           -- combined count; NULL when no camera gave a picture
    playing        INTEGER,           -- 1 when people >= the rule's play_min
    cameras_read   INTEGER NOT NULL,
    cameras_total  INTEGER NOT NULL,
    counting       TEXT NOT NULL,     -- 'add', 'largest', 'average' or 'vote'
    duration_ms    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pgchecks ON playground_checks(playground_id, checked_at);

-- the free vision-model second opinion (`assist/`): every question asked, and its answer
CREATE TABLE IF NOT EXISTS assist_opinions (
    opinion_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    check_id       INTEGER,           -- playground_checks.check_id
    playground_id  TEXT NOT NULL,
    asked_at       TEXT NOT NULL,
    doubtful       INTEGER NOT NULL,  -- 1 = people seen but fewer than a game needs
    provider       TEXT,              -- NULL when no provider answered
    model          TEXT,
    state          TEXT,              -- 'playing', 'other' or 'empty'
    people         INTEGER,
    confidence     REAL,
    reason         TEXT,
    tokens         INTEGER,
    latency_ms     INTEGER,
    upgraded       INTEGER NOT NULL,  -- 1 = the check was changed to playing
    problem        TEXT
);

CREATE INDEX IF NOT EXISTS idx_assist_pg ON assist_opinions(playground_id, asked_at);

-- each free provider's use per UTC day, so a restart cannot reset the budget
CREATE TABLE IF NOT EXISTS assist_usage (
    provider        TEXT NOT NULL,
    day             TEXT NOT NULL,
    requests        INTEGER NOT NULL DEFAULT 0,
    tokens          INTEGER NOT NULL DEFAULT 0,
    failures        INTEGER NOT NULL DEFAULT 0,
    cooldown_until  TEXT,
    PRIMARY KEY (provider, day)
);
"""

#: Columns added to existing tables after their first release. `CREATE TABLE IF NOT EXISTS`
#: leaves an existing table as it was, so a database created before these columns existed
#: gets them here - without this, every older install would fail on the first insert.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("camera_checks", "playground_check_id", "INTEGER"),
    ("camera_checks", "grab_ms", "INTEGER"),
    ("camera_checks", "detect_ms", "INTEGER"),
    # What the detector said before a second opinion upgraded the check, and who decided.
    ("playground_checks", "detector_playing", "INTEGER"),
    ("playground_checks", "playing_source", "TEXT"),
)


def connect(path: Path | str, *, same_thread: bool = True) -> sqlite3.Connection:
    """Open a connection with foreign keys on and rows accessible by name.

    Args:
        same_thread: sqlite3 binds a connection to its creating thread by default. The API
            opens one connection per request, inside the thread that serves it, so the
            default holds there. Pass ``False`` only to share a connection deliberately -
            a test client running the app on another thread, for instance.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Wait up to 30 s for another writer instead of the default 5 s: playgrounds checked in
    # parallel finish together, and on a slow night (2026-10-19) one of them hit
    # "database is locked" and lost its check.
    conn = sqlite3.connect(str(p), check_same_thread=same_thread, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Readers no longer wait for the writer: the pages poll every few seconds while up to
    # eight playground threads and the worker write. Kept by the file once set.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def open_db(path: Path | str | None = None) -> sqlite3.Connection:
    """The database (``settings.db_path`` by default) read-write, schema brought up to date.

    The caller closes it.
    """
    if path is None:
        from pitch_occupancy.config import settings

        path = settings.db_path
    conn = connect(path)
    initialise(conn)
    return conn


def open_readonly(path: Path | str | None = None) -> sqlite3.Connection:
    """The database (``settings.db_path`` by default) opened so it cannot be written:
    ``mode=ro`` plus ``query_only``, rows by name. The caller closes it.

    Never creates the file or its directory, and sets no journal pragma - changing the
    journal mode is a write. Raises `FileNotFoundError` naming the file if it is missing
    (``mode=ro`` alone would only say "unable to open database file").
    """
    if path is None:
        from pitch_occupancy.config import settings

        path = settings.db_path
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no SQLite database at {p}")
    conn = sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


#: Database files this process has already brought up to date. Every request used to run the
#: whole schema script again; once per file per process is enough.
_initialised: set[str] = set()
_initialised_lock = threading.Lock()


def _database_file(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    return row[2] if row is not None else ""


def initialise(conn: sqlite3.Connection) -> None:
    """Create the schema. Idempotent - safe on an existing database, and cheap the second
    time in a process (an in-memory database is always initialised)."""
    name = _database_file(conn)
    # The file may have been replaced since (a restore, a test's fresh database), so a cheap
    # look for the newest table confirms it before skipping.
    if name and name in _initialised and conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'assist_usage'"
    ).fetchone():
        return
    conn.executescript(SCHEMA)
    for table, column, kind in _ADDED_COLUMNS:
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    conn.commit()
    if name:
        with _initialised_lock:
            _initialised.add(name)


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
