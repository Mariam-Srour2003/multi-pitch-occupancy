"""SQLite persistence — blueprint §5.2 schema, corrected to a proper cameras table
(exactly 2 cameras per field, one per half)."""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "pitch_monitor.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS fields (
    field_id   TEXT PRIMARY KEY,
    field_name TEXT NOT NULL,
    is_active  BOOLEAN DEFAULT TRUE
);
CREATE TABLE IF NOT EXISTS cameras (
    camera_id  TEXT PRIMARY KEY,
    field_id   TEXT REFERENCES fields(field_id),
    side       TEXT CHECK(side IN ('A','B')),
    rtsp_url   TEXT,
    roi_json   TEXT
);
CREATE TABLE IF NOT EXISTS rental_slots (
    slot_id     TEXT PRIMARY KEY,
    field_id    TEXT REFERENCES fields(field_id),
    day_of_week TEXT,
    start_time  TEXT,
    end_time    TEXT
);
CREATE TABLE IF NOT EXISTS frame_samples (
    sample_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id           TEXT,
    slot_id            TEXT,
    camera_id          TEXT,
    t_s                INTEGER,
    model_key          TEXT,
    predicted_category TEXT NOT NULL,
    confidence_score   REAL NOT NULL,
    fused_category     TEXT,
    image_storage_path TEXT
);
CREATE TABLE IF NOT EXISTS slot_evaluations (
    evaluation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    field_id         TEXT,
    slot_id          TEXT,
    evaluation_date  TEXT,
    model_key        TEXT,
    final_status     TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    decision_reason  TEXT NOT NULL,
    ratio_playing    REAL, ratio_empty REAL, ratio_people REAL, ratio_maintenance REAL,
    evidence_image_1 TEXT, evidence_image_2 TEXT, evidence_image_3 TEXT,
    is_overridden    BOOLEAN DEFAULT FALSE,
    overridden_by    TEXT,
    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con
