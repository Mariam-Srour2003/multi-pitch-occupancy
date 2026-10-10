"""Booking sources: CSV, SQLite and other databases behind one read-only interface.

What matters here is that a client's own table can be mapped without changing it, that every
source only reads, and that a password in the config never reaches a page.
"""

from __future__ import annotations

import builtins
import json
import sqlite3
from datetime import date, time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import booking_sources as bs
from pitch_occupancy.api import bookings_page
from pitch_occupancy.api.routes import get_conn
from pitch_occupancy.bookings import EXAMPLE_PATH, BookingSource, read_bookings
from pitch_occupancy.db.schema import initialise

# --- CSV --------------------------------------------------------------------------------


def test_csv_source_reads_the_example_like_read_bookings():
    src = bs.source_from_config({"type": "csv", "path": str(EXAMPLE_PATH)})
    assert isinstance(src, BookingSource)
    assert src.read() == read_bookings(EXAMPLE_PATH)
    assert src.source_tag == "example"


def test_csv_source_with_column_and_status_mapping(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text(
        "Pitch,Day,From,To,Ref,State,Clerk\n"
        "P1,11/07/2026,10.00,11.00,R1,Booked,x\n"
        "P1,11/07/2026,11.00,12.00,R2,DNA,y\n", encoding="utf-8")
    src = bs.source_from_config({
        "type": "csv", "path": str(path),
        "columns": {"field_id": "Pitch", "date": "Day", "start": "From", "end": "To",
                    "customer_ref": "Ref", "status": "State", "entered_by": "Clerk"},
        "status_map": {"booked": "confirmed", "DNA": "no_show"},
        "date_format": "%d/%m/%Y", "time_format": "%H.%M",
    })
    recs = src.read()
    assert [(r.field_id, r.date, r.start, r.end, r.status) for r in recs] == [
        ("P1", date(2026, 7, 11), time(10), time(11), "confirmed"),
        ("P1", date(2026, 7, 11), time(11), time(12), "no_show"),
    ]
    assert recs[0].source == "export"


def test_csv_datetime_columns_are_split_into_date_and_time(tmp_path):
    path = tmp_path / "b.csv"
    path.write_text("pitch,starts_at,ends_at,status\n"
                    "f1,2026-07-11 18:00:00,2026-07-11T19:30,confirmed\n", encoding="utf-8")
    src = bs.source_from_config({"type": "csv", "path": str(path), "columns": {
        "field_id": "pitch", "start": "starts_at", "end": "ends_at"}})
    (rec,) = src.read()
    assert (rec.date, rec.start, rec.end) == (date(2026, 7, 11), time(18), time(19, 30))
    assert rec.customer_ref == "" and rec.entered_by == ""


def test_missing_required_column_names_the_columns_found(tmp_path):
    path = tmp_path / "b.csv"
    path.write_text("pitch,when\nf1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no column for.*Columns found: pitch, when"):
        bs.source_from_config({"type": "csv", "path": str(path)}).rows()


def test_strict_read_refuses_but_rows_keep_the_bad_value(tmp_path):
    path = tmp_path / "b.csv"
    path.write_text("field_id,date,start,end,customer_ref,status,entered_by\n"
                    "f1,2026-07-11,10:00,11:00,,booked,\n", encoding="utf-8")
    src = bs.source_from_config({"type": "csv", "path": str(path)})
    with pytest.raises(ValueError, match="row 2: status 'booked'"):
        src.read()
    assert src.rows()[0]["status"] == "booked"


def test_parse_csv_text_uses_the_strict_reader():
    text = EXAMPLE_PATH.read_text(encoding="utf-8")
    recs = bs.parse_csv_text(text)
    assert len(recs) == len(read_bookings(EXAMPLE_PATH))
    assert {r.source for r in recs} == {"upload"}
    with pytest.raises(ValueError, match="the uploaded CSV is missing column"):
        bs.parse_csv_text("field_id,date\nf1,2026-07-11\n")
    with pytest.raises(ValueError, match="row 2: end"):
        bs.parse_csv_text("field_id,date,start,end,customer_ref,status,entered_by\n"
                          "f1,2026-07-11,11:00,10:00,,confirmed,\n")


# --- SQLite -----------------------------------------------------------------------------


@pytest.fixture()
def client_db(tmp_path):
    """A booking system's own database, with its own names for everything."""
    path = tmp_path / "their_app.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE reservations (court TEXT, starts_at TEXT, ends_at TEXT, "
                 "ref TEXT, state TEXT, clerk TEXT)")
    conn.executemany("INSERT INTO reservations VALUES (?,?,?,?,?,?)", [
        ("field_01", "2026-07-11 10:00:00", "2026-07-11 11:00:00", "R1", "Paid", "a"),
        ("field_01", "2026-07-11 20:30:00", "2026-07-11 21:30:00", "R2", "Cancelled", "b"),
        ("field_02", "2026-07-12 09:00:00", "2026-07-12 10:00:00", None, "Closed", "c"),
    ])
    conn.commit()
    conn.close()
    return path


SQLITE_CONFIG = {
    "columns": {"field_id": "court", "start": "starts_at", "end": "ends_at",
                "customer_ref": "ref", "status": "state", "entered_by": "clerk"},
    "status_map": {"Paid": "confirmed", "Cancelled": "cancelled", "Closed": "blocked"},
}


def test_sqlite_source_maps_a_foreign_table(client_db):
    src = bs.source_from_config({"type": "sqlite", "path": str(client_db),
                                 "query": "SELECT * FROM reservations ORDER BY starts_at",
                                 **SQLITE_CONFIG})
    recs = src.read()
    assert [(r.field_id, r.date.isoformat(), f"{r.start:%H:%M}", r.status) for r in recs] == [
        ("field_01", "2026-07-11", "10:00", "confirmed"),
        ("field_01", "2026-07-11", "20:30", "cancelled"),
        ("field_02", "2026-07-12", "09:00", "blocked"),
    ]
    assert recs[2].customer_ref == ""
    assert src.source_tag == "their_app"
    assert [r["row"] for r in src.rows()] == [1, 2, 3]


def test_sqlite_connection_is_read_only(client_db):
    src = bs.source_from_config({"type": "sqlite", "path": str(client_db),
                                 "query": "SELECT * FROM reservations", **SQLITE_CONFIG})
    conn = src._open()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM reservations")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("PRAGMA query_only = OFF")
            conn.execute("INSERT INTO reservations (court) VALUES ('x')")
    finally:
        conn.close()
    check = sqlite3.connect(client_db)
    assert check.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 3
    check.close()


def test_sqlite_missing_file_is_a_clear_error(tmp_path):
    src = bs.source_from_config({"type": "sqlite", "path": str(tmp_path / "nope.db"),
                                 "query": "SELECT 1"})
    with pytest.raises(FileNotFoundError, match="no SQLite database"):
        src.rows()
    assert not (tmp_path / "nope.db").exists()


# --- the SELECT-only guard --------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "SELECT * FROM bookings",
    "  -- comment\n select * from bookings;",
    "/* lead */ WITH b AS (SELECT * FROM bookings) SELECT * FROM b",
    "SELECT * FROM bookings WHERE note = 'please update; then drop'",
    'SELECT "delete", updated_at, REPLACE(ref, \'-\', \'\') FROM bookings',
])
def test_guard_accepts_reads(query):
    assert bs.check_read_only(query) == query


@pytest.mark.parametrize("query", [
    "UPDATE bookings SET status = 'cancelled'",
    "DELETE FROM bookings",
    "SELECT 1; DROP TABLE bookings",
    "select 1;\n-- harmless?\ndelete from bookings",
    "WITH gone AS (DELETE FROM bookings RETURNING *) SELECT * FROM gone",
    "SELECT * INTO backup FROM bookings",
    "INSERT INTO bookings VALUES (1)",
    "PRAGMA writable_schema = 1",
    "",
])
def test_guard_rejects_writes(query):
    with pytest.raises(ValueError):
        bs.check_read_only(query)


def test_config_with_a_write_query_is_refused_at_load(tmp_path):
    cfg = tmp_path / "bookings_source.json"
    cfg.write_text(json.dumps({"type": "sqlite", "path": "x.db",
                               "query": "SELECT 1; DROP TABLE bookings"}), encoding="utf-8")
    with pytest.raises(ValueError, match="more than one statement"):
        bs.load_source(cfg)


# --- config loading ---------------------------------------------------------------------


def test_missing_config_means_no_source(tmp_path):
    assert bs.load_source(tmp_path / "bookings_source.json") is None


@pytest.mark.parametrize("config,match", [
    ([], "JSON object"),
    ({"type": "excel"}, "must be one of"),
    ({"type": "csv"}, "needs \"path\""),
    ({"type": "csv", "path": "a.csv", "colums": {}}, "unknown setting"),
    ({"type": "csv", "path": "a.csv", "columns": {"pitch": "x"}}, "unknown name"),
    ({"type": "csv", "path": "a.csv", "status_map": {"Paid": "sold"}}, "unknown status"),
    ({"type": "sql", "url": "postgresql://h/db"}, "needs \"query\""),
])
def test_malformed_config_is_a_readable_error(tmp_path, config, match):
    cfg = tmp_path / "bookings_source.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        bs.load_source(cfg)


def test_bad_json_is_a_readable_error(tmp_path):
    cfg = tmp_path / "bookings_source.json"
    cfg.write_text("{type: csv}", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        bs.load_source(cfg)


def test_the_committed_example_config_loads():
    src = bs.load_source(bs.EXAMPLE_SOURCE_CONFIG)
    assert src is not None and len(src.read()) == len(read_bookings())


# --- masking ----------------------------------------------------------------------------


def test_describe_masks_passwords():
    cfg = {"type": "sql", "_comment": "x",
           "url": "postgresql://reader:s3cr%40t@db.example:5432/bookings",
           "query": "SELECT 1"}
    shown = bs.describe(cfg)
    assert shown["url"] == "postgresql://reader:***@db.example:5432/bookings"
    assert "_comment" not in shown
    assert "s3cr" not in json.dumps(shown)
    assert bs.mask_url("mssql+pyodbc:///?odbc_connect=DSN=x;UID=u;PWD=hunter2") \
        == "mssql+pyodbc:///?odbc_connect=DSN=x;UID=u;PWD=***"
    assert bs.mask_url("mysql+pymysql://u@h/db?password=pw&x=1") \
        == "mysql+pymysql://u@h/db?password=***&x=1"


# --- other databases --------------------------------------------------------------------


def test_sql_without_sqlalchemy_says_what_to_install(monkeypatch):
    real_import = builtins.__import__

    def no_sqlalchemy(name, *args, **kwargs):
        if name == "sqlalchemy" or name.startswith("sqlalchemy."):
            raise ImportError("No module named 'sqlalchemy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sqlalchemy)
    src = bs.source_from_config({"type": "sql", "url": "postgresql://u:p@h/db",
                                 "query": "SELECT * FROM bookings"})
    with pytest.raises(RuntimeError, match="uv add sqlalchemy psycopg2-binary"):
        src.read()


def test_sql_guard_runs_before_any_import():
    src = bs.SqlSource({"type": "sql", "url": "postgresql://u:p@h/db",
                        "query": "DELETE FROM bookings"})
    with pytest.raises(ValueError, match="SELECT or WITH"):
        src.rows()


# --- the API ----------------------------------------------------------------------------


@pytest.fixture()
def api(tmp_path, monkeypatch, client_db):
    cfg = tmp_path / "bookings_source.json"
    monkeypatch.setattr(bookings_page, "SOURCE_CONFIG", cfg)
    monkeypatch.setattr(bookings_page, "_schedule", lambda: None)

    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    initialise(db)
    db.executescript("""
        INSERT INTO venues VALUES ('venue_01', 'V');
        INSERT INTO fields VALUES ('field_01', 'venue_01', 'F1');
        INSERT INTO rental_slots VALUES
            ('venue_01_2026-07-11_1000', 'field_01', '2026-07-11', '10:00', '11:00'),
            ('venue_01_2026-07-11_2030', 'field_01', '2026-07-11', '20:30', '21:30');
        INSERT INTO slot_evaluations (slot_id, status, reason, play_ratio, empty_ratio,
            maintenance_ratio, n_samples, mean_confidence, evaluated_at, model_key)
        VALUES ('venue_01_2026-07-11_1000', 'USED', 'r', .9, .1, 0, 60, .9, 'now', 'm'),
               ('venue_01_2026-07-11_2030', 'USED', 'r', .9, .1, 0, 60, .9, 'now', 'm');
    """)

    app = FastAPI()
    app.include_router(bookings_page.router)
    app.dependency_overrides[get_conn] = lambda: db
    yield TestClient(app), cfg, client_db
    db.close()


def test_api_without_a_source(api):
    client, _cfg, _ = api
    assert client.get("/api/v1/bookings/source").json()["configured"] is False
    r = client.post("/api/v1/bookings/test")
    assert r.status_code == 422 and "No booking source is configured" in r.json()["detail"]
    page = client.get("/bookings")
    assert page.status_code == 200 and "read-only" in page.text
    cmp = client.get("/api/v1/bookings/compare", params={"day": "2026-07-11"}).json()
    assert cmp["configured"] is False and len(cmp["slots"]) == 2
    assert cmp["slots"][0]["anomaly"] is None


def test_api_with_a_sqlite_source(api):
    client, cfg, db_path = api
    cfg.write_text(json.dumps({"type": "sqlite", "path": str(db_path),
                               "query": "SELECT * FROM reservations", **SQLITE_CONFIG}),
                   encoding="utf-8")
    src = client.get("/api/v1/bookings/source").json()
    assert src["valid"] and src["source"]["type"] == "sqlite"

    t = client.post("/api/v1/bookings/test").json()
    assert t["rows"] == 3 and t["date_from"] == "2026-07-11" and len(t["preview"]) == 3

    rep = client.post("/api/v1/bookings/check").json()
    assert rep["checked"] == "the configured sqlite source" and rep["total_rows"] == 3

    cmp = client.get("/api/v1/bookings/compare", params={"day": "2026-07-11"}).json()
    by_start = {s["start"]: s for s in cmp["slots"]}
    assert by_start["10:00"]["booking"] == "confirmed"
    # Cancelled and played: the finding reconciliation exists to make.
    assert by_start["20:30"]["anomaly"] == "UNBOOKED_USAGE"
    empty = client.get("/api/v1/bookings/compare", params={"day": "2026-07-20"}).json()
    assert empty["slots"] == [] and "no camera slots" in empty["message"]


def test_api_check_an_uploaded_csv(api):
    client, _cfg, _ = api
    body = ("field_id,date,start,end,customer_ref,status,entered_by\n"
            "f1,2026-07-11,10:00,11:00,R1,confirmed,a\n"
            "f1,2026-07-11,10:30,11:30,R2,booked,a\n")
    rep = client.post("/api/v1/bookings/check", content=body,
                      headers={"Content-Type": "text/csv"}).json()
    assert rep["verdict"] == "has errors"
    assert any(i["code"] == "unknown_status" for i in rep["issues"])
    bad = client.post("/api/v1/bookings/check", content="a,b\n1,2\n")
    assert bad.status_code == 422


def test_api_never_echoes_a_password(api):
    client, cfg, _ = api
    cfg.write_text(json.dumps({"type": "sql", "url": "postgresql://u:topsecret@h/db",
                               "query": "SELECT * FROM b"}), encoding="utf-8")
    assert "topsecret" not in client.get("/api/v1/bookings/source").text
    r = client.post("/api/v1/bookings/test")
    assert r.status_code == 422 and "topsecret" not in r.text
