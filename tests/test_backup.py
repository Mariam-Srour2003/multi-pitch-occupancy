"""`pitch backup`: a checked copy of the database, pruned, and config files without passwords.

Every test works on a database it creates in ``tmp_path`` and points ``settings.db_path`` and
``settings.data_dir`` there, so the real ``data/db`` is never opened.
"""

from __future__ import annotations

import sqlite3
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pitch_occupancy import backup
from pitch_occupancy.cli import app
from pitch_occupancy.config import settings


@pytest.fixture
def db(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "data" / "db" / "pitch_monitor.db"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE slot_results (slot_id TEXT PRIMARY KEY, status TEXT)")
    conn.executemany("INSERT INTO slot_results VALUES (?, ?)",
                     [(f"venue_01_2026-10-0{i}_1000", "USED") for i in range(1, 6)])
    conn.commit()
    conn.close()
    monkeypatch.setattr(settings, "db_path", path)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    return path


@pytest.fixture
def configs(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "configs"
    folder.mkdir()
    for name in ("rules.json", "roi.json", "roi_derived.json", "slots_schedule.json",
                 "cameras.json", "bookings_source.json", "alerts.json", "clip_venues.csv"):
        (folder / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(backup, "CONFIGS", folder)
    return folder


def _rows(path: Path) -> int:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT COUNT(*) FROM slot_results").fetchone()[0]
    finally:
        conn.close()


def test_the_copy_is_complete_and_passes_its_integrity_check(db, tmp_path) -> None:
    # A writer holds the database open, its newest rows still only in the WAL - the worker.
    writer = sqlite3.connect(str(db))
    writer.execute("INSERT INTO slot_results VALUES ('venue_01_2026-10-09_1000', 'NOTUSED')")
    writer.commit()
    try:
        out = backup.backup_database(tmp_path / "backups",
                                     when=datetime(2026, 10, 10, 3, 30, 0))
    finally:
        writer.close()
    assert out.name == "pitch_monitor-20261010-033000.db"
    assert _rows(out) == 6, "the backup missed rows still in the WAL"
    conn = sqlite3.connect(str(out))
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()
    assert not list((tmp_path / "backups").glob("*.partial"))


def test_two_backups_in_one_second_do_not_overwrite(db, tmp_path) -> None:
    when = datetime(2026, 10, 10, 3, 30, 0)
    a = backup.backup_database(tmp_path / "b", when=when)
    b = backup.backup_database(tmp_path / "b", when=when)
    assert a != b and a.exists() and b.exists()


def test_old_backups_are_pruned_newest_kept(db, tmp_path) -> None:
    dest = tmp_path / "backups"
    dest.mkdir()
    (dest / "unrelated.db").write_bytes(b"keep me")
    start = datetime(2026, 10, 1, 3, 30)
    made = [backup.backup_database(dest, keep=3, when=start + timedelta(days=i))
            for i in range(5)]
    (made[-1].with_name(made[-1].stem + "-configs.zip")).write_bytes(b"zip")
    left = sorted(p.name for p in dest.glob("pitch_monitor-*.db"))
    assert left == [p.name for p in made[-3:]]
    assert (dest / "unrelated.db").exists()


def test_a_copy_that_fails_its_check_is_not_kept(db, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backup, "_integrity", lambda path: "*** page 3 is never used")
    with pytest.raises(RuntimeError, match="integrity"):
        backup.backup_database(tmp_path / "backups")
    assert not list((tmp_path / "backups").iterdir())


def test_no_database_is_an_error_not_an_empty_file(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "data" / "db" / "pitch_monitor.db"
    monkeypatch.setattr(settings, "db_path", missing)
    with pytest.raises(FileNotFoundError):
        backup.backup_database(tmp_path / "backups")
    assert not missing.exists()


def test_configs_leave_out_the_files_with_passwords_by_default(db, configs, tmp_path) -> None:
    result = backup.run_backup(tmp_path / "backups")
    assert result.configs is not None and not result.secrets_included
    names = zipfile.ZipFile(result.configs).namelist()
    assert sorted(names) == ["configs/roi.json", "configs/roi_derived.json",
                             "configs/rules.json", "configs/slots_schedule.json"]


def test_secrets_only_when_asked(db, configs, tmp_path) -> None:
    result = backup.run_backup(tmp_path / "backups", with_secrets=True)
    names = set(zipfile.ZipFile(result.configs).namelist())
    assert {"configs/cameras.json", "configs/bookings_source.json",
            "configs/alerts.json"} <= names
    assert "configs/clip_venues.csv" not in names


def test_the_command(db, configs, tmp_path) -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["backup", "--keep", "2"])
    assert r.exit_code == 0, r.output
    dest = tmp_path / "data" / "backups"
    assert len(list(dest.glob("pitch_monitor-*.db"))) == 1
    assert "integrity check ok" in r.output and "--with-secrets" in r.output

    r = runner.invoke(app, ["backup", "--dest", str(tmp_path / "private"), "--with-secrets"])
    assert r.exit_code == 0, r.output
    assert "WARNING" in r.output and "passwords" in r.output


def test_the_command_fails_cleanly_without_a_database(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "db_path", tmp_path / "none.db")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    r = CliRunner().invoke(app, ["backup"])
    assert r.exit_code == 1
