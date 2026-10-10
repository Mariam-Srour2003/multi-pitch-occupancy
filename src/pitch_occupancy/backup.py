"""Copy the verdict database (and, optionally, this system's own config files) to a backup folder.

The database at ``settings.db_path`` is the one thing the running product writes that cannot be
rebuilt: every slot verdict, every operator correction, every anomaly. The footage policy in
``docs/backup.md`` covers ``data/`` as a whole; this module is the small, daily, automatic part
of it, run as ``pitch backup``.

Four things about it are deliberate.

**SQLite's online backup API, not a file copy.** The worker may be writing a verdict at the
moment the backup runs. Copying the ``.db`` file then can catch half a transaction - and with
WAL journaling the newest rows are not even in that file yet. ``sqlite3.Connection.backup``
copies a consistent snapshot page by page while the writer carries on, so the backup is safe to
schedule without stopping anything.

**Every copy is checked before it counts.** The new file is opened and ``PRAGMA
integrity_check`` must answer ``ok``; a copy that fails is deleted and the command fails, so a
folder of backups never holds one nobody could restore from. The copy is written under a
``.partial`` name and renamed only once it has passed, so a crash mid-copy leaves no file that
looks like a backup.

**Old copies are pruned, newest kept.** ``pitch_monitor-YYYYMMDD-HHMMSS.db`` sorts by time, so
pruning keeps the newest ``keep`` files (14 by default: two weeks of daily backups) and their
config zips, and touches nothing else in the folder.

**Config files are optional, and secrets need asking for.** The config zip holds this
system's own state - rules, boundaries, schedule, and with ``--with-secrets`` the camera list,
booking source and alerts. **Those three files contain passwords** (camera stream logins, a
booking database login, mail or chat credentials). They are left out unless asked for, and when
they are included the backup folder must be private: anyone who can read it can log in to the
cameras.
"""

from __future__ import annotations

import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pitch_occupancy.config import CONFIGS_DIR, settings

__all__ = [
    "BackupResult", "backup_database", "backup_configs", "run_backup", "prune",
    "CONFIG_FILES", "SECRET_FILES", "DB_PREFIX",
]

#: Backup files are named ``<DB_PREFIX>-YYYYMMDD-HHMMSS.db``.
DB_PREFIX = "pitch_monitor"

#: Where the config files are read from (a module attribute so tests can point it elsewhere).
CONFIGS: Path = CONFIGS_DIR

#: This system's own state in ``configs/``, without passwords. Globs, matched in ``CONFIGS``.
CONFIG_FILES: tuple[str, ...] = ("rules.json", "roi*.json", "slots_schedule.json")

#: Config files that hold passwords: included only with ``--with-secrets``.
SECRET_FILES: tuple[str, ...] = ("cameras.json", "bookings_source.json", "alerts.json")

_STAMP = "%Y%m%d-%H%M%S"


@dataclass
class BackupResult:
    """What one ``pitch backup`` produced."""

    database: Path
    configs: Path | None = None
    config_names: list[str] = field(default_factory=list)
    secrets_included: bool = False
    removed: list[Path] = field(default_factory=list)


def _target(dest_dir: Path, when: datetime) -> Path:
    """A free ``pitch_monitor-<stamp>.db`` name; a second backup in the same second gets a
    ``-2`` suffix rather than overwriting the first."""
    base = f"{DB_PREFIX}-{when.strftime(_STAMP)}"
    candidate, n = dest_dir / f"{base}.db", 2
    while candidate.exists():
        candidate, n = dest_dir / f"{base}-{n}.db", n + 1
    return candidate


def _integrity(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    finally:
        conn.close()
    return "; ".join(str(r[0]) for r in rows) or "no answer"


def prune(dest_dir: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` database backups in ``dest_dir``, with their config
    zips. Only files this module names are considered. Returns what was removed."""
    backups = sorted(dest_dir.glob(f"{DB_PREFIX}-*.db"), key=lambda p: p.name)
    removed = []
    for old in backups[:max(0, len(backups) - max(keep, 1))]:
        for path in (old, old.with_name(old.stem + "-configs.zip")):
            if path.exists():
                path.unlink()
                removed.append(path)
    return removed


def backup_database(dest_dir: Path, keep: int = 14, *, db_path: Path | None = None,
                    when: datetime | None = None) -> Path:
    """Copy the live database to ``dest_dir/pitch_monitor-YYYYMMDD-HHMMSS.db`` and return it.

    Safe while the worker is writing (SQLite's online backup API). The copy must pass
    ``PRAGMA integrity_check`` or it is deleted and ``RuntimeError`` raised. Afterwards the
    folder is pruned to the newest ``keep`` backups.

    Raises:
        FileNotFoundError: there is no database yet - nothing to back up, and opening one
            would create an empty file.
        RuntimeError: the copy failed its integrity check.
    """
    target = _copy_database(Path(dest_dir), db_path=db_path, when=when)
    prune(Path(dest_dir), keep)
    return target


def _copy_database(dest_dir: Path, *, db_path: Path | None = None,
                   when: datetime | None = None) -> Path:
    """The copy and its check, without pruning - see :func:`backup_database`."""
    source = Path(db_path or settings.db_path)
    if not source.is_file():
        raise FileNotFoundError(f"no database at {source} - nothing to back up yet")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = _target(dest_dir, when or datetime.now())
    partial = target.with_name(target.name + ".partial")
    partial.unlink(missing_ok=True)

    src = sqlite3.connect(str(source))
    try:
        dst = sqlite3.connect(str(partial))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    verdict = _integrity(partial)
    if verdict != "ok":
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"the backup copy failed its integrity check ({verdict}); nothing "
                           f"was kept. Check the database at {source}.")
    partial.replace(target)
    return target


def backup_configs(database_backup: Path, *,
                   with_secrets: bool = False) -> tuple[Path | None, list[str]]:
    """Zip this system's config files beside a database backup, as ``<name>-configs.zip``.

    Without ``with_secrets`` the files that hold passwords (:data:`SECRET_FILES`) are left
    out. **With it, the zip holds camera, booking-database and alert logins**: keep the
    backup folder private. Returns the zip (None if there was nothing to put in it) and the
    names it holds.
    """
    patterns = CONFIG_FILES + (SECRET_FILES if with_secrets else ())
    names = sorted({p.name for pattern in patterns for p in CONFIGS.glob(pattern)
                    if p.is_file()})
    if not with_secrets:
        names = [n for n in names if n not in SECRET_FILES]
    if not names:
        return None, []
    target = database_backup.with_name(database_backup.stem + "-configs.zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(CONFIGS / name, arcname=f"configs/{name}")
    return target, names


def run_backup(dest_dir: Path, keep: int = 14, *, configs: bool = True,
               with_secrets: bool = False) -> BackupResult:
    """The whole ``pitch backup``: the database, then (by default) the config zip beside it."""
    database = _copy_database(Path(dest_dir))
    result = BackupResult(database=database, secrets_included=with_secrets)
    if configs:
        result.configs, result.config_names = backup_configs(database,
                                                             with_secrets=with_secrets)
    result.removed = prune(Path(dest_dir), keep)
    return result
