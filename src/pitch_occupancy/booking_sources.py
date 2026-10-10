"""Where the bookings come from: a CSV export, a SQLite file, or another database.

`bookings.py` defines what a booking *is* and how one CSV export becomes records. This module is
the layer in front of it that lets a facility point the system at whatever it already has,
**without changing anything on the facility's side**: a client's table keeps its own column
names and its own status words, and a small config file says how they map onto the normalised
schema (``field_id, date, start, end, customer_ref, status, entered_by``).

The config lives at ``configs/bookings_source.json`` and is **gitignored**, because a database
URL carries a password. ``configs/bookings_source.example.json`` shows each shape::

    {"type": "csv",    "path": "exports/bookings.csv"}
    {"type": "sqlite", "path": "C:/booking_app/bookings.db", "query": "SELECT ..."}
    {"type": "sql",    "url": "postgresql://reader:***@host/db", "query": "SELECT ..."}

and every type accepts ``columns`` (normalised name -> their column), ``status_map`` (their
value -> one of `bookings.STATUSES`), and ``date_format`` / ``time_format`` /
``datetime_format`` (strptime patterns). A ``start``/``end`` column holding a full datetime is
split into the date and the time, so a table with ``starts_at``/``ends_at`` needs no ``date``
column at all. Relative paths are resolved against the project root.

**Strictly read-only.** The reasons are `bookings.py`'s and `slots/authority.py`'s: the booking
system is the facility's financial record, and this project is an instrument pointed at it.
Every source here has the two methods a reader needs and none a writer would:

* :meth:`read` - strict, like `bookings.read_bookings`: the first bad row raises, because a
  booking silently dropped at import becomes an "unbooked usage" anomaly against a customer;
* :meth:`rows` - the same rows mapped but *not* judged, for `booking_check`, which wants to
  see every problem at once rather than stop at the first.

How each type is kept to reading:

* **csv** - opened for reading. There is nothing else to say.
* **sqlite** - opened with ``mode=ro`` in the URI *and* ``PRAGMA query_only``, so the file
  cannot be written through this connection even by a query that slipped past the guard.
* **sql** - :func:`check_read_only` rejects anything but a single ``SELECT``/``WITH``
  statement, and PostgreSQL sessions are additionally set ``READ ONLY``. **The guard is a
  guard against mistakes, not a security boundary**: SQL is too rich a language to be made
  safe by inspecting its text, and a stored procedure called from a SELECT can do anything.
  The real protection is the one only the facility can provide - give this system a database
  user that has been granted ``SELECT`` on the bookings table and nothing else. The example
  config says so too, because that is where the person setting it up will be looking.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from pitch_occupancy.bookings import (
    EXAMPLE_PATH,
    FIELDS,
    STATUSES,
    BookingRecord,
    read_bookings,
)
from pitch_occupancy.config import CONFIGS_DIR, PROJECT_ROOT

__all__ = [
    "DEFAULT_SOURCE_CONFIG",
    "EXAMPLE_SOURCE_CONFIG",
    "REQUIRED",
    "CsvSource",
    "SqliteSource",
    "SqlSource",
    "check_read_only",
    "describe",
    "load_source",
    "mask_url",
    "parse_csv_text",
    "records_from_rows",
    "rows_from_csv_text",
    "source_from_config",
]

DEFAULT_SOURCE_CONFIG: Path = CONFIGS_DIR / "bookings_source.json"
EXAMPLE_SOURCE_CONFIG: Path = CONFIGS_DIR / "bookings_source.example.json"

#: Columns a source must provide. ``date`` is not in the list because it can come from a
#: datetime ``start`` column; ``customer_ref`` and ``entered_by`` are optional because a
#: client's table may simply not have them, and an empty value is honest where a guess is not.
REQUIRED: tuple[str, ...] = ("field_id", "start", "end", "status")

TYPES: tuple[str, ...] = ("csv", "sqlite", "sql")
_COMMON_KEYS = {
    "type", "name", "columns", "status_map", "date_format", "time_format", "datetime_format",
    "_comment",
}
_TYPE_KEYS = {"csv": {"path"}, "sqlite": {"path", "query"}, "sql": {"url", "query"}}

_DEFAULT_TIME_FORMATS = ("%H:%M", "%H:%M:%S")
_DEFAULT_DATE_FORMAT = "%Y-%m-%d"

#: Words that make a statement write, or change the database, or reach outside the query.
#: ``REPLACE`` is deliberately absent - it is also an everyday string function in a SELECT, and
#: ``INSERT OR REPLACE`` is caught by ``INSERT`` anyway. ``INTO`` is present because
#: ``SELECT ... INTO new_table`` creates a table in PostgreSQL and SQL Server.
_WRITE_WORDS = (
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "EXEC", "EXECUTE",
    "CALL", "COPY", "INTO", "LOCK", "SET", "COMMIT", "ROLLBACK", "BEGIN", "DECLARE",
)
_WRITE_RE = re.compile(r"\b(" + "|".join(_WRITE_WORDS) + r")\b", re.IGNORECASE)


# --- the read-only guard ------------------------------------------------------------------


def _strip_sql(query: str) -> str:
    """The query with comments removed and string/identifier literals blanked out.

    Blanking the literals is what lets the keyword check run without false alarms: a
    ``WHERE note = 'please update'`` is a read, and a column called ``"delete"`` in double
    quotes is a name.
    """
    out: list[str] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if query.startswith("--", i):
            j = query.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
        elif query.startswith("/*", i):
            j = query.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        elif ch in ("'", '"', "`", "["):
            close = "]" if ch == "[" else ch
            j = i + 1
            while j < n:
                if query[j] == close:
                    if close != "]" and j + 1 < n and query[j + 1] == close:
                        j += 2  # doubled quote: an escaped quote inside the literal
                        continue
                    break
                j += 1
            out.append(ch + "x" + close)
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def check_read_only(query: str) -> str:
    """Return ``query`` if it looks like one read-only statement; raise ValueError otherwise.

    Accepts a single statement beginning with ``SELECT`` or ``WITH`` (after comments and
    whitespace), with at most one trailing ``;``. Rejects a second statement and any of the
    obvious write keywords at statement level.

    **A guard against mistakes, not a security boundary** - see the module docstring. The
    protection that holds is a database user with SELECT permission and nothing more.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("the query is empty; it should be a SELECT over the bookings table")
    bare = _strip_sql(query).strip()
    if bare.endswith(";"):
        bare = bare[:-1].rstrip()
    if ";" in bare:
        raise ValueError(
            "the query contains more than one statement (a ';' followed by more text). "
            "Only a single SELECT is allowed"
        )
    first = bare.split(None, 1)[0].upper() if bare else ""
    if first not in ("SELECT", "WITH"):
        raise ValueError(
            f"the query must start with SELECT or WITH, not {first or 'nothing'!r}. "
            "This system only ever reads booking data"
        )
    word = _WRITE_RE.search(bare)
    if word:
        raise ValueError(
            f"the query contains {word.group(1).upper()!r}, which can change data. "
            "Only plain SELECT queries are allowed - this system never writes to a "
            "booking system"
        )
    return query


# --- masking ------------------------------------------------------------------------------

_URL_PASSWORD = re.compile(r"(://[^:/@\s]*:)([^@\s]*)(@)")
_KV_PASSWORD = re.compile(r"(?i)\b(password|passwd|pwd)=([^&;\s]+)")
_SECRET_KEYS = re.compile(r"(?i)pass|pwd|secret|token|key")


def mask_url(url: str) -> str:
    """The URL with any password replaced by ``***``.

    Covers ``user:password@host`` and ``password=``/``pwd=`` parameters, which is where an
    ODBC connection string (``mssql+pyodbc:///?odbc_connect=...PWD=...``) keeps it.
    """
    masked = _URL_PASSWORD.sub(r"\1***\3", url)
    return _KV_PASSWORD.sub(r"\1=***", masked)


def describe(config: Mapping[str, Any]) -> dict:
    """A copy of a source config that is safe to show on a page: credentials masked.

    The ``_comment`` is dropped; any key that looks like it holds a secret is masked.
    """
    out: dict[str, Any] = {}
    for key, value in config.items():
        if key == "_comment":
            continue
        if key == "url" and isinstance(value, str):
            out[key] = mask_url(value)
        elif key not in ("columns", "status_map", "query") and _SECRET_KEYS.search(key):
            out[key] = "***"
        else:
            out[key] = value
    return out


# --- parsing values -----------------------------------------------------------------------


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # a numeric field id read back from a spreadsheet-made table
    return str(value).strip()


def _fmt_time(t: time) -> str:
    return f"{t:%H:%M}" if t.second == 0 else f"{t:%H:%M:%S}"


class _Formats:
    def __init__(self, config: Mapping[str, Any]):
        self.date = config.get("date_format") or _DEFAULT_DATE_FORMAT
        tf = config.get("time_format")
        self.times = (tf,) if tf else _DEFAULT_TIME_FORMATS
        self.datetime = config.get("datetime_format")

    def as_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        text = _text(value)
        if not text:
            return None
        if self.datetime:
            try:
                return datetime.strptime(text, self.datetime)
            except ValueError:
                return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        # A bare "2026-07-11" parses as midnight; that is a date, not a start time.
        return parsed if ("T" in text or " " in text) else None

    def as_time(self, value: Any) -> time | None:
        if isinstance(value, time):
            return value
        text = _text(value)
        for fmt in self.times:
            try:
                return datetime.strptime(text, fmt).time()
            except ValueError:
                continue
        return None

    def as_date(self, value: Any) -> date | None:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = _text(value)
        try:
            return datetime.strptime(text, self.date).date()
        except ValueError:
            pass
        moment = self.as_datetime(text)
        return moment.date() if moment else None


def _normalise(raw: Mapping[str, Any], *, row: int, columns: Mapping[str, str],
               status_map: Mapping[str, str], formats: _Formats) -> dict:
    """Map one raw row onto the normalised schema, leaving anything unparseable as written.

    Nothing is judged here. A value that does not parse is kept verbatim so that the strict
    reader can refuse it and the checker can quote it back to a person.
    """
    def get(name: str) -> Any:
        return raw.get(columns.get(name, name))

    out: dict[str, Any] = {"row": row}
    out["field_id"] = _text(get("field_id"))
    out["customer_ref"] = _text(get("customer_ref"))
    out["entered_by"] = _text(get("entered_by"))

    day = formats.as_date(get("date")) if _text(get("date")) else None
    out["date"] = day.isoformat() if day else _text(get("date"))

    for name in ("start", "end"):
        value = get(name)
        t = formats.as_time(value)
        if t is None:
            moment = formats.as_datetime(value)
            if moment is not None:
                t = moment.time()
                if name == "start" and not out["date"]:
                    out["date"] = moment.date().isoformat()
        out[name] = _fmt_time(t) if t is not None else _text(value)

    status = _text(get("status"))
    mapped = status_map.get(status)
    if mapped is None:
        lowered = {k.strip().lower(): v for k, v in status_map.items()}
        mapped = lowered.get(status.lower(), status.lower())
    out["status"] = mapped
    return out


def _missing_columns(header: Sequence[str], columns: Mapping[str, str]) -> list[str]:
    present = set(header)
    missing = [f"{n} (their column {columns.get(n, n)!r})" if columns.get(n, n) != n else n
               for n in REQUIRED if columns.get(n, n) not in present]
    return missing


def _map_rows(header: Sequence[str], raws: Iterable[tuple[int, Mapping[str, Any]]],
              config: Mapping[str, Any], *, what: str) -> list[dict]:
    columns = dict(config.get("columns") or {})
    missing = _missing_columns(header, columns)
    if missing:
        raise ValueError(
            f"{what} has no column for: {', '.join(missing)}. Columns found: "
            f"{', '.join(header) or 'none'}. Use \"columns\" in the source config to say which "
            f"of their columns holds each value"
        )
    formats = _Formats(config)
    status_map = dict(config.get("status_map") or {})
    return [_normalise(raw, row=i, columns=columns, status_map=status_map, formats=formats)
            for i, raw in raws]


def _csv_rows(fh: io.TextIOBase, config: Mapping[str, Any], *, what: str) -> list[dict]:
    reader = csv.DictReader(fh)
    header = [h.strip() for h in (reader.fieldnames or [])]
    reader.fieldnames = header
    raws = (
        (i, raw) for i, raw in enumerate(reader, start=2)
        if any(_text(v) for v in raw.values() if not isinstance(v, list))
    )
    return _map_rows(header, raws, config, what=what)


def rows_from_csv_text(text: str, config: Mapping[str, Any] | None = None) -> list[dict]:
    """Lenient rows from CSV text, for the checker. Raises only if a required column is absent.

    Row numbers are file line numbers (the header is line 1), as `read_bookings` counts them.
    """
    return _csv_rows(io.StringIO(text.lstrip("\ufeff"), newline=""), config or {},
                     what="the CSV")


def parse_csv_text(text: str, source: str = "upload") -> list[BookingRecord]:
    """Strictly parse uploaded CSV text with the standard columns, via `read_bookings` itself.

    Goes through a temporary file rather than a copy of the reader's rules, so an upload is
    accepted or refused for exactly the reasons a file on disk would be.

    Raises:
        ValueError: with the reader's own message, naming the upload rather than a temp file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.csv"
        path.write_text(text.lstrip("\ufeff"), encoding="utf-8", newline="")
        try:
            return read_bookings(path, source=source)
        except ValueError as exc:
            raise ValueError(str(exc).replace("upload.csv", "the uploaded CSV")) from exc


def records_from_rows(rows: Iterable[Mapping[str, Any]], *, source: str) -> list[BookingRecord]:
    """Turn normalised rows into records, refusing the first bad one (as `read_bookings` does).

    The rules are `read_bookings`'s: a known status, a date, two times with the end after the
    start, a field, and one booking per field and start time.
    """
    records: list[BookingRecord] = []
    seen: set[tuple[str, date, time]] = set()
    for r in rows:
        i = r.get("row", "?")
        status = _text(r.get("status")).lower()
        if status not in STATUSES:
            raise ValueError(f"row {i}: status {r.get('status')!r} is not one of "
                             f"{', '.join(STATUSES)}; a \"status_map\" in the source config "
                             f"can translate the booking system's own words")
        try:
            day = date.fromisoformat(_text(r.get("date")))
        except ValueError:
            raise ValueError(f"row {i}: date {r.get('date')!r} is not a date (set "
                             f"\"date_format\" if the system writes dates differently)") from None
        times = []
        for name in ("start", "end"):
            try:
                times.append(time.fromisoformat(_text(r.get(name))))
            except ValueError:
                raise ValueError(f"row {i}: {name} {r.get(name)!r} is not a time (set "
                                 f"\"time_format\" if needed)") from None
        start, end = times
        if end <= start:
            raise ValueError(f"row {i}: end {r.get('end')} is not after start {r.get('start')}; "
                             f"a slot crossing midnight must be split into two rows")
        field_id = _text(r.get("field_id"))
        if not field_id:
            raise ValueError(f"row {i}: field_id is empty")
        key = (field_id, day, start)
        if key in seen:
            raise ValueError(f"row {i}: a second booking for {field_id} at {day} "
                             f"{start:%H:%M}; the data needs deduplicating before import")
        seen.add(key)
        records.append(BookingRecord(
            field_id=field_id, date=day, start=start, end=end,
            customer_ref=_text(r.get("customer_ref")), status=status,
            entered_by=_text(r.get("entered_by")), source=source,
        ))
    return records


# --- the sources --------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


class _MappedSource:
    """Shared behaviour: subclasses fetch raw rows, this maps and (for `read`) judges them."""

    kind = ""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)

    @property
    def source_tag(self) -> str:
        return str(self.config.get("name") or self.kind)

    def rows(self) -> list[dict]:  # pragma: no cover - overridden
        raise NotImplementedError

    def read(self) -> list[BookingRecord]:
        return records_from_rows(self.rows(), source=self.source_tag)

    def describe(self) -> dict:
        return describe(self.config)


class CsvSource(_MappedSource):
    """A CSV export, with optional column and status mapping."""

    kind = "csv"

    @property
    def path(self) -> Path:
        return _resolve(self.config["path"])

    @property
    def source_tag(self) -> str:
        if self.config.get("name"):
            return str(self.config["name"])
        # Same rule as read_bookings: the committed example is always labelled as such.
        return "example" if self.path.resolve() == EXAMPLE_PATH.resolve() else self.path.stem

    def rows(self) -> list[dict]:
        if not self.path.exists():
            raise FileNotFoundError(f"no booking CSV at {self.path}")
        with self.path.open(newline="", encoding="utf-8-sig") as fh:
            return _csv_rows(fh, self.config, what=self.path.name)


class SqliteSource(_MappedSource):
    """A SQLite database file, opened read-only."""

    kind = "sqlite"

    @property
    def path(self) -> Path:
        return _resolve(self.config["path"])

    @property
    def source_tag(self) -> str:
        return str(self.config.get("name") or self.path.stem)

    def _open(self) -> sqlite3.Connection:
        """A connection that cannot write: ``mode=ro`` and ``query_only``, both."""
        if not self.path.exists():
            # mode=ro would refuse too, but with "unable to open database file", which
            # does not tell anyone which file.
            raise FileNotFoundError(f"no SQLite database at {self.path}")
        conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        return conn

    def rows(self) -> list[dict]:
        query = check_read_only(self.config["query"])
        conn = self._open()
        try:
            cur = conn.execute(query)
            header = [d[0] for d in cur.description or ()]
            raws = [(i, dict(zip(header, values, strict=True)))
                    for i, values in enumerate(cur.fetchall(), start=1)]
        except sqlite3.Error as exc:
            raise ValueError(f"the SQLite query failed: {exc}") from exc
        finally:
            conn.close()
        return _map_rows(header, raws, self.config, what="the query result")


_SQL_HELP = (
    "Reading a database other than SQLite needs SQLAlchemy and the driver for that database, "
    "which are not installed. From the project folder run one of:\n"
    "  uv add sqlalchemy psycopg2-binary   (PostgreSQL)\n"
    "  uv add sqlalchemy pymysql           (MySQL / MariaDB)\n"
    "  uv add sqlalchemy pyodbc            (SQL Server, with Microsoft's ODBC driver)"
)


class SqlSource(_MappedSource):
    """Any database SQLAlchemy can reach. SQLAlchemy is imported only when this is used."""

    kind = "sql"

    def rows(self) -> list[dict]:
        query = check_read_only(self.config["query"])
        try:
            import sqlalchemy
        except ImportError as exc:
            raise RuntimeError(_SQL_HELP) from exc
        try:
            engine = sqlalchemy.create_engine(self.config["url"])
        except ImportError as exc:  # the dialect's driver, e.g. psycopg2
            raise RuntimeError(f"{_SQL_HELP}\n(missing: {exc.name or exc})") from exc
        try:
            with engine.connect() as conn:
                if engine.dialect.name == "postgresql":
                    # A second lock on the door: the server itself refuses writes for this
                    # transaction, whatever the query turns out to do.
                    conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                result = conn.execute(sqlalchemy.text(query))
                header = list(result.keys())
                raws = [(i, dict(m)) for i, m in enumerate(result.mappings(), start=1)]
                conn.rollback()
        except sqlalchemy.exc.SQLAlchemyError as exc:
            # The message can quote the URL; mask it before it reaches a page.
            raise ValueError(f"the database query failed: {mask_url(str(exc))}") from None
        finally:
            engine.dispose()
        return _map_rows(header, raws, self.config, what="the query result")


_CLASSES = {"csv": CsvSource, "sqlite": SqliteSource, "sql": SqlSource}


# --- loading the config -------------------------------------------------------------------


def _validate(config: Any, where: str) -> dict:
    if not isinstance(config, dict):
        raise ValueError(f"{where} must hold a JSON object like {{\"type\": \"csv\", ...}}")
    kind = config.get("type")
    if kind not in TYPES:
        raise ValueError(f"{where}: \"type\" must be one of {', '.join(TYPES)}, not {kind!r}")
    allowed = _COMMON_KEYS | _TYPE_KEYS[kind]
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown setting(s) {', '.join(unknown)} for type {kind!r}. "
                         f"Allowed: {', '.join(sorted(allowed - {'_comment'}))}")
    for key in sorted(_TYPE_KEYS[kind]):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{where}: type {kind!r} needs \"{key}\" (text)")
    columns = config.get("columns", {})
    if not isinstance(columns, dict) or not all(
            isinstance(v, str) and v for v in columns.values()):
        raise ValueError(f"{where}: \"columns\" must map normalised names to column names")
    bad = sorted(set(columns) - set(FIELDS))
    if bad:
        raise ValueError(f"{where}: \"columns\" has unknown name(s) {', '.join(bad)}; the "
                         f"names on the left must be from: {', '.join(FIELDS)}")
    status_map = config.get("status_map", {})
    if not isinstance(status_map, dict):
        raise ValueError(f"{where}: \"status_map\" must map their values to ours")
    wrong = {k: v for k, v in status_map.items() if v not in STATUSES}
    if wrong:
        raise ValueError(f"{where}: \"status_map\" maps to unknown status(es) "
                         f"{', '.join(map(repr, wrong.values()))}; use one of "
                         f"{', '.join(STATUSES)}")
    for key in ("date_format", "time_format", "datetime_format", "name"):
        if key in config and not isinstance(config[key], str):
            raise ValueError(f"{where}: \"{key}\" must be text")
    if "query" in config:
        try:
            check_read_only(config["query"])
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
    return config


def source_from_config(config: Any, *, where: str = "the booking source config"
                       ) -> CsvSource | SqliteSource | SqlSource:
    """Build a source from an already-parsed config, validating it first."""
    config = _validate(config, where)
    return _CLASSES[config["type"]](config)


def load_source(path: Path | None = None) -> CsvSource | SqliteSource | SqlSource | None:
    """The configured booking source, or ``None`` if none is configured.

    Booking data is optional: a facility without a booking system still gets occupancy
    verdicts, so a missing config file is a normal state, not an error.

    Raises:
        ValueError: the file exists but is not a valid config, with a readable reason.
    """
    target = path or DEFAULT_SOURCE_CONFIG
    if not target.exists():
        return None
    try:
        config = json.loads(target.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target.name} is not valid JSON: {exc}") from None
    return source_from_config(config, where=target.name)
