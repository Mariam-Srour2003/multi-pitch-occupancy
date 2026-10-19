"""Alerts: telling the facility when something needs attention, without anyone watching a page.

The dashboard answers questions when somebody asks them. The client also wants to be *told*,
on the phone they already carry, when one of four things happens:

* ``anomaly`` - a slot's reconciliation disagrees with the records: booked but not played
  (a possible no-show), played but not booked, played but not checked in, sold while closed.
  Read from the ``reconciliations`` table. CONSISTENT and NEEDS_REVIEW are not anomalies and
  are never sent (`slots/reconcile.py` is explicit that REVIEW is routed to a human, not
  flagged), and neither is a row someone has already marked resolved;
* ``camera_silent`` - a camera that the schedule says should be recording right now has sent no
  picture for longer than ``camera_silent_minutes`` (default 10). Read through
  `cameras.recording_health`, matched on the camera id exactly;
* ``worker_down`` / ``worker_error`` - the recording program is not running while a slot is in
  progress (so that slot is not being recorded), or it says its state is ``error``. Read
  through `pitch_occupancy.status.read_status`, imported lazily because the status file is
  written by another part of the system; when it is not there the check is skipped and the
  notes say so;
* ``daily_summary`` (optional) - once a day, at the configured time, yesterday in one line:
  slots used / not used / to review, no-shows and unbooked play, from
  `reports.daily_summary`, so it agrees with the Reports page by construction.

Two channels, both from the standard library so nothing new is installed:

* **webhook** - a JSON POST. ``text`` holds one plain-English line, which Slack, Microsoft
  Teams and Discord (its ``/slack`` URL) show as the message, and which automation services
  (Zapier, Make, n8n, Power Automate) can forward to WhatsApp, Telegram or SMS. The rest of the
  payload (``event``, ``severity``, ``pitch``, ``slot``, ``time``, ``details``) is there for an
  automation that wants to route on it. 10-second timeout, one retry;
* **email** - SMTP with STARTTLS or SSL and a user name and password (for Gmail or Outlook, an
  *app password*), plain-text body.

The settings live in ``configs/alerts.json``, which is **gitignored**: a webhook URL is itself a
secret (anyone holding it can post into the channel) and the email block holds a password.
``configs/alerts.example.json`` explains every key. Nothing this module returns or logs carries
a secret: webhook URLs are shown as scheme and host only, the password is never echoed, and
error messages are scrubbed of both before they are stored.

**Deciding and doing are separate**, as in `scheduler.py`. :func:`check` computes what is wrong
right now and sends nothing - the page's "Check now" calls it. :func:`dispatch` sends, and it
is the half that remembers:

* the same event (type + pitch / camera / slot - its ``key``) is not re-sent to the same
  channel within ``cooldown_hours`` (default 6), so a camera that stays dark does not message
  every five minutes. An anomaly and a daily summary are sent **once ever** per channel;
* delivery is tracked per channel, so if Slack accepted an alert and the mail server was down,
  only the email is retried on the next check;
* during the optional quiet hours nothing is sent and nothing is marked sent, so whatever is
  still true when they end goes out then.

State is ``data/alerts_state.json``; ``data/alerts_log.json`` keeps the last 200 deliveries
(time, event, channels, result) and checker errors. Both are written atomically - a temporary
file moved into place - so a crash mid-write never leaves a half file.

:func:`run_checker` is the loop for a background thread, and :func:`start_background_checker`
starts one when alerts are configured (the API app does this at start-up). An exception inside
a check is written to the alert log and the loop carries on: an alerting loop that dies quietly
is worse than none, because people stop watching once they believe they will be told.

The network is injectable: :data:`SENDERS` maps a channel type to its send function, and the
send functions call :data:`_urlopen` / :data:`_SMTP` / :data:`_SMTP_SSL`, which tests replace.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import smtplib
import sqlite3
import ssl
import tempfile
import threading
import time as _time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pitch_occupancy.config import CONFIGS_DIR, settings

__all__ = [
    "EVENT_TYPES", "Event", "Channel", "AlertConfig", "DeliveryError", "CONFIG_PATH",
    "load_config", "describe_config", "mask_url", "check", "preview", "dispatch", "send_test",
    "run_once", "run_checker", "start_background_checker", "stop_background_checker",
    "read_log", "send_webhook", "send_email",
]

log = logging.getLogger(__name__)

#: The settings file. Module-level so a test (or another deployment) can point it elsewhere.
CONFIG_PATH: Path = CONFIGS_DIR / "alerts.json"
#: The schedule read for the camera and worker checks. ``None`` means the scheduler's default.
SCHEDULE_PATH: Path | None = None

#: Every event type, with the sentence the page and the example config use for it.
EVENT_TYPES: dict[str, str] = {
    "anomaly": "A slot's records disagree with what the cameras saw (possible no-show, played "
               "but not booked, played but not checked in, booked while closed).",
    "camera_silent": "A camera that should be recording has sent no picture for longer than "
                     "the threshold.",
    "worker_down": "The recording program is not running while a slot is in progress.",
    "worker_error": "The recording program reports an error.",
    "daily_summary": "Once a day: yesterday's slots used, not used and to review, no-shows and "
                     "unbooked play.",
}

SEVERITIES = ("info", "warning", "serious")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}

#: The anomaly codes in the words a client would use.
ANOMALY_LABELS: dict[str, str] = {
    "NO_SHOW_OR_OVERRECORDED": "Possible no-show",
    "UNBOOKED_USAGE": "Played but not booked",
    "PLAYED_NOT_RECORDED": "Played but not checked in",
    "BLOCKED_SLOT_SOLD": "Booked while closed for maintenance",
}
_NOT_ANOMALIES = ("CONSISTENT", "NEEDS_REVIEW")

WEBHOOK_TIMEOUT_S = 10
SMTP_TIMEOUT_S = 20
#: Pause before the one webhook retry. Tests set it to 0.
RETRY_DELAY_S = 3.0
LOG_KEEP = 200
#: Once-ever keys (anomalies, summaries) are forgotten after this long. It is far longer than
#: the anomaly look-back, so a forgotten key can never be sent again.
ONCE_KEEP_DAYS = 60

_lock = threading.RLock()


class DeliveryError(RuntimeError):
    """A channel did not accept a message. The text never contains a secret."""


# --- configuration --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Channel:
    """One place alerts go: a webhook or a mailbox, with which events it wants."""

    name: str
    type: str  # "webhook" or "email"
    enabled: bool
    events: frozenset[str]
    min_severity: str
    settings: Mapping[str, Any] = field(repr=False)  # holds the URL or password

    def wants(self, event: Event) -> bool:
        return (self.enabled and event.type in self.events
                and _RANK.get(event.severity, 0) >= _RANK[self.min_severity])

    def describe(self) -> dict:
        """The channel as the page shows it: no URL path, no token, no password."""
        out: dict[str, Any] = {"name": self.name, "type": self.type, "enabled": self.enabled,
                               "events": sorted(self.events), "min_severity": self.min_severity}
        s = self.settings
        if self.type == "webhook":
            out["url"] = mask_url(str(s.get("url", "")))
        else:
            out.update(host=s.get("host"), port=_smtp_port(s), security=_security(s),
                       username=s.get("username") or None, sender=_sender(s),
                       to=_recipients(s), password_set=bool(s.get("password")))
        return out


@dataclass(frozen=True, slots=True)
class AlertConfig:
    channels: dict[str, Channel]
    camera_silent_minutes: int = 10
    cooldown_hours: float = 6.0
    check_interval_minutes: float = 5.0
    anomaly_lookback_hours: float = 48.0
    quiet_hours: tuple[time, time] | None = None
    daily_summary_at: time | None = None

    @property
    def enabled_channels(self) -> list[Channel]:
        return [c for c in self.channels.values() if c.enabled]


def _hhmm(value: Any, where: str) -> time:
    try:
        return time.fromisoformat(str(value).strip())
    except ValueError:
        raise ValueError(f"{where} must be a time like \"07:30\", not {value!r}") from None


def _positive(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, not {value!r}") from None
    if number <= 0:
        raise ValueError(f"{key} must be more than 0")
    return number


def _parse_channel(name: str, raw: Any) -> Channel:
    where = f"channel {name!r}"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be an object")
    kind = str(raw.get("type", "")).lower()
    if kind == "smtp":
        kind = "email"
    if kind not in ("webhook", "email"):
        raise ValueError(f"{where}: type must be \"webhook\" or \"email\", not {kind!r}")
    events = raw.get("events", list(EVENT_TYPES))
    if isinstance(events, str):
        events = [events]
    unknown = set(events) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"{where}: unknown event(s) {sorted(unknown)}; "
                         f"known: {sorted(EVENT_TYPES)}")
    min_sev = str(raw.get("min_severity", "info")).lower()
    if min_sev not in _RANK:
        raise ValueError(f"{where}: min_severity must be one of {list(SEVERITIES)}")
    settings_ = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    if kind == "webhook":
        parts = urlsplit(str(raw.get("url", "")))
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"{where}: url must be an http(s) address")
    else:
        if not raw.get("host"):
            raise ValueError(f"{where}: host is required (for example smtp.gmail.com)")
        if _security(raw) not in ("starttls", "ssl", "none"):
            raise ValueError(f"{where}: security must be \"starttls\", \"ssl\" or \"none\"")
        if _security(raw) == "none" and raw.get("password"):
            raise ValueError(f"{where}: a password over security \"none\" would be sent "
                             f"unencrypted; use \"starttls\" or \"ssl\"")
        if not _recipients(raw):
            raise ValueError(f"{where}: to must name at least one address")
        if not _sender(raw):
            raise ValueError(f"{where}: from (or username) is required")
    return Channel(name=name, type=kind, enabled=bool(raw.get("enabled", True)),
                   events=frozenset(events), min_severity=min_sev, settings=settings_)


def parse_config(raw: Any) -> AlertConfig:
    """Validate the decoded JSON of ``configs/alerts.json``. Keys starting with ``_`` are
    comments. Raises ``ValueError`` with a sentence a person can act on."""
    if not isinstance(raw, Mapping):
        raise ValueError("the file must hold one JSON object")
    channels_raw = raw.get("channels", {})
    if not isinstance(channels_raw, Mapping):
        raise ValueError("channels must be an object of name -> channel settings")
    channels = {str(n): _parse_channel(str(n), c) for n, c in channels_raw.items()
                if not str(n).startswith("_")}
    quiet = raw.get("quiet_hours")
    quiet_hours = None
    if quiet:
        if not isinstance(quiet, Mapping) or "start" not in quiet or "end" not in quiet:
            raise ValueError('quiet_hours must look like {"start": "22:00", "end": "07:00"}')
        quiet_hours = (_hhmm(quiet["start"], "quiet_hours.start"),
                       _hhmm(quiet["end"], "quiet_hours.end"))
    summary = raw.get("daily_summary")
    if isinstance(summary, Mapping):
        summary = summary.get("time") if summary.get("enabled", True) else None
    return AlertConfig(
        channels=channels,
        camera_silent_minutes=int(_positive(raw, "camera_silent_minutes", 10)),
        cooldown_hours=_positive(raw, "cooldown_hours", 6),
        check_interval_minutes=_positive(raw, "check_interval_minutes", 5),
        anomaly_lookback_hours=_positive(raw, "anomaly_lookback_hours", 48),
        quiet_hours=quiet_hours,
        daily_summary_at=_hhmm(summary, "daily_summary") if summary else None,
    )


def load_config(path: Path | None = None) -> AlertConfig | None:
    """The alert settings, or ``None`` when the file does not exist (alerts not set up).

    A file that exists but cannot be used raises ``ValueError`` - silently treating a typo as
    "no alerts" is how a facility finds out a month later that nothing was ever sent.
    """
    source = path or CONFIG_PATH
    if not source.is_file():
        return None
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source.name} is not valid JSON (line {exc.lineno}: "
                         f"{exc.msg})") from None
    return parse_config(raw)


def mask_url(url: str) -> str:
    """Scheme and host only: ``https://hooks.slack.com/...``. The path of a webhook URL *is*
    its password, and a user name or password before the host is dropped too."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "***"
    if not parts.scheme or not parts.hostname:
        return "***"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}/..."


def describe_config(path: Path | None = None) -> dict:
    """Everything the page shows about the settings, with every secret masked."""
    source = path or CONFIG_PATH
    out: dict[str, Any] = {"file": f"configs/{source.name}", "event_types": EVENT_TYPES,
                           "checker_running": checker_running()}
    try:
        cfg = load_config(source)
    except ValueError as exc:
        return {**out, "configured": True, "valid": False, "problem": str(exc)}
    if cfg is None:
        return {**out, "configured": False}
    return {
        **out, "configured": True, "valid": True,
        "channels": [c.describe() for c in cfg.channels.values()],
        "settings": {
            "camera_silent_minutes": cfg.camera_silent_minutes,
            "cooldown_hours": cfg.cooldown_hours,
            "check_interval_minutes": cfg.check_interval_minutes,
            "anomaly_lookback_hours": cfg.anomaly_lookback_hours,
            "quiet_hours": ({"start": f"{cfg.quiet_hours[0]:%H:%M}",
                             "end": f"{cfg.quiet_hours[1]:%H:%M}"}
                            if cfg.quiet_hours else None),
            "daily_summary": f"{cfg.daily_summary_at:%H:%M}" if cfg.daily_summary_at else None,
        },
    }


# --- events ---------------------------------------------------------------------------------


@dataclass(slots=True)
class Event:
    """One thing worth telling someone about.

    ``key`` identifies it for de-duplication: the same key is the same problem. ``once`` marks
    events that are sent once ever (an anomaly row, a day's summary) rather than again after
    the cooldown.
    """

    type: str
    key: str
    severity: str
    text: str
    time: str
    pitch: str | None = None
    slot: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    once: bool = False

    def payload(self) -> dict[str, Any]:
        """The webhook body."""
        return {"text": self.text, "event": self.type, "severity": self.severity,
                "pitch": self.pitch, "slot": self.slot, "time": self.time,
                "details": self.details}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now(now: datetime | None) -> datetime:
    return now or datetime.now()


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _local(stamp: Any) -> datetime | None:
    """An ISO timestamp as naive local time; one without an offset is taken as local."""
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed


def _open_db() -> sqlite3.Connection | None:
    """The database read-only, or ``None`` if there is none yet. Looking never creates it."""
    db = Path(settings.db_path)
    if not db.is_file():
        return None
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _anomaly_events(conn: sqlite3.Connection, now: datetime, cfg: AlertConfig) -> list[Event]:
    rows = conn.execute(
        """SELECT rc.reconciliation_id, rc.slot_id, rc.anomaly, rc.severity, rc.explanation,
                  rc.created_at, rs.field_id, rs.slot_date, rs.start_time, rs.end_time,
                  COALESCE(f.name, rs.field_id) AS field_name
           FROM reconciliations rc
           LEFT JOIN rental_slots rs ON rs.slot_id = rc.slot_id
           LEFT JOIN fields f ON f.field_id = rs.field_id
           WHERE rc.anomaly NOT IN (?, ?) AND rc.resolved_at IS NULL
           ORDER BY rc.created_at""", _NOT_ANOMALIES).fetchall()
    since = now - timedelta(hours=cfg.anomaly_lookback_hours)
    out = []
    for r in rows:
        created = _local(r["created_at"])
        if created is not None and created < since:
            continue  # older than the look-back: switching alerts on must not replay history
        severity = str(r["severity"] or "warning").lower()
        if severity not in ("warning", "serious"):
            continue
        label = ANOMALY_LABELS.get(r["anomaly"], str(r["anomaly"]).replace("_", " ").capitalize())
        pitch = r["field_name"] or r["field_id"] or "an unknown pitch"
        when = " ".join(x for x in (r["slot_date"], f"{r['start_time']}-{r['end_time']}"
                                    if r["start_time"] else None) if x)
        text = f"{label} on {pitch}" + (f", {when}" if when else "") + f": {r['explanation']}."
        out.append(Event(
            type="anomaly", key=f"anomaly:{r['slot_id']}:{r['anomaly']}", severity=severity,
            text=text, time=_stamp(created or now), pitch=r["field_id"], slot=r["slot_id"],
            once=True,
            details={"anomaly": r["anomaly"], "label": label, "explanation": r["explanation"],
                     "field_name": r["field_name"], "date": r["slot_date"],
                     "start": r["start_time"], "end": r["end_time"],
                     "reconciliation_id": r["reconciliation_id"],
                     "created_at": r["created_at"]}))
    return out


def _summary_event(conn: sqlite3.Connection, now: datetime) -> Event:
    from pitch_occupancy.reports import daily_summary

    day = now.date() - timedelta(days=1)
    summary = daily_summary(conn, start=day, end=day)
    t = summary["totals"]
    if not t["scheduled"]:
        text = (f"Daily summary for {day:%A %d %B}: no slots were recorded. If the pitches were "
                f"open, check that the recording program was running.")
    else:
        text = (f"Daily summary for {day:%A %d %B}: {t['scheduled']} slot(s) - {t['used']} used, "
                f"{t['not_used']} not used, {t['review']} to review"
                + (f", {t['not_checked']} not checked" if t["not_checked"] else "")
                + f"; {t['no_shows']} possible no-show(s), {t['unbooked_play']} played but not "
                f"booked; {t['hours_played']} hour(s) played.")
    return Event(type="daily_summary", key=f"daily_summary:{day.isoformat()}", severity="info",
                 text=text, time=_stamp(now), once=True,
                 details={"date": day.isoformat(), "totals": t, "pitches": summary["rows"]})


#: The status fields copied into an alert's details.
_STATUS_KEYS = ("state", "alive", "age_s", "updated_at", "slots_running", "next_slots")


def _read_status(notes: list[str], now: datetime) -> tuple[bool, dict | None]:
    """``(available, status)``. ``available`` is False when the status module is missing."""
    try:
        from pitch_occupancy.status import read_status
    except ImportError:
        notes.append("The recording program's status could not be read (the status module is "
                     "not installed yet), so the 'recording program stopped' check was skipped.")
        return False, None
    return True, read_status(now=now)


def _slot_label(slot: Any, day: date) -> tuple[str, str]:
    begins, ends = slot.window(day)
    return f"{begins:%H:%M}-{ends:%H:%M}", slot.slot_id(day)


def check(now: datetime | None = None, *, config: AlertConfig | None = None,
          notes: list[str] | None = None) -> list[Event]:
    """What is wrong right now. Reads the database, the camera health and the program status;
    sends nothing and remembers nothing.

    ``config`` supplies the thresholds; without one the defaults are used, so the page's
    "Check now" works before any channel is set up. ``notes``, when given, collects a sentence
    for every check that could not run (no database yet, no schedule, no status module).
    """
    now = _now(now)
    cfg = config or AlertConfig(channels={})
    notes = notes if notes is not None else []
    events: list[Event] = []

    conn = None
    try:
        conn = _open_db()
        if conn is None:
            notes.append("There is no database yet, so there are no anomalies or summaries "
                         "to send.")
    except sqlite3.Error as exc:
        notes.append(f"The database could not be opened ({exc}), so anomalies were not checked.")
    if conn is not None:
        try:
            events += _anomaly_events(conn, now, cfg)
            if cfg.daily_summary_at and now.time() >= cfg.daily_summary_at:
                events.append(_summary_event(conn, now))
        except sqlite3.Error as exc:
            notes.append(f"The database could not be read ({exc}), so anomalies were not "
                         f"checked.")
        finally:
            conn.close()

    from pitch_occupancy import scheduler
    try:
        schedule = scheduler.load_schedule(SCHEDULE_PATH)
    except (FileNotFoundError, ValueError, OSError) as exc:
        notes.append(f"The recording schedule could not be read ({exc}), so cameras and the "
                     f"recording program were not checked.")
        return events
    running = scheduler.due(schedule, now)

    program_alerted = False
    available, status = _read_status(notes, now)
    if available:
        state = str((status or {}).get("state") or "").lower()
        summary = {k: status.get(k) for k in _STATUS_KEYS if k in status} if status else None
        if status and state == "error":
            program_alerted = True
            errors = status.get("errors") or []
            detail = (errors[0].get("message") if errors and isinstance(errors[0], Mapping)
                      else None) or status.get("error") or status.get("message")
            events.append(Event(
                # One crash is one problem: a later crash has a new stamp and a new key.
                type="worker_error", key=f"worker_error:{status.get('updated_at') or ''}",
                severity="serious", time=_stamp(now),
                text="The recording program stopped with an error"
                     + (f": {detail}" if detail else "") + ". Slots are not being recorded "
                     "until it is restarted.",
                details={**(summary or {}), "error": detail}))
        elif running and not (status or {}).get("alive"):
            program_alerted = True
            first = running[0]
            span, slot_id = _slot_label(first, now.date())
            when = ("it has never reported" if status is None else
                    f"its last sign of life was {status.get('updated_at') or 'unknown'}")
            more = f" (and {len(running) - 1} other slot(s))" if len(running) > 1 else ""
            events.append(Event(
                type="worker_down", key="worker_down", severity="serious", time=_stamp(now),
                pitch=first.field_id, slot=slot_id,
                text=f"The recording program is not running, but the {span} slot on "
                     f"{first.field_id}{more} is in progress and is not being recorded "
                     f"({when}).",
                details={"slots_in_progress": [_slot_label(s, now.date())[1] for s in running],
                         "status": summary}))

    if running and not program_alerted:
        events += _camera_events(running, now, cfg)
    return events


def _camera_events(running: Iterable[Any], now: datetime, cfg: AlertConfig) -> list[Event]:
    """A camera in a running slot with no picture for longer than the threshold.

    Not raised in a slot's first ``camera_silent_minutes``: at 10:00 every camera's last
    picture is from the previous slot, which is not silence. Not raised while the recording
    program is down either - that one alert explains every dark camera.
    """
    from pitch_occupancy.cameras import recording_health

    limit = cfg.camera_silent_minutes
    out = []
    for slot in running:
        begins, _ = slot.window(now.date())
        into = (now - begins).total_seconds() / 60
        if into < limit:
            continue
        span, slot_id = _slot_label(slot, now.date())
        cams = [(slot.venue_id, c) for c in slot.cameras]
        health = recording_health(Path(settings.db_path), cams, now=now)
        for venue, cam in cams:
            h = health.get((venue, cam), {})
            since = h.get("minutes_since")
            if since is not None and since <= limit:
                continue
            if since is None or since > into:
                quiet_for = f"has not sent a picture since the {span} slot began {int(into)} " \
                            f"minutes ago"
            else:
                quiet_for = f"has sent no picture for {since} minutes"
            out.append(Event(
                type="camera_silent", key=f"camera_silent:{venue}:{cam}:{slot_id}",
                severity="warning", time=_stamp(now), pitch=slot.field_id, slot=slot_id,
                text=f"Camera {cam} on {slot.field_id} {quiet_for}, during a slot that should "
                     f"be recording. Check its power and network.",
                details={"venue": venue, "camera": cam, "minutes_since": since,
                         "last_recorded": h.get("last_recorded"), "threshold_minutes": limit,
                         "note": h.get("note")}))
    return out


# --- state, log and quiet hours -------------------------------------------------------------


def _state_path() -> Path:
    return Path(settings.data_dir) / "alerts_state.json"


def _log_path() -> Path:
    return Path(settings.data_dir) / "alerts_log.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data: Any) -> None:
    """Atomic: a temporary file beside the target, moved into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _append_log(entries: list[dict]) -> None:
    if not entries:
        return
    with _lock:
        current = _read_json(_log_path(), [])
        if not isinstance(current, list):
            current = []
        _write_json(_log_path(), (current + entries)[-LOG_KEEP:])


def read_log(limit: int = LOG_KEEP) -> list[dict]:
    """The alert log, newest first."""
    entries = _read_json(_log_path(), [])
    return list(reversed(entries))[:limit] if isinstance(entries, list) else []


def in_quiet_hours(cfg: AlertConfig, now: datetime) -> bool:
    if not cfg.quiet_hours:
        return False
    start, end = cfg.quiet_hours
    t = now.time()
    return start <= t < end if start <= end else (t >= start or t < end)


def _blocked(entry: Any, event: Event, cfg: AlertConfig, now: datetime) -> str | None:
    """Why ``event`` must not be sent again on a channel whose last send was ``entry``."""
    if not isinstance(entry, Mapping):
        return None
    if event.once:
        return "already sent"
    last = _local(entry.get("at"))
    if last is not None and now - last < timedelta(hours=cfg.cooldown_hours):
        until = last + timedelta(hours=cfg.cooldown_hours)
        return f"sent at {last:%H:%M}; not repeated before {until:%H:%M}"
    return None


def _prune(sent: dict, cfg: AlertConfig, now: datetime) -> dict:
    keep = {}
    for k, v in sent.items():
        at = _local(v.get("at")) if isinstance(v, Mapping) else None
        if at is None:
            continue
        horizon = (timedelta(days=ONCE_KEEP_DAYS) if v.get("once")
                   else timedelta(hours=cfg.cooldown_hours * 2))
        if now - at < horizon:
            keep[k] = v
    return keep


def preview(events: Iterable[Event], *, config: AlertConfig | None = None,
            now: datetime | None = None) -> list[dict]:
    """Each event with what :func:`dispatch` would do with it right now, per channel - so the
    page can show "would send to Slack; email already sent" without sending anything."""
    now = _now(now)
    cfg = config
    state = _read_json(_state_path(), {})
    sent = state.get("sent", {}) if isinstance(state, Mapping) else {}
    quiet = cfg is not None and in_quiet_hours(cfg, now)
    out = []
    for ev in events:
        channels = {}
        for ch in (cfg.channels.values() if cfg else []):
            if not ch.enabled:
                continue
            if not ch.wants(ev):
                channels[ch.name] = "not sent: this channel does not take this event"
            elif reason := _blocked(sent.get(f"{ch.name}|{ev.key}"), ev, cfg, now):
                channels[ch.name] = f"not sent: {reason}"
            elif quiet:
                channels[ch.name] = "held: quiet hours"
            else:
                channels[ch.name] = "would send"
        out.append({**ev.to_dict(), "channels": channels})
    return out


# --- sending --------------------------------------------------------------------------------

#: Replaced by tests. Nothing in this module opens a socket except through these three.
_urlopen = urllib.request.urlopen
_SMTP = smtplib.SMTP
_SMTP_SSL = smtplib.SMTP_SSL


def _scrub(message: str, *secrets: Any) -> str:
    for s in secrets:
        if s:
            message = message.replace(str(s), "***")
    return message


def send_webhook(channel: Channel, event: Event) -> str:
    """POST the event as JSON. 10-second timeout, one retry. Returns ``"HTTP 200"``-style
    text; raises :class:`DeliveryError` (with the URL scrubbed) when both attempts fail."""
    url = str(channel.settings["url"])
    body = json.dumps(event.payload(), default=str).encode("utf-8")
    problem = "no attempt made"
    for attempt in range(2):
        request = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "pitch-occupancy-alerts"})
        try:
            with _urlopen(request, timeout=WEBHOOK_TIMEOUT_S) as response:
                code = getattr(response, "status", None) or response.getcode()
            if 200 <= int(code) < 300:
                return f"HTTP {code}"
            problem = f"the webhook answered HTTP {code}"
        except urllib.error.HTTPError as exc:
            problem = f"the webhook answered HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 - every failure becomes a logged result
            reason = getattr(exc, "reason", None) or exc
            problem = f"could not reach the webhook ({type(exc).__name__}: {reason})"
        if attempt == 0 and RETRY_DELAY_S:
            _time.sleep(RETRY_DELAY_S)
    raise DeliveryError(_scrub(problem, url, urlsplit(url).path))


def _security(s: Mapping[str, Any]) -> str:
    return str(s.get("security", "starttls")).lower()


def _smtp_port(s: Mapping[str, Any]) -> int:
    return int(s.get("port") or (465 if _security(s) == "ssl" else 587))


def _recipients(s: Mapping[str, Any]) -> list[str]:
    to = s.get("to") or []
    if isinstance(to, str):
        to = [a.strip() for a in to.split(",")]
    return [str(a) for a in to if str(a).strip()]


def _sender(s: Mapping[str, Any]) -> str:
    return str(s.get("from") or s.get("username") or "")


def send_email(channel: Channel, event: Event) -> str:
    """Send the event as a plain-text email. Raises :class:`DeliveryError` on failure, with
    the password scrubbed from the message."""
    s = channel.settings
    password = s.get("password") or ""
    msg = EmailMessage()
    msg["Subject"] = f"Pitch alert: {event.text[:110]}" + ("..." if len(event.text) > 110
                                                           else "")
    msg["From"] = _sender(s)
    msg["To"] = ", ".join(_recipients(s))
    lines = [event.text, ""]
    for k, v in event.details.items():
        if v not in (None, "", [], {}) and not isinstance(v, (dict, list)):
            lines.append(f"{k.replace('_', ' ')}: {v}")
    lines += ["", f"Event: {event.type}  Severity: {event.severity}  Time: {event.time}",
              "Sent by Pitch Occupancy. Alert settings: configs/alerts.json on the recording "
              "computer."]
    msg.set_content("\n".join(lines))
    host, port, security = str(s["host"]), _smtp_port(s), _security(s)
    try:
        if security == "ssl":
            server = _SMTP_SSL(host, port, timeout=SMTP_TIMEOUT_S,
                               context=ssl.create_default_context())
        else:
            server = _SMTP(host, port, timeout=SMTP_TIMEOUT_S)
        try:
            if security == "starttls":
                server.starttls(context=ssl.create_default_context())
            if s.get("username"):
                server.login(str(s["username"]), str(password))
            server.send_message(msg)
        finally:
            # The message is already sent or already failed; a rude goodbye changes nothing.
            with contextlib.suppress(Exception):
                server.quit()
    except smtplib.SMTPAuthenticationError:
        raise DeliveryError("the mail server refused the user name or password (for Gmail "
                            "and Outlook, use an app password)") from None
    except Exception as exc:  # noqa: BLE001
        raise DeliveryError(_scrub(f"could not send the email ({type(exc).__name__}: {exc})",
                                   password)) from None
    return f"sent to {len(_recipients(s))} address(es)"


#: Channel type -> send function ``(channel, event) -> result text``. Tests replace entries.
SENDERS: dict[str, Callable[[Channel, Event], str]] = {
    "webhook": send_webhook, "email": send_email}

Sender = Callable[[Channel, Event], str]


def _deliver(channel: Channel, event: Event, sender: Sender | None) -> tuple[bool, str]:
    send = sender or SENDERS[channel.type]
    try:
        return True, str(send(channel, event) or "sent")
    except Exception as exc:  # noqa: BLE001 - a failed channel must not stop the others
        secrets = [channel.settings.get("password"), channel.settings.get("url")]
        return False, _scrub(str(exc) or type(exc).__name__, *secrets)


def dispatch(events: Iterable[Event], *, config: AlertConfig | None = None,
             now: datetime | None = None, sender: Sender | None = None) -> list[dict]:
    """Send each event to the channels that want it, skipping repeats. Returns the log entries
    written (one per event that was sent somewhere).

    ``config`` defaults to :func:`load_config`; with no config nothing is sent. ``sender``
    replaces every channel's send function (tests).
    """
    now = _now(now)
    cfg = config if config is not None else load_config()
    events = list(events)
    if cfg is None or not events or in_quiet_hours(cfg, now):
        return []
    entries = []
    with _lock:
        state = _read_json(_state_path(), {})
        sent = _prune(state.get("sent", {}) if isinstance(state, Mapping) else {}, cfg, now)
        for ev in events:
            results: dict[str, str] = {}
            ok_any = False
            for ch in cfg.channels.values():
                key = f"{ch.name}|{ev.key}"
                if not ch.wants(ev) or _blocked(sent.get(key), ev, cfg, now):
                    continue
                ok, detail = _deliver(ch, ev, sender)
                results[ch.name] = ("sent" if ok else "failed") + f": {detail}"
                if ok:
                    ok_any = True
                    sent[key] = {"at": _stamp(now), "once": ev.once}
            if results:
                entries.append({"time": _stamp(now), "event": ev.type, "key": ev.key,
                                "severity": ev.severity, "text": ev.text, "channels": results,
                                "ok": ok_any})
        _write_json(_state_path(), {"sent": sent, "updated_at": _stamp(now)})
        _append_log(entries)
    for e in entries:
        if not e["ok"]:
            log.warning("alert %s not delivered: %s", e["key"], e["channels"])
    return entries


def send_test(channel_name: str, *, config: AlertConfig | None = None,
              sender: Sender | None = None) -> dict:
    """Send a test message to one channel now, ignoring its event filter, the quiet hours and
    the cooldown. Raises ``KeyError`` for an unknown channel and ``ValueError`` when alerts are
    not configured."""
    cfg = config if config is not None else load_config()
    if cfg is None:
        raise ValueError("Alerts are not set up yet: create configs/alerts.json first.")
    channel = cfg.channels[channel_name]
    now = datetime.now()
    event = Event(type="test", key=f"test:{channel_name}", severity="info", time=_stamp(now),
                  text=f"Test alert from Pitch Occupancy: the '{channel_name}' channel works. "
                       f"Real alerts will arrive here like this one.",
                  details={"channel": channel_name})
    ok, detail = _deliver(channel, event, sender)
    entry = {"time": _stamp(now), "event": "test", "key": event.key, "severity": "info",
             "text": event.text, "channels": {channel_name: ("sent" if ok else "failed")
                                              + f": {detail}"}, "ok": ok}
    _append_log([entry])
    return {"ok": ok, "channel": channel_name, "result": detail}


# --- the background checker -----------------------------------------------------------------


def run_once(now: datetime | None = None, *, sender: Sender | None = None) -> list[dict]:
    """One tick: read the settings, check, send. Nothing happens when alerts are not set up."""
    cfg = load_config()
    if cfg is None or not cfg.enabled_channels:
        return []
    return dispatch(check(now, config=cfg), config=cfg, now=now, sender=sender)


def run_checker(stop_event: threading.Event, interval_s: float = 300, *,
                tick: Callable[[], Any] | None = None) -> None:
    """Check every ``interval_s`` seconds until ``stop_event`` is set.

    An exception inside a tick is written to the alert log and the loop carries on.
    """
    tick = tick or run_once
    while not stop_event.is_set():
        try:
            tick()
        except Exception as exc:  # noqa: BLE001 - the loop must outlive any one bad tick
            log.exception("alert check failed")
            try:
                _append_log([{"time": _stamp(datetime.now()), "event": "checker_error",
                              "key": "checker_error", "severity": "warning",
                              "text": f"The alert check failed: {type(exc).__name__}: {exc}",
                              "channels": {}, "ok": False}])
            except Exception:  # noqa: BLE001 - not even a full disk may stop the loop
                log.exception("could not write the alert log")
        stop_event.wait(interval_s)


_thread: threading.Thread | None = None
_stop: threading.Event | None = None


def checker_running() -> bool:
    return _thread is not None and _thread.is_alive()


def start_background_checker() -> threading.Thread | None:
    """Start the checker on a daemon thread, or return ``None`` when there is nothing to do:
    no ``configs/alerts.json``, no enabled channel, or a file that cannot be used (logged).

    Calling it again while a checker runs returns the running thread.
    """
    global _thread, _stop
    if checker_running():
        return _thread
    try:
        cfg = load_config()
    except ValueError as exc:
        log.warning("alerts not started: configs/%s has a problem: %s", CONFIG_PATH.name, exc)
        return None
    if cfg is None or not cfg.enabled_channels:
        return None
    _stop = threading.Event()
    _thread = threading.Thread(target=run_checker, name="pitch-alerts", daemon=True,
                               args=(_stop, cfg.check_interval_minutes * 60))
    _thread.start()
    return _thread


def stop_background_checker(timeout: float = 5.0) -> None:
    """Stop the checker started by :func:`start_background_checker`, if any."""
    global _thread, _stop
    if _stop is not None:
        _stop.set()
    if _thread is not None:
        _thread.join(timeout)
    _thread, _stop = None, None
