"""The alert engine: what it finds, what it sends, and what it refuses to repeat or reveal.

Every test runs against a temporary database, data folder, schedule and config file, with
``settings.db_path`` and ``settings.data_dir`` pointed at them, so the real database is never
opened. The recording program's status module is replaced through ``sys.modules`` (present,
missing, or reporting a stopped or failed program), and no test opens a socket: webhooks go
through a fake ``urlopen`` and email through a fake SMTP class.
"""

from __future__ import annotations

import json
import smtplib
import sys
import threading
import urllib.error
from datetime import datetime, timedelta
from types import ModuleType

import pytest

from pitch_occupancy import alerts
from pitch_occupancy.config import settings
from pitch_occupancy.db.schema import connect, initialise

NOW = datetime(2026, 10, 10, 10, 30)
HOOK = "https://hooks.slack.com/services/T000/B000/SECRETTOKEN123"
PASSWORD = "app-pass-XYZ987"


def _status_module(value):
    mod = ModuleType("pitch_occupancy.status")
    mod.read_status = lambda **kw: value
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A temporary world: database, data folder, empty schedule, no config, a live worker."""
    db = tmp_path / "db" / "test.db"
    conn = connect(db)
    initialise(conn)
    conn.execute("INSERT INTO venues VALUES ('venue_01', 'Venue One')")
    conn.execute("INSERT INTO fields VALUES ('field_01', 'venue_01', 'Pitch 1')")
    conn.execute("INSERT INTO cameras (camera_id, field_id, side) VALUES ('camA', 'field_01', 'A')")
    conn.execute("INSERT INTO cameras (camera_id, field_id, side) VALUES ('camB', 'field_01', 'B')")
    conn.commit()
    monkeypatch.setattr(settings, "db_path", db)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(alerts, "CONFIG_PATH", tmp_path / "alerts.json")
    schedule = tmp_path / "schedule.json"
    schedule.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(alerts, "SCHEDULE_PATH", schedule)
    monkeypatch.setattr(alerts, "RETRY_DELAY_S", 0)
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status",
                        _status_module({"alive": True, "state": "running", "age_s": 5}))
    yield conn, tmp_path
    conn.close()


def _schedule(tmp_path, start="10:00", cameras=("camA", "camB")):
    (tmp_path / "schedule.json").write_text(json.dumps([{
        "venue_id": "venue_01", "field_id": "field_01", "start": start,
        "duration_minutes": 60, "cameras": list(cameras)}]), encoding="utf-8")


def _slot(conn, slot_id, day="2026-10-10", start="10:00", end="11:00"):
    conn.execute("INSERT OR IGNORE INTO rental_slots VALUES (?, 'field_01', ?, ?, ?)",
                 (slot_id, day, start, end))


def _recon(conn, slot_id, anomaly, severity, created, resolved=None):
    _slot(conn, slot_id, start=slot_id[-5:])
    conn.execute("INSERT INTO reconciliations (slot_id, anomaly, severity, explanation, "
                 "created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?)",
                 (slot_id, anomaly, severity, "the explanation", created, resolved))
    conn.commit()


def _sample(conn, cam, at, minute):
    _slot(conn, "venue_01_2026-10-10_1000")
    conn.execute("INSERT INTO frame_samples (slot_id, camera_id, captured_at, minute_index, "
                 "predicted, confidence) VALUES ('venue_01_2026-10-10_1000', ?, ?, ?, "
                 "'C1_EMPTY', 0.9)", (cam, at.isoformat(), minute))
    conn.commit()


def _cfg(**extra) -> alerts.AlertConfig:
    raw = {"channels": {"a": {"type": "webhook", "url": HOOK},
                        "b": {"type": "webhook", "url": "https://example.org/hook/b"}}}
    raw.update(extra)
    return alerts.parse_config(raw)


def _event(key="camera_silent:x", once=False, type_="camera_silent", severity="warning"):
    return alerts.Event(type=type_, key=key, severity=severity, text="t", time="x", once=once)


class Recorder:
    def __init__(self, fail: set[str] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail or set()

    def __call__(self, channel, event):
        self.calls.append((channel.name, event.key))
        if channel.name in self.fail:
            raise RuntimeError("down")
        return "ok"


# --- 1. anomalies ---------------------------------------------------------------------------


def test_anomalies_skip_agreements_reviews_resolved_and_old_rows(env):
    conn, _ = env
    recent = (NOW - timedelta(hours=1)).isoformat()
    _recon(conn, "s_12:00", "NO_SHOW_OR_OVERRECORDED", "warning", recent)
    _recon(conn, "s_13:00", "UNBOOKED_USAGE", "serious", "2026-10-10T08:00:00+00:00")
    _recon(conn, "s_14:00", "CONSISTENT", "none", recent)
    _recon(conn, "s_15:00", "NEEDS_REVIEW", "info", recent)
    _recon(conn, "s_16:00", "PLAYED_NOT_RECORDED", "warning", recent, resolved=recent)
    _recon(conn, "s_17:00", "BLOCKED_SLOT_SOLD", "serious", "2026-09-01T10:00:00")
    events = alerts.check(NOW)
    assert [e.details["anomaly"] for e in events] == ["UNBOOKED_USAGE",
                                                       "NO_SHOW_OR_OVERRECORDED"] or \
        sorted(e.details["anomaly"] for e in events) == ["NO_SHOW_OR_OVERRECORDED",
                                                         "UNBOOKED_USAGE"]
    no_show = next(e for e in events if e.details["anomaly"] == "NO_SHOW_OR_OVERRECORDED")
    assert no_show.type == "anomaly" and no_show.once and no_show.severity == "warning"
    assert no_show.key == "anomaly:s_12:00:NO_SHOW_OR_OVERRECORDED"
    assert no_show.text.startswith("Possible no-show on Pitch 1")
    assert no_show.pitch == "field_01" and no_show.slot == "s_12:00"


def test_no_database_is_a_note_not_an_error(env, monkeypatch):
    _, tmp = env
    monkeypatch.setattr(settings, "db_path", tmp / "missing" / "none.db")
    notes: list[str] = []
    assert alerts.check(NOW, notes=notes) == []
    assert any("no database" in n for n in notes)
    assert not (tmp / "missing").exists()  # looking never creates it


# --- 2. cameras -----------------------------------------------------------------------------


def test_silent_camera_in_a_running_slot(env):
    conn, tmp = env
    _schedule(tmp)
    _sample(conn, "camA", NOW - timedelta(minutes=2), 28)
    _sample(conn, "camB", NOW - timedelta(minutes=25), 5)
    events = alerts.check(NOW)
    assert [e.key for e in events] == ["camera_silent:venue_01:camB:venue_01_2026-10-10_1000"]
    assert events[0].details["minutes_since"] == 25
    assert "25 minutes" in events[0].text


def test_camera_never_recorded_and_threshold(env):
    conn, tmp = env
    _schedule(tmp, cameras=("camA",))
    events = alerts.check(NOW, config=_cfg(camera_silent_minutes=20))
    assert len(events) == 1 and "has not sent a picture since" in events[0].text
    # Inside the slot's first minutes nothing is silent yet.
    assert alerts.check(datetime(2026, 10, 10, 10, 5)) == []
    # Outside any slot nothing is expected.
    assert alerts.check(datetime(2026, 10, 10, 12, 0)) == []


# --- 3. the recording program ---------------------------------------------------------------


def test_worker_not_alive_during_a_slot(env, monkeypatch):
    _, tmp = env
    _schedule(tmp)
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", _status_module(
        {"alive": False, "state": "running", "age_s": 900, "updated_at": "2026-10-10T10:15"}))
    events = alerts.check(NOW)
    # One alert explains every dark camera: no camera_silent alongside it.
    assert [e.type for e in events] == ["worker_down"]
    assert events[0].severity == "serious" and "2026-10-10T10:15" in events[0].text
    # Not alive outside any slot is not news.
    assert alerts.check(datetime(2026, 10, 10, 15, 0)) == []


def test_worker_status_file_absent_during_a_slot(env, monkeypatch):
    _, tmp = env
    _schedule(tmp)
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", _status_module(None))
    events = alerts.check(NOW)
    assert [e.type for e in events] == ["worker_down"] and "never reported" in events[0].text


def test_worker_error_state_is_sent_even_between_slots(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", _status_module(
        {"alive": False, "state": "error", "updated_at": "2026-10-10T14:00:00+02:00",
         "errors": [{"at": "x", "message": "RuntimeError: camera stream lost"}]}))
    events = alerts.check(datetime(2026, 10, 10, 15, 0))
    assert [e.type for e in events] == ["worker_error"]
    assert "camera stream lost" in events[0].text
    assert events[0].key == "worker_error:2026-10-10T14:00:00+02:00"


def test_worker_error_during_a_slot_is_one_alert(env, monkeypatch):
    _, tmp = env
    _schedule(tmp)
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", _status_module(
        {"alive": False, "state": "error", "updated_at": "t", "errors": []}))
    assert [e.type for e in alerts.check(NOW)] == ["worker_error"]


def test_missing_status_module_skips_that_check_and_says_so(env, monkeypatch):
    conn, tmp = env
    _schedule(tmp, cameras=("camA",))
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", None)  # import fails
    notes: list[str] = []
    events = alerts.check(NOW, notes=notes)
    assert [e.type for e in events] == ["camera_silent"]  # the camera check still ran
    assert any("skipped" in n for n in notes)


# --- 4. daily summary -----------------------------------------------------------------------


def test_daily_summary_after_its_time_only_and_once(env):
    conn, _ = env
    for start, status in (("10:00", "USED"), ("11:00", "NOTUSED"), ("12:00", "REVIEW")):
        sid = f"venue_01_2026-10-09_{start.replace(':', '')}"
        _slot(conn, sid, day="2026-10-09", start=start, end=f"{int(start[:2]) + 1}:00")
        conn.execute("INSERT INTO slot_evaluations (slot_id, status, reason, play_ratio, "
                     "empty_ratio, maintenance_ratio, n_samples, mean_confidence, "
                     "evaluated_at, model_key) VALUES (?, ?, 'r', 0.5, 0.5, 0, 60, 0.9, 'x', "
                     "'m')", (sid, status))
    conn.commit()
    cfg = _cfg(daily_summary="07:30")
    assert alerts.check(datetime(2026, 10, 10, 7, 0), config=cfg) == []
    events = alerts.check(datetime(2026, 10, 10, 7, 45), config=cfg)
    assert [e.key for e in events] == ["daily_summary:2026-10-09"]
    assert "3 slot(s) - 1 used, 1 not used, 1 to review" in events[0].text
    rec = Recorder()
    alerts.dispatch(events, config=cfg, now=datetime(2026, 10, 10, 7, 45), sender=rec)
    later = alerts.check(datetime(2026, 10, 10, 18, 0), config=cfg)
    assert alerts.dispatch(later, config=cfg, now=datetime(2026, 10, 10, 18, 0),
                           sender=rec) == []
    assert len(rec.calls) == 2  # once per channel, never again


# --- 5. dispatch: de-duplication, cooldown, quiet hours, log --------------------------------


def test_cooldown_and_once(env):
    cfg, rec = _cfg(), Recorder()
    events = [_event("camera_silent:x"), _event("anomaly:1", once=True, type_="anomaly")]
    first = alerts.dispatch(events, config=cfg, now=NOW, sender=rec)
    assert len(first) == 2 and len(rec.calls) == 4
    assert alerts.dispatch(events, config=cfg, now=NOW + timedelta(hours=5), sender=rec) == []
    again = alerts.dispatch(events, config=cfg, now=NOW + timedelta(hours=7), sender=rec)
    assert [e["key"] for e in again] == ["camera_silent:x"]  # the anomaly never repeats
    log = alerts.read_log()
    assert len(log) == 3 and log[0]["key"] == "camera_silent:x"
    assert log[0]["channels"] == {"a": "sent: ok", "b": "sent: ok"}


def test_failed_channel_is_retried_alone(env):
    cfg = _cfg()
    alerts.dispatch([_event()], config=cfg, now=NOW, sender=Recorder(fail={"b"}))
    assert alerts.read_log()[0]["channels"]["b"] == "failed: down"
    rec = Recorder()
    alerts.dispatch([_event()], config=cfg, now=NOW + timedelta(minutes=5), sender=rec)
    assert rec.calls == [("b", "camera_silent:x")]


def test_channel_event_filter_and_severity(env):
    cfg = alerts.parse_config({"channels": {
        "only_cams": {"type": "webhook", "url": HOOK, "events": ["camera_silent"]},
        "serious": {"type": "webhook", "url": HOOK, "min_severity": "serious"},
        "off": {"type": "webhook", "url": HOOK, "enabled": False}}})
    rec = Recorder()
    alerts.dispatch([_event(), _event("worker_down", type_="worker_down", severity="serious")],
                    config=cfg, now=NOW, sender=rec)
    assert sorted(rec.calls) == [("only_cams", "camera_silent:x"), ("serious", "worker_down")]


def test_quiet_hours_hold_without_marking_sent(env):
    cfg = _cfg(quiet_hours={"start": "22:00", "end": "07:00"})
    assert alerts.in_quiet_hours(cfg, datetime(2026, 10, 10, 23, 0))
    assert alerts.in_quiet_hours(cfg, datetime(2026, 10, 10, 6, 59))
    assert not alerts.in_quiet_hours(cfg, datetime(2026, 10, 10, 7, 0))
    rec = Recorder()
    assert alerts.dispatch([_event()], config=cfg, now=datetime(2026, 10, 10, 23, 0),
                           sender=rec) == []
    assert rec.calls == []
    preview = alerts.preview([_event()], config=cfg, now=datetime(2026, 10, 10, 23, 0))
    assert preview[0]["channels"]["a"] == "held: quiet hours"
    alerts.dispatch([_event()], config=cfg, now=datetime(2026, 10, 10, 7, 30), sender=rec)
    assert len(rec.calls) == 2
    daytime = alerts.parse_config({"channels": {}, "quiet_hours": {"start": "12:00",
                                                                   "end": "14:00"}})
    assert alerts.in_quiet_hours(daytime, datetime(2026, 10, 10, 13, 0))
    assert not alerts.in_quiet_hours(daytime, datetime(2026, 10, 10, 15, 0))


def test_state_is_written_atomically_as_json(env):
    _, tmp = env
    alerts.dispatch([_event()], config=_cfg(), now=NOW, sender=Recorder())
    state = json.loads((tmp / "data" / "alerts_state.json").read_text(encoding="utf-8"))
    assert set(state["sent"]) == {"a|camera_silent:x", "b|camera_silent:x"}
    assert not list((tmp / "data").glob("*.tmp"))


def test_log_keeps_the_last_200(env):
    cfg = alerts.parse_config({"channels": {"a": {"type": "webhook", "url": HOOK}}})
    events = [_event(f"k{i}") for i in range(205)]
    alerts.dispatch(events, config=cfg, now=NOW, sender=Recorder())
    log = alerts.read_log()
    assert len(log) == 200 and log[0]["key"] == "k204"


# --- 6. configuration and masking -----------------------------------------------------------


def test_masking_hides_paths_tokens_and_passwords(env):
    _, tmp = env
    assert alerts.mask_url(HOOK) == "https://hooks.slack.com/..."
    assert alerts.mask_url("https://user:pw@example.org:8443/x?token=1") == \
        "https://example.org:8443/..."
    (tmp / "alerts.json").write_text(json.dumps({"channels": {
        "slack": {"type": "webhook", "url": HOOK},
        "mail": {"type": "email", "host": "smtp.gmail.com", "username": "me@x.org",
                 "password": PASSWORD, "to": "a@x.org, b@x.org"}}}), encoding="utf-8")
    described = alerts.describe_config()
    text = json.dumps(described)
    assert "SECRETTOKEN123" not in text and PASSWORD not in text
    mail = next(c for c in described["channels"] if c["name"] == "mail")
    assert mail["password_set"] is True and mail["port"] == 587 and mail["to"] == [
        "a@x.org", "b@x.org"]
    cfg = alerts.load_config()
    assert PASSWORD not in repr(cfg) and "SECRETTOKEN123" not in repr(cfg)


@pytest.mark.parametrize("raw, words", [
    ("{not json", "not valid JSON"),
    ('{"channels": {"x": {"type": "fax"}}}', "type must be"),
    ('{"channels": {"x": {"type": "webhook", "url": "ftp://a"}}}', "http(s)"),
    ('{"channels": {"x": {"type": "webhook", "url": "https://a", "events": ["nope"]}}}',
     "unknown event"),
    ('{"channels": {"x": {"type": "email", "host": "h", "username": "u"}}}', "to must"),
    ('{"channels": {}, "daily_summary": "7.30pm"}', "daily_summary"),
    ('{"channels": {}, "camera_silent_minutes": 0}', "more than 0"),
])
def test_bad_config_is_refused_with_a_reason(env, raw, words):
    _, tmp = env
    (tmp / "alerts.json").write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=words.replace("(", r"\(").replace(")", r"\)")):
        alerts.load_config()
    assert alerts.describe_config()["valid"] is False


def test_the_example_config_is_valid():
    from pitch_occupancy.config import CONFIGS_DIR
    cfg = alerts.load_config(CONFIGS_DIR / "alerts.example.json")
    assert cfg is not None and set(cfg.channels) == {"slack", "office_email"}
    assert cfg.daily_summary_at is not None and cfg.quiet_hours is not None


# --- 7. channels ----------------------------------------------------------------------------


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def getcode(self):
        return self.status


def test_webhook_payload_shape(env, monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(alerts, "_urlopen", fake_urlopen)
    channel = _cfg().channels["a"]
    ev = alerts.Event(type="camera_silent", key="k", severity="warning", text="Camera B quiet",
                      time="2026-10-10T10:30:00", pitch="field_01", slot="s1",
                      details={"camera": "camB"})
    assert alerts.send_webhook(channel, ev) == "HTTP 200"
    request, timeout = seen[0]
    assert timeout == 10 and request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    body = json.loads(request.data)
    assert body == {"text": "Camera B quiet", "event": "camera_silent", "severity": "warning",
                    "pitch": "field_01", "slot": "s1", "time": "2026-10-10T10:30:00",
                    "details": {"camera": "camB"}}


def test_webhook_retries_once_and_scrubs_the_url(env, monkeypatch):
    calls = []

    def flaky(request, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("timed out")
        return FakeResponse()

    monkeypatch.setattr(alerts, "_urlopen", flaky)
    channel = _cfg().channels["a"]
    assert alerts.send_webhook(channel, _event()) == "HTTP 200" and len(calls) == 2

    def broken(request, timeout):
        raise urllib.error.URLError(f"cannot reach {HOOK}")

    monkeypatch.setattr(alerts, "_urlopen", broken)
    with pytest.raises(alerts.DeliveryError) as info:
        alerts.send_webhook(channel, _event())
    assert "SECRETTOKEN123" not in str(info.value)


class FakeSMTP:
    instances: list[FakeSMTP] = []
    refuse_login = False

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.actions, self.sent = host, port, [], []
        FakeSMTP.instances.append(self)

    def starttls(self, context=None):
        self.actions.append("starttls")

    def login(self, user, password):
        if self.refuse_login:
            raise smtplib.SMTPAuthenticationError(535, f"bad {password}".encode())
        self.actions.append(("login", user, password))

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        self.actions.append("quit")


def _mail(security="starttls"):
    return alerts.parse_config({"channels": {"mail": {
        "type": "email", "host": "smtp.example.org", "security": security,
        "username": "me@example.org", "password": PASSWORD, "to": ["boss@example.org"]}}}
    ).channels["mail"]


def test_email_starttls_and_ssl(env, monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(alerts, "_SMTP", FakeSMTP)
    monkeypatch.setattr(alerts, "_SMTP_SSL", FakeSMTP)
    ev = alerts.Event(type="worker_down", key="k", severity="serious", text="Program stopped",
                      time="t", details={"camera": "camB"})
    assert "1 address" in alerts.send_email(_mail(), ev)
    smtp = FakeSMTP.instances[-1]
    assert smtp.port == 587 and smtp.actions[0] == "starttls"
    assert smtp.actions[1] == ("login", "me@example.org", PASSWORD)
    msg = smtp.sent[0]
    assert msg["To"] == "boss@example.org" and "Program stopped" in msg["Subject"]
    assert "Program stopped" in msg.get_content() and "camera: camB" in msg.get_content()
    alerts.send_email(_mail("ssl"), ev)
    assert FakeSMTP.instances[-1].port == 465
    assert "starttls" not in FakeSMTP.instances[-1].actions


def test_email_login_failure_never_echoes_the_password(env, monkeypatch):
    monkeypatch.setattr(alerts, "_SMTP", FakeSMTP)
    monkeypatch.setattr(FakeSMTP, "refuse_login", True)
    with pytest.raises(alerts.DeliveryError) as info:
        alerts.send_email(_mail(), _event())
    assert PASSWORD not in str(info.value) and "app password" in str(info.value)


def test_dispatch_routes_by_type_through_senders(env, monkeypatch):
    seen = []
    monkeypatch.setitem(alerts.SENDERS, "webhook", lambda ch, ev: seen.append(ch.name) or "ok")
    alerts.dispatch([_event()], config=_cfg(), now=NOW)
    assert sorted(seen) == ["a", "b"]


# --- 8. the background checker --------------------------------------------------------------


def test_checker_loop_survives_an_exception(env):
    stop = threading.Event()
    ticks = []

    def tick():
        ticks.append(1)
        if len(ticks) == 1:
            raise RuntimeError("database locked")
        stop.set()

    alerts.run_checker(stop, interval_s=0, tick=tick)
    assert len(ticks) == 2
    entry = alerts.read_log()[0]
    assert entry["event"] == "checker_error" and "database locked" in entry["text"]


def test_background_checker_starts_only_when_configured(env, monkeypatch):
    _, tmp = env
    assert alerts.start_background_checker() is None  # no file
    (tmp / "alerts.json").write_text(json.dumps({"channels": {
        "a": {"type": "webhook", "url": HOOK, "enabled": False}}}), encoding="utf-8")
    assert alerts.start_background_checker() is None  # nothing enabled
    (tmp / "alerts.json").write_text("{broken", encoding="utf-8")
    assert alerts.start_background_checker() is None  # unusable file, logged
    (tmp / "alerts.json").write_text(json.dumps({"channels": {
        "a": {"type": "webhook", "url": HOOK}}}), encoding="utf-8")
    monkeypatch.setitem(alerts.SENDERS, "webhook", lambda ch, ev: "ok")
    try:
        thread = alerts.start_background_checker()
        assert thread is not None and thread.is_alive() and alerts.checker_running()
        assert alerts.start_background_checker() is thread
    finally:
        alerts.stop_background_checker()
    assert not alerts.checker_running()


def test_run_once_does_nothing_without_config(env):
    assert alerts.run_once(NOW) == []


def test_real_status_file_fresh_stale_and_missing(env, monkeypatch):
    """Against the real status module when it is installed: the file in the temp data folder."""
    _, tmp = env
    monkeypatch.delitem(sys.modules, "pitch_occupancy.status")
    status = pytest.importorskip("pitch_occupancy.status")
    _schedule(tmp, cameras=("camA",))
    assert [e.type for e in alerts.check(NOW)] == ["worker_down"]  # no file: never reported
    status.write_status({"state": "running", "updated_at": (NOW - timedelta(minutes=1))
                         .astimezone().isoformat()})
    assert [e.type for e in alerts.check(NOW)] == ["camera_silent"]  # alive; camera is dark
    status.write_status({"state": "running", "updated_at": (NOW - timedelta(hours=2))
                         .astimezone().isoformat()})
    assert [e.type for e in alerts.check(NOW)] == ["worker_down"]  # stale stamp: not alive
