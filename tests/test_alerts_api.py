"""The `/alerts` page and its API, on a fresh app with a temporary database and config.

Every send goes through a fake channel sender, so nothing leaves the machine, and every
response is checked for the webhook token and the email password.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pitch_occupancy import alerts
from pitch_occupancy.api import alerts_page
from pitch_occupancy.config import settings
from pitch_occupancy.db.schema import connect, initialise

HOOK = "https://hooks.slack.com/services/T000/B000/SECRETTOKEN123"
PASSWORD = "app-pass-XYZ987"
SECRETS = ("SECRETTOKEN123", PASSWORD)


@pytest.fixture
def world(tmp_path, monkeypatch):
    db = tmp_path / "db" / "test.db"
    conn = connect(db)
    initialise(conn)
    conn.execute("INSERT INTO venues VALUES ('venue_01', 'Venue One')")
    conn.execute("INSERT INTO fields VALUES ('field_01', 'venue_01', 'Pitch 1')")
    conn.execute("INSERT INTO rental_slots VALUES ('s1', 'field_01', '2026-10-10', '09:00', "
                 "'10:00')")
    conn.execute("INSERT INTO reconciliations (slot_id, anomaly, severity, explanation, "
                 "created_at) VALUES ('s1', 'UNBOOKED_USAGE', 'serious', 'played, no booking', "
                 "datetime('now', 'localtime'))")
    conn.commit()
    conn.close()
    monkeypatch.setattr(settings, "db_path", db)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(alerts, "CONFIG_PATH", tmp_path / "alerts.json")
    schedule = tmp_path / "schedule.json"
    schedule.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(alerts, "SCHEDULE_PATH", schedule)
    status = ModuleType("pitch_occupancy.status")
    status.read_status = lambda **kw: {"alive": True, "state": "running"}
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", status)
    sent: list[tuple[str, str]] = []

    def fake(channel, event):
        sent.append((channel.name, event.type))
        return "HTTP 200"

    monkeypatch.setitem(alerts.SENDERS, "webhook", fake)
    monkeypatch.setitem(alerts.SENDERS, "email", fake)
    app = FastAPI()
    app.include_router(alerts_page.router)
    return TestClient(app), tmp_path, sent


def _configure(tmp_path):
    (tmp_path / "alerts.json").write_text(json.dumps({"channels": {
        "slack": {"type": "webhook", "url": HOOK},
        "mail": {"type": "email", "host": "smtp.gmail.com", "username": "me@x.org",
                 "password": PASSWORD, "to": ["boss@x.org"], "events": ["daily_summary"]}}}),
        encoding="utf-8")


def _clean(response):
    for s in SECRETS:
        assert s not in response.text
    return response


def test_page_renders_with_examples(world):
    client, _, _ = world
    r = client.get("/alerts")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    for words in ("What you are told about", "Check now", "Recent alerts", "smtp.gmail.com",
                  "hooks.slack.com", "discord.com", "camera_silent", "Send a test alert"):
        assert words in r.text


def test_config_unconfigured_then_masked(world):
    client, tmp, _ = world
    assert client.get("/api/v1/alerts/config").json()["configured"] is False
    _configure(tmp)
    d = _clean(client.get("/api/v1/alerts/config")).json()
    assert d["valid"] is True
    slack = next(c for c in d["channels"] if c["name"] == "slack")
    assert slack["url"] == "https://hooks.slack.com/..."
    (tmp / "alerts.json").write_text("{oops", encoding="utf-8")
    d = client.get("/api/v1/alerts/config").json()
    assert d["configured"] is True and d["valid"] is False


def test_send_a_test_alert(world):
    client, tmp, sent = world
    assert client.post("/api/v1/alerts/test", json={"channel": "slack"}).status_code == 422
    _configure(tmp)
    r = _clean(client.post("/api/v1/alerts/test", json={"channel": "mail"}))
    assert r.status_code == 200 and r.json() == {"ok": True, "channel": "mail",
                                                  "result": "HTTP 200"}
    assert sent == [("mail", "test")]
    assert client.post("/api/v1/alerts/test", json={"channel": "nope"}).status_code == 404
    log = client.get("/api/v1/alerts/log").json()["entries"]
    assert log[0]["event"] == "test" and log[0]["channels"] == {"mail": "sent: HTTP 200"}


def test_failed_test_alert_is_reported_not_raised(world, monkeypatch):
    client, tmp, _ = world
    _configure(tmp)

    def broken(channel, event):
        raise RuntimeError(f"refused at {HOOK}")

    monkeypatch.setitem(alerts.SENDERS, "webhook", broken)
    r = _clean(client.post("/api/v1/alerts/test", json={"channel": "slack"}))
    assert r.status_code == 200 and r.json()["ok"] is False


def test_check_previews_then_sends_once(world):
    client, tmp, sent = world
    d = client.post("/api/v1/alerts/check").json()  # works before any setup
    assert [e["type"] for e in d["events"]] == ["anomaly"] and d["configured"] is False
    assert client.post("/api/v1/alerts/check?send=true").status_code == 422
    _configure(tmp)
    d = _clean(client.post("/api/v1/alerts/check")).json()
    channels = d["events"][0]["channels"]
    assert channels["slack"] == "would send" and channels["mail"].startswith("not sent")
    assert d["sent"] is None and sent == []
    d = _clean(client.post("/api/v1/alerts/check?send=true")).json()
    assert len(d["sent"]) == 1 and sent == [("slack", "anomaly")]
    d = client.post("/api/v1/alerts/check?send=true").json()
    assert d["sent"] == [] and d["events"][0]["channels"]["slack"] == "not sent: already sent"
    log = _clean(client.get("/api/v1/alerts/log")).json()["entries"]
    assert len(log) == 1 and log[0]["event"] == "anomaly"
