"""Other websites cannot drive the pages: host check, cross-site writes, sign-in throttling."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from pitch_occupancy import cameras
from pitch_occupancy.api import access
from pitch_occupancy.api.app import app
from pitch_occupancy.config import settings


@pytest.fixture
def open_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(settings, "access_password", SecretStr(""))
    monkeypatch.setattr(settings, "allowed_hosts", "")
    return TestClient(app)


def test_without_a_password_only_this_machines_names_are_answered(open_client) -> None:
    assert open_client.get("/health").status_code == 200
    assert open_client.get("/health", headers={"host": "localhost:8000"}).status_code == 200
    assert open_client.get("/health", headers={"host": "evil.example"}).status_code == 400


def test_configured_hosts_are_answered(open_client, monkeypatch) -> None:
    monkeypatch.setattr(settings, "allowed_hosts", "192.168.1.20, pitch.local")
    assert open_client.get("/health", headers={"host": "192.168.1.20:8000"}).status_code == 200
    assert open_client.get("/health", headers={"host": "evil.example"}).status_code == 400


def test_a_write_from_another_site_is_refused(open_client) -> None:
    for headers in ({"sec-fetch-site": "cross-site"},
                    {"origin": "https://evil.example"},
                    {"origin": "null"}):
        response = open_client.post("/api/v1/alerts/check", headers=headers)
        assert response.status_code == 403, headers


def test_a_write_from_the_page_itself_is_not_refused_by_the_guard(open_client) -> None:
    response = open_client.post("/api/v1/alerts/check",
                                headers={"origin": "http://testserver",
                                         "sec-fetch-site": "same-origin"})
    assert response.status_code != 403


def test_every_response_refuses_to_be_framed(open_client) -> None:
    headers = open_client.get("/health").headers
    assert headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"


def test_repeated_wrong_passwords_lock_the_address_out(monkeypatch) -> None:
    monkeypatch.setattr(settings, "access_password", SecretStr("a-long-password"))
    monkeypatch.setattr(access, "_failures", {})
    client = TestClient(app)
    for i in range(access.MAX_FAILURES):
        guess = {"Authorization": "Basic " + base64.b64encode(f"admin:guess{i}".encode()).decode()}
        assert client.get("/", headers=guess).status_code == 401
    wrong = {"Authorization": "Basic " + base64.b64encode(b"admin:nope").decode()}
    assert client.get("/", headers=wrong).status_code == 429
    right = {"Authorization": "Basic " + base64.b64encode(b"admin:a-long-password").decode()}
    assert client.get("/", headers=right).status_code == 429


def test_a_masked_password_is_never_sent_to_a_different_address() -> None:
    stored = "rtsp://admin:realpw@10.0.0.5:554/stream1"
    assert cameras.merge_password("rtsp://admin:***@10.0.0.5:554/stream2", stored) == \
        "rtsp://admin:realpw@10.0.0.5:554/stream2"
    for moved in ("rtsp://admin:***@attacker.example:554/stream1",
                  "rtsp://admin:***@10.0.0.5:8554/stream1",
                  "http://admin:***@10.0.0.5:554/stream1"):
        with pytest.raises(ValueError):
            cameras.merge_password(moved, stored)


def test_serve_refuses_a_short_password(monkeypatch) -> None:
    from typer.testing import CliRunner

    from pitch_occupancy import cli

    monkeypatch.setattr(settings, "access_password", SecretStr("short"))
    result = CliRunner().invoke(cli.app, ["serve", "--host", "127.0.0.1"])
    assert result.exit_code == 2 and "shorter than" in result.output


def test_a_write_from_another_port_of_this_machine_is_refused(open_client) -> None:
    for origin in ("http://localhost:3000", "http://testserver:8080"):
        r = open_client.post("/api/v1/alerts/check", headers={"origin": origin})
        assert r.status_code == 403, origin
    r = open_client.post("/api/v1/alerts/check", headers={"sec-fetch-site": "same-site"})
    assert r.status_code == 403


def test_a_stale_tab_repeating_one_old_password_does_not_lock_anyone_out(monkeypatch) -> None:
    monkeypatch.setattr(settings, "access_password", SecretStr("a-long-password"))
    monkeypatch.setattr(access, "_failures", {})
    client = TestClient(app)
    old = {"Authorization": "Basic " + base64.b64encode(b"admin:old-password").decode()}
    for _ in range(access.MAX_FAILURES * 3):
        assert client.get("/", headers=old).status_code == 401
    right = {"Authorization": "Basic " + base64.b64encode(b"admin:a-long-password").decode()}
    assert client.get("/", headers=right).status_code == 200
