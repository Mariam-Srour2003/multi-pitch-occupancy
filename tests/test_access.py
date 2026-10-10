"""The password in front of the pages, and the refusal to serve a network without one."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from typer.testing import CliRunner

from pitch_occupancy import cli
from pitch_occupancy.api.access import credentials_ok
from pitch_occupancy.api.app import app
from pitch_occupancy.config import settings


def _basic(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def locked(monkeypatch) -> TestClient:
    monkeypatch.setattr(settings, "access_password", SecretStr("s3cret:with colon"))
    monkeypatch.setattr(settings, "access_user", "manager")
    return TestClient(app)


def test_without_a_password_everything_is_open(monkeypatch) -> None:
    monkeypatch.setattr(settings, "access_password", SecretStr(""))
    assert TestClient(app).get("/testing").status_code == 200


@pytest.mark.parametrize("path", ["/", "/testing", "/cameras", "/api/v1/cameras/registry",
                                  "/bookings", "/docs"])
def test_with_a_password_every_page_asks_for_it(locked, path) -> None:
    response = locked.get(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Basic")


def test_the_right_password_lets_you_in(locked) -> None:
    assert locked.get("/testing", headers=_basic("manager", "s3cret:with colon")).status_code \
        == 200


@pytest.mark.parametrize("user,password", [("manager", "wrong"), ("admin", "s3cret:with colon"),
                                           ("manager", "")])
def test_a_wrong_user_or_password_is_refused(locked, user, password) -> None:
    assert locked.get("/testing", headers=_basic(user, password)).status_code == 401


def test_a_mutating_route_is_refused_too(locked) -> None:
    response = locked.put("/api/v1/cameras", json={"venue": "v", "camera_id": "c",
                                                  "url": "rtsp://10.0.0.1/x"})
    assert response.status_code == 401


def test_health_stays_open_for_monitoring(locked) -> None:
    assert locked.get("/health").status_code == 200


@pytest.mark.parametrize("header", [None, "Bearer abc", "Basic !!!notbase64", "Basic " +
                                    base64.b64encode(b"no-colon").decode()])
def test_malformed_headers_are_refused(header) -> None:
    assert not credentials_ok(header, "admin", "pw")


def test_serve_refuses_a_network_address_without_a_password(monkeypatch) -> None:
    monkeypatch.setattr(settings, "access_password", SecretStr(""))
    started = []
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append(k))
    result = CliRunner().invoke(cli.app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 2 and not started
    assert CliRunner().invoke(cli.app, ["serve", "--host", "0.0.0.0", "--no-password"]
                              ).exit_code == 0
    assert started


def test_serve_on_this_machine_needs_no_password(monkeypatch) -> None:
    monkeypatch.setattr(settings, "access_password", SecretStr(""))
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    assert CliRunner().invoke(cli.app, ["serve", "--host", "127.0.0.1"]).exit_code == 0
