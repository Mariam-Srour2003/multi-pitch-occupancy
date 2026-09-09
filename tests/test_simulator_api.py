"""The simulator snapshot API (WP6-T1).

This endpoint serves frames of identifiable people over HTTP, so the tests that matter are the
ones about refusing. An auth check that passes when misconfigured is the same defect class as a
threshold that can never fire, and this project has found several of those - so "unset token"
is tested as *disabled*, explicitly, rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api import simulator
from pitch_occupancy.api.app import app
from pitch_occupancy.config import settings
from pitch_occupancy.frame_source import Frame

TOKEN = "test-token-value"


class FakeSlotSource:
    """A two-camera slot where minute 2 of camera_B is a gap."""

    def cameras(self) -> list[str]:
        return ["camera_A", "camera_B"]

    @property
    def n_minutes(self) -> int:
        return 3

    def read(self, camera_id: str, minute_index: int) -> Frame | None:
        if camera_id == "camera_B" and minute_index == 2:
            return None
        image = np.full((16, 16, 3), 128, np.uint8)
        return Frame(camera_id, minute_index, image, f"{camera_id}.mp4")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "simulator_token", TOKEN)
    simulator.set_source(simulator.RecordedSnapshots(FakeSlotSource()))
    yield TestClient(app)
    simulator.set_source(None)


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


# --- failing closed ---------------------------------------------------------------------


def test_an_unconfigured_token_disables_the_endpoint_rather_than_opening_it(monkeypatch) -> None:
    """The test this module exists for. An empty secret must not degrade to "no auth"."""
    monkeypatch.setattr(settings, "simulator_token", "")
    simulator.set_source(simulator.RecordedSnapshots(FakeSlotSource()))
    try:
        response = TestClient(app).get("/api/v1/cameras/camera_A/snapshot")
        assert response.status_code == 503
        assert "disabled" in response.json()["detail"]
    finally:
        simulator.set_source(None)


def test_a_missing_token_is_rejected(client) -> None:
    assert client.get("/api/v1/cameras/camera_A/snapshot").status_code == 401


def test_a_wrong_token_is_rejected(client) -> None:
    response = client.get(
        "/api/v1/cameras/camera_A/snapshot", headers={"Authorization": "Bearer nope"}
    )
    assert response.status_code == 401


def test_a_token_without_the_bearer_scheme_is_rejected(client) -> None:
    response = client.get(
        "/api/v1/cameras/camera_A/snapshot", headers={"Authorization": TOKEN}
    )
    assert response.status_code == 401


def test_a_prefix_of_the_token_is_rejected(client) -> None:
    """Guards against a comparison that stops at the shorter string."""
    response = client.get(
        "/api/v1/cameras/camera_A/snapshot",
        headers={"Authorization": f"Bearer {TOKEN[:-1]}"},
    )
    assert response.status_code == 401


def test_no_source_configured_is_a_503_not_an_empty_image(client) -> None:
    simulator.set_source(None)
    response = client.get("/api/v1/cameras/camera_A/snapshot", headers=auth())
    assert response.status_code == 503


# --- the acceptance criterion: curl returns a JPEG plus metadata ---------------------------


def test_a_snapshot_is_a_real_jpeg(client) -> None:
    response = client.get("/api/v1/cameras/camera_A/snapshot", headers=auth())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8", "not a JPEG SOI marker"
    assert response.content[-2:] == b"\xff\xd9", "not a JPEG EOI marker"


def test_the_jpeg_decodes_back_to_an_image(client) -> None:
    """A body that is JPEG-shaped but does not decode would pass the marker check above."""
    import cv2

    response = client.get("/api/v1/cameras/camera_A/snapshot", headers=auth())
    decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (16, 16, 3)


def test_metadata_rides_on_headers_so_the_body_stays_a_plain_image(client) -> None:
    response = client.get("/api/v1/cameras/camera_A/snapshot?minute=1", headers=auth())
    assert response.headers["X-Camera-Id"] == "camera_A"
    assert response.headers["X-Minute-Index"] == "1"
    assert response.headers["X-Mode"] == "simulation"
    assert response.headers["X-Captured-At"]


def test_the_json_format_returns_metadata_without_an_image(client) -> None:
    body = client.get(
        "/api/v1/cameras/camera_A/snapshot?format=json", headers=auth()
    ).json()
    assert body["camera_id"] == "camera_A"
    assert body["bytes"] > 0


def test_the_camera_list_is_served(client) -> None:
    assert client.get("/api/v1/cameras", headers=auth()).json() == {
        "cameras": ["camera_A", "camera_B"]
    }


# --- refusing to invent frames --------------------------------------------------------------


def test_live_mode_refuses_rather_than_falling_back_to_the_recording(client) -> None:
    """The worst failure this endpoint could have: a deployment check that passes against a
    file because the live path silently served footage."""
    response = client.get(
        "/api/v1/cameras/camera_A/snapshot?mode=live", headers=auth()
    )
    assert response.status_code == 501
    assert "does not fall back" in response.json()["detail"]


def test_a_gap_is_a_404_not_a_placeholder_image(client) -> None:
    """A gap must stay a gap through every layer. A substituted frame here would be a
    fabricated observation served with a 200."""
    response = client.get("/api/v1/cameras/camera_B/snapshot?minute=2", headers=auth())
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_an_unknown_camera_is_named(client) -> None:
    response = client.get("/api/v1/cameras/camera_Z/snapshot", headers=auth())
    assert response.status_code == 404
    assert "camera_Z" in response.json()["detail"]


def test_a_negative_minute_is_rejected_by_validation(client) -> None:
    assert client.get(
        "/api/v1/cameras/camera_A/snapshot?minute=-1", headers=auth()
    ).status_code == 422


def test_an_unknown_mode_is_rejected_by_validation(client) -> None:
    assert client.get(
        "/api/v1/cameras/camera_A/snapshot?mode=whatever", headers=auth()
    ).status_code == 422


# --- the token is never committed ------------------------------------------------------------


def test_the_default_token_is_empty() -> None:
    """A committed default would be a shared secret in a public repository, and every
    deployment that forgot to change it would be open."""
    from pitch_occupancy.config import Settings

    assert Settings(_env_file=None).simulator_token == ""
