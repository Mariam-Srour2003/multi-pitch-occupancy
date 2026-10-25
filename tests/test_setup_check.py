"""The "Getting started" checklist, from configuration files in a temporary folder.

Every config path the checklist reads is pointed at ``tmp_path``, so the tests describe this
machine's states without depending on (or touching) the real ``configs/``. What is pinned
beyond each state: a broken file is a warning step carrying its error, never an exception, and
the endpoint answers 200 whatever the configuration.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from pitch_occupancy import booking_sources, cameras, scheduler, setup_check
from pitch_occupancy.api import setup_api
from pitch_occupancy.config import settings
from pitch_occupancy.vision import detector, roi

SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Path:
    """An empty configs folder that every check reads from."""
    folder = tmp_path / "configs"
    folder.mkdir()
    monkeypatch.setattr(cameras, "CONFIG_PATH", folder / "cameras.json")
    monkeypatch.setattr(roi, "STORE", folder / "roi.json")
    monkeypatch.setattr(roi, "DERIVED_STORE", folder / "roi_derived.json")
    monkeypatch.setattr(scheduler, "DEFAULT_SCHEDULE", folder / "slots_schedule.json")
    monkeypatch.setattr(booking_sources, "DEFAULT_SOURCE_CONFIG",
                        folder / "bookings_source.json")
    monkeypatch.setattr(detector, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(setup_check, "ALERTS_PATH", folder / "alerts.json")
    monkeypatch.setattr(settings, "access_password", SecretStr(""))
    monkeypatch.setattr(settings, "api_host", "127.0.0.1")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "db_path", tmp_path / "data" / "db" / "x.db")
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", None)  # -> ImportError
    return folder


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")


def _status(monkeypatch, value) -> None:
    fake = types.ModuleType("pitch_occupancy.status")
    fake.read_status = lambda: value
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", fake)


def _steps() -> dict[str, setup_check.Step]:
    return {s.key: s for s in setup_check.checklist()}


def _complete(cfg: Path, tmp_path: Path, monkeypatch) -> None:
    _write(cfg / "cameras.json", {
        "venue_01": {"camera_A": "rtsp://u:secret@10.0.0.5/a", "camera_B": "usb:0"},
        "venue_02": {"camera_C": "rtsp://10.0.0.6/c", "camera_D": "rtsp://10.0.0.7/d"},
    })
    _write(cfg / "roi.json", {f"{v}/{c}": SQUARE for v, c in (
        ("venue_01", "camera_A"), ("venue_01", "camera_B"),
        ("venue_02", "camera_C"), ("venue_02", "camera_D"))})
    _write(cfg / "slots_schedule.json", [
        {"venue_id": "venue_01", "start": "10:00", "duration_minutes": 60,
         "cameras": ["camera_A", "camera_B"]},
        {"venue_id": "venue_02", "start": "18:00", "duration_minutes": 90,
         "cameras": ["camera_C", "camera_D"]},
    ])
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "yolov8n.pt").write_bytes(b"not really weights")
    _status(monkeypatch, {"alive": True, "age_s": 12})


def test_a_fresh_install_lists_what_is_missing(cfg) -> None:
    steps = _steps()
    assert list(steps) == ["cameras", "boundaries", "schedule", "detector", "bookings",
                           "password", "worker", "playgrounds", "assist", "alerts"]
    assert steps["cameras"].state == "todo" and steps["cameras"].link == "/cameras"
    assert steps["boundaries"].state == "todo" and steps["boundaries"].link == "/roi"
    assert steps["schedule"].state == "todo"
    assert steps["detector"].state == "todo"
    assert steps["detector"].hint == "uv run pitch fetch-weights"
    assert steps["bookings"].state == "optional" and not steps["bookings"].required
    assert steps["password"].state == "optional" and not steps["password"].required
    assert steps["worker"].state == "unknown"
    assert steps["alerts"].state == "optional"
    assert steps["playgrounds"].state == "optional" and not steps["playgrounds"].required
    assert steps["assist"].state == "optional" and not steps["assist"].required
    assert setup_check.summary()["required_done"] is False


def test_a_complete_setup_is_all_done(cfg, tmp_path, monkeypatch) -> None:
    _complete(cfg, tmp_path, monkeypatch)
    steps = _steps()
    assert steps["cameras"].detail == "4 cameras for 2 venues."
    assert steps["boundaries"].state == "done"
    assert steps["schedule"].state == "done", steps["schedule"].detail
    assert steps["detector"].state == "done"
    assert steps["worker"].state == "done" and "12 seconds" in steps["worker"].detail
    out = setup_check.summary()
    assert out["required_done"] is True
    # A camera's password never reaches the checklist.
    assert "secret" not in json.dumps(out)


def test_a_camera_without_a_boundary_is_named(cfg, tmp_path, monkeypatch) -> None:
    _complete(cfg, tmp_path, monkeypatch)
    _write(cfg / "roi.json", {"venue_01/camera_A": SQUARE})
    step = _steps()["boundaries"]
    assert step.state == "todo"
    assert "camera_B (venue_01)" in step.detail and "camera_C (venue_02)" in step.detail


def test_schedule_problems(cfg, tmp_path, monkeypatch) -> None:
    _complete(cfg, tmp_path, monkeypatch)
    _write(cfg / "slots_schedule.json", [
        {"venue_id": "venue_01", "start": "10:00", "duration_minutes": 60,
         "cameras": ["camera_A", "camera_Z"]},
    ])
    step = _steps()["schedule"]
    assert step.state == "warning" and "camera_Z (venue_01)" in step.detail

    _write(cfg / "slots_schedule.json", [
        {"venue_id": "venue_01", "start": "10:00", "duration_minutes": 60,
         "cameras": ["camera_A"]},
    ])
    step = _steps()["schedule"]
    assert step.state == "todo" and "venue_02" in step.detail


def test_optional_steps_when_configured(cfg, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(booking_sources, "load_source",
                        lambda path=None: types.SimpleNamespace(kind="csv"))
    _write(cfg / "alerts.json", {})
    monkeypatch.setattr(settings, "access_password", SecretStr("long-password"))
    steps = _steps()
    assert steps["bookings"].state == "done" and "csv" in steps["bookings"].detail
    assert steps["alerts"].state == "done"
    assert steps["password"].state == "done"


def test_a_network_address_without_a_password_is_required(cfg, monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_host", "0.0.0.0")
    step = _steps()["password"]
    assert step.state == "todo" and step.required and "0.0.0.0" in step.detail


@pytest.mark.parametrize(("status", "state"), [
    (None, "todo"), ({"alive": False, "age_s": 7200}, "warning"),
    ({"alive": True, "age_s": 3}, "done"),
])
def test_the_recording_program_status(cfg, monkeypatch, status, state) -> None:
    _status(monkeypatch, status)
    assert _steps()["worker"].state == state


def test_broken_configs_are_warnings_never_errors(cfg, monkeypatch) -> None:
    _write(cfg / "cameras.json", "{ not json")
    _write(cfg / "slots_schedule.json", "[{\"venue_id\": \"v\"}]")
    _write(cfg / "bookings_source.json", "{ nope")

    def boom():
        raise RuntimeError("status file unreadable")

    fake = types.ModuleType("pitch_occupancy.status")
    fake.read_status = boom
    monkeypatch.setitem(sys.modules, "pitch_occupancy.status", fake)

    steps = _steps()
    assert steps["cameras"].state == "warning"
    assert "not valid JSON" in steps["cameras"].detail
    assert steps["boundaries"].state == "warning"
    assert steps["schedule"].state == "warning" and "missing" in steps["schedule"].detail
    assert steps["bookings"].state == "warning"
    assert steps["worker"].state == "warning" and "unreadable" in steps["worker"].detail
    assert len(steps) == 10


def test_the_endpoint_answers_even_when_broken(cfg) -> None:
    _write(cfg / "cameras.json", "[1, 2")
    app = FastAPI()
    app.include_router(setup_api.router)
    r = TestClient(app).get("/api/v1/setup/checklist")
    assert r.status_code == 200
    body = r.json()
    assert len(body["steps"]) == 10 and body["required_done"] is False
    assert body["steps"][0]["state"] == "warning"
    assert {"key", "title", "state", "detail", "required", "link", "hint"} <= set(
        body["steps"][0])


def test_the_dashboard_has_the_getting_started_card() -> None:
    from pitch_occupancy.api.dashboard import DASHBOARD_HTML

    assert "Getting started" in DASHBOARD_HTML
    assert "/api/v1/setup/checklist" in DASHBOARD_HTML
    assert "localStorage" in DASHBOARD_HTML


def test_the_dashboard_escapes_what_it_reads_from_the_database() -> None:
    """Slot ids come from the schedule's venue id, which the schedule editor writes. Put
    into innerHTML and an inline onclick unescaped, a crafted venue id ran script in every
    operator's browser once its slot was recorded."""
    from pitch_occupancy.api.dashboard import DASHBOARD_HTML

    assert "onclick=\"openSlot(" not in DASHBOARD_HTML
    for raw in ("${a.explanation}", "${a.slot_id}", "${s.slot_id}", "${f.field_id}",
                "${p}</figcaption>"):
        assert raw not in DASHBOARD_HTML, raw
