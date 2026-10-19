"""Every test runs against its own database and its own copies of the operator's files.

The real `data/db/pitch_monitor.db`, `configs/cameras.json`, `configs/alerts.json`,
`configs/playgrounds.json`, `configs/assist.json` and the saved pictures belong to the
installation. A test that forgot to redirect one of them used to write into it. This fixture
redirects them all before each test; a test that needs a specific file still sets its own.
The rule file (`configs/rules.json`) and the pitch boundaries are read, never written, and stay
real: several tests describe the shipped rule on purpose.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_stores(tmp_path_factory, monkeypatch):
    from pitch_occupancy import alerts, cameras, playgrounds
    from pitch_occupancy.assist import config as assist_config
    from pitch_occupancy.config import settings

    home = tmp_path_factory.mktemp("stores")
    monkeypatch.setattr(settings, "db_path", home / "db" / "pitch_monitor.db")
    monkeypatch.setattr(cameras, "CONFIG_PATH", home / "cameras.json")
    monkeypatch.setattr(alerts, "CONFIG_PATH", home / "alerts.json")
    # On the module that reads each one (see `playgrounds/__init__`).
    monkeypatch.setattr(playgrounds.store, "CONFIG_PATH", home / "playgrounds.json")
    monkeypatch.setattr(playgrounds.capacity, "SPEED_PATH", home / "computer_speed.json")
    monkeypatch.setattr(playgrounds.snapshots, "SNAPSHOT_DIR", home / "snapshots")
    monkeypatch.setattr(playgrounds.snapshots, "_snap_bytes", None)
    monkeypatch.setattr(assist_config, "CONFIG_PATH", home / "assist.json")
