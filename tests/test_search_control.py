"""Starting and monitoring the preprocessing search from the browser.

The failure this guards against already happened once: three searches ran at the same
time, each rewriting the shared state file, and the survivor mixed evaluations scored on
different frame counts. Refusing to start is the whole point of these endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api import search_control
from pitch_occupancy.api.app import app


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(search_control, "RESULTS", tmp_path)
    monkeypatch.setattr(search_control, "STATE", tmp_path / "preprocess_search.json")
    monkeypatch.setattr(search_control, "LOCK", tmp_path / ".lock")
    monkeypatch.setattr(search_control, "LOG", tmp_path / ".log")
    return tmp_path


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def write_state(path: Path, evals: list[dict]) -> None:
    path.write_text(json.dumps({"generated": "", "evaluations": evals, "best": {}}))


def ev(label: str, recall: float, *, n_frames: int = 1578, rnd: int = 1, fp: float = 0.0) -> dict:
    return {
        "model": "convnextv2", "n_frames": n_frames, "hash": label, "label": label,
        "round": rnd, "config": {}, "describe": label, "seconds": 1.0,
        "play_recall": recall, "worst_fold": recall - 0.05, "false_play": fp,
    }


# --- status -----------------------------------------------------------------


def test_status_with_no_results(env, client) -> None:
    s = client.get("/api/v1/search/preprocess").json()
    assert s["running"] is False
    assert s["evaluations"] == 0
    assert s["results"] == []


def test_status_reports_progress_and_ranks_results(env, client) -> None:
    write_state(search_control.STATE, [ev("baseline", 0.90), ev("top_crop=0.2", 0.95)])
    s = client.get("/api/v1/search/preprocess").json()
    assert s["evaluations"] == 2
    assert s["best_label"] == "top_crop=0.2"
    assert s["results"][0]["delta"] == pytest.approx(0.05)


def test_running_is_driven_by_the_lock(env, client) -> None:
    search_control.LOCK.write_text("4321")
    s = client.get("/api/v1/search/preprocess").json()
    assert s["running"] is True
    assert s["pid"] == 4321


def test_mixed_frame_counts_raise_a_warning(env, client) -> None:
    """The exact corruption that happened: two frame counts in one state file."""
    write_state(search_control.STATE, [ev("baseline", 0.9), ev("x", 0.95, n_frames=500)])
    s = client.get("/api/v1/search/preprocess").json()
    assert "not comparable" in s["warning"]


def test_subsampled_results_are_flagged_as_not_quotable(env, client) -> None:
    write_state(search_control.STATE, [ev("baseline", 0.9, n_frames=500)])
    assert "quote" in client.get("/api/v1/search/preprocess").json()["warning"]


def test_full_size_results_carry_no_warning(env, client) -> None:
    write_state(search_control.STATE, [ev("baseline", 0.9)])
    assert client.get("/api/v1/search/preprocess").json()["warning"] is None


def test_a_half_written_state_file_does_not_crash_the_poll(env, client) -> None:
    """The search rewrites the whole file, so a poll can catch it mid-write."""
    search_control.STATE.write_text('{"evaluations": [{"lab')
    assert client.get("/api/v1/search/preprocess").json()["evaluations"] == 0


# --- starting ---------------------------------------------------------------


def test_refuses_to_start_while_one_is_running(env, client) -> None:
    search_control.LOCK.write_text("999")
    r = client.post("/api/v1/search/preprocess", json={"model": "convnextv2"})
    assert r.status_code == 409
    assert "already running" in r.json()["detail"]


def test_refuses_to_mix_frame_counts(env, client) -> None:
    write_state(search_control.STATE, [ev("baseline", 0.9, n_frames=500)])
    r = client.post("/api/v1/search/preprocess", json={"model": "convnextv2"})
    assert r.status_code == 409
    assert "cannot be compared" in r.json()["detail"]


def test_matching_frame_count_is_allowed_to_continue(env, client, monkeypatch) -> None:
    started = {}
    monkeypatch.setattr(
        search_control.subprocess, "Popen",
        lambda cmd, **kw: started.setdefault("cmd", cmd) or type("P", (), {"pid": 1})(),
    )
    write_state(search_control.STATE, [ev("baseline", 0.9, n_frames=500)])
    r = client.post("/api/v1/search/preprocess", json={"model": "convnextv2", "limit": 500})
    assert r.status_code == 200
    assert "--limit" in started["cmd"]


def test_rejects_an_unknown_rounds_value(env, client) -> None:
    r = client.post("/api/v1/search/preprocess", json={"model": "convnextv2", "rounds": 99})
    assert r.status_code == 422


# --- clearing ---------------------------------------------------------------


def test_clear_removes_the_state(env, client) -> None:
    write_state(search_control.STATE, [ev("baseline", 0.9)])
    assert client.request("DELETE", "/api/v1/search/preprocess").json()["evaluations"] == 0
    assert not search_control.STATE.exists()


def test_clear_refuses_while_running(env, client) -> None:
    search_control.LOCK.write_text("77")
    r = client.request("DELETE", "/api/v1/search/preprocess")
    assert r.status_code == 409
