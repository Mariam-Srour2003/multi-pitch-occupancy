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


def ev(
    label: str,
    recall: float,
    *,
    n_frames: int = 1578,
    rnd: int = 1,
    fp: float = 0.0,
    rescored: bool = True,
) -> dict:
    """One evaluation as the search writes it.

    ``rescored=False`` omits ``balanced``, which is how entries written *before* the
    false-play control was repaired look: they carry ``false_play: 0.0`` for everything,
    and that zero is a broken measurement rather than a clean sheet. The presence of
    ``balanced`` is the only reliable way to tell the two apart, because some
    configurations genuinely do score 0.0.
    """
    out = {
        "model": "convnextv2", "n_frames": n_frames, "hash": label, "label": label,
        "round": rnd, "config": {}, "describe": label, "seconds": 1.0,
        "play_recall": recall, "worst_fold": recall - 0.05, "false_play": fp,
    }
    if rescored:
        out["balanced"] = recall - fp
    return out


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


# --- progress and eta -------------------------------------------------------


def test_expected_evaluations_covers_the_greedy_rounds() -> None:
    from pitch_occupancy.api.search_control import expected_evaluations
    from pitch_occupancy.vision.preprocess import SWITCHES

    one_round = sum(len(v) for v in SWITCHES.values())
    assert expected_evaluations(1) == one_round + 1  # + baseline
    assert expected_evaluations(3) > expected_evaluations(1)


def test_eta_is_measured_from_this_run_not_assumed(env, client) -> None:
    """Pace differs threefold between frame counts and again between backbones, so the
    estimate has to come from the run's own timings."""
    search_control.LOCK.write_text("1")
    evals = [ev("baseline", 0.9), ev("a", 0.91), ev("b", 0.92)]
    for e in evals:
        e["seconds"] = 240.0
    write_state(search_control.STATE, evals)
    s = client.get("/api/v1/search/preprocess").json()
    assert s["seconds_per_eval"] == 240.0
    assert s["eta_seconds"] == pytest.approx((s["expected"] - 3) * 240.0)
    assert 0 < s["progress"] < 1


def test_no_eta_when_nothing_is_running(env, client) -> None:
    write_state(search_control.STATE, [ev("baseline", 0.9)])
    assert client.get("/api/v1/search/preprocess").json()["eta_seconds"] is None


def test_progress_never_exceeds_one(env, client) -> None:
    search_control.LOCK.write_text("1")
    write_state(search_control.STATE, [ev(f"s{i}", 0.9) for i in range(300)])
    s = client.get("/api/v1/search/preprocess").json()
    assert s["progress"] == 1.0
    assert s["eta_seconds"] == 0.0


# --- the false-play control has to reach the front end ---------------------
#
# The safeguard existed in thesis_site._search_summary(), which nothing but a test ever
# called. The panel that is actually rendered took its data from here, and here ranked by
# `-play_recall` with no reference to false-play at all - so 45 un-rescored entries in the
# real state file, all carrying the placeholder 0.0, were served as clean top results.


def test_ranking_uses_the_balanced_score_not_recall(env, client) -> None:
    """Recall is buyable on a 100%-active-play fold; that is why balanced exists."""
    write_state(search_control.STATE, [
        ev("baseline", 0.90, fp=0.02),
        ev("bought_it", 0.99, fp=0.60),   # higher recall, far worse false-play
        ev("earned_it", 0.95, fp=0.03),
    ])
    s = client.get("/api/v1/search/preprocess").json()
    assert s["best_label"] == "earned_it", "the highest recall must not win by itself"
    assert [r["label"] for r in s["results"]][0] == "earned_it"


def test_an_un_rescored_entry_is_never_reported_as_best(env, client) -> None:
    write_state(search_control.STATE, [
        ev("legacy_high", 0.99, rescored=False),
        ev("scored_low", 0.80, fp=0.01),
    ])
    s = client.get("/api/v1/search/preprocess").json()
    assert s["best_label"] == "scored_low"


def test_an_un_rescored_entry_carries_no_false_play_figure(env, client) -> None:
    """`None`, not 0.0. Sending the placeholder is what made the panel render it green."""
    write_state(search_control.STATE, [ev("legacy", 0.9, rescored=False)])
    r = client.get("/api/v1/search/preprocess").json()["results"][0]
    assert r["false_play"] is None
    assert r["balanced"] is None
    assert r["rescored"] is False


def test_a_rescored_entry_keeps_its_figures(env, client) -> None:
    write_state(search_control.STATE, [ev("fresh", 0.9, fp=0.04)])
    r = client.get("/api/v1/search/preprocess").json()["results"][0]
    assert r["false_play"] == pytest.approx(0.04)
    assert r["rescored"] is True


def test_un_rescored_entries_are_listed_after_the_scored_ones(env, client) -> None:
    write_state(search_control.STATE, [
        ev("legacy_a", 0.99, rescored=False),
        ev("scored", 0.70, fp=0.01),
        ev("legacy_b", 0.98, rescored=False),
    ])
    labels = [r["label"] for r in client.get("/api/v1/search/preprocess").json()["results"]]
    assert labels[0] == "scored"
    assert set(labels[1:]) == {"legacy_a", "legacy_b"}


def test_the_rescore_notice_does_not_displace_a_frame_count_warning(env, client) -> None:
    """Mixed frame counts mean the numbers are not comparable at all.

    That is the worse problem and it must keep the slot the front end reads. The first
    draft of this used `elif` and buried it behind the rescore notice.
    """
    write_state(search_control.STATE, [
        ev("a", 0.9, n_frames=500, rescored=False),
        ev("b", 0.9, n_frames=1578, rescored=False),
    ])
    s = client.get("/api/v1/search/preprocess").json()
    assert "not comparable" in s["warning"]
    assert "predate the repair" in s["rescore_warning"]


def test_no_rescore_notice_when_everything_was_scored(env, client) -> None:
    write_state(search_control.STATE, [ev("a", 0.9, fp=0.02)])
    assert client.get("/api/v1/search/preprocess").json()["rescore_warning"] is None
