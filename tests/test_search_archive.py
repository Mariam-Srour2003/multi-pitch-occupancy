"""Saved preprocessing searches: the six-cell grid behind "View old results".

Two failures are worth pinning here, and both have already happened in this project in one
form or another.

The first is a saved run that is ranked differently from the live one. Every cross-venue
test fold is 100% ACTIVE_PLAY, so recall can be bought by answering "playing" more often,
and the balanced score is the only order that survives that. A second renderer for archived
runs would have been a second chance to sort by recall, so the archive stores evaluations
and nothing else - the ranking is recomputed by the same function either way.

The second is the difference between "never run" and "ran and scored nothing". Both are an
empty chart; one means press the button and the other means the metric is broken. The grid
reports them apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pitch_occupancy.api import search_archive, search_control
from pitch_occupancy.api.app import app


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """Redirect the results directory, which the archive follows.

    `search_archive.ARCHIVE` is a module constant, but every entry point takes the base
    directory from `search_control.RESULTS` at call time. That is not tidiness: the first
    run of this suite before it worked that way wrote a fabricated `convnextv2__all.json`
    into the repository's own results.
    """
    monkeypatch.setattr(search_control, "RESULTS", tmp_path)
    monkeypatch.setattr(search_control, "STATE", tmp_path / "preprocess_search.json")
    monkeypatch.setattr(search_control, "LOCK", tmp_path / ".lock")
    monkeypatch.setattr(search_control, "LOG", tmp_path / ".log")
    return tmp_path


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def ev(label: str, recall: float | None, *, model: str = "dinov2", n_frames: int = 1599,
       fp: float = 0.0, rnd: int = 1) -> dict:
    """One evaluation as the search writes it; `recall=None` is an unscored one."""
    out = {
        "model": model, "n_frames": n_frames, "hash": label, "label": label, "round": rnd,
        "config": {}, "describe": label, "seconds": 1.0,
        "play_recall": float("nan") if recall is None else recall,
        "worst_fold": float("nan") if recall is None else recall - 0.05,
        "false_play": fp,
    }
    out["balanced"] = float("nan") if recall is None else recall - fp
    return out


def write_state(env: Path, evals: list[dict]) -> dict:
    state = {"generated": "2026-09-23T10:00:00", "evaluations": evals, "best": {}}
    (env / "preprocess_search.json").write_text(json.dumps(state))
    return state


# --- the grid ---------------------------------------------------------------


def test_the_grid_lists_all_six_cells_even_when_none_have_been_run(env, client) -> None:
    """A selector that offered only the runs that exist would hide the missing ones, and
    "which have I not run yet" is the question this grid answers."""
    cells = client.get("/api/v1/search/runs").json()["cells"]
    assert len(cells) == 6
    assert {(c["model"], c["scope"]) for c in cells} == {
        (m, s) for m in search_archive.MODELS for s in ("all", "500")
    }
    assert all(c["saved"] is False for c in cells)


def test_a_cell_that_was_never_run_is_not_the_same_as_one_that_scored_nothing(
    env, client
) -> None:
    """The distinction the whole panel rests on. Both draw no chart; one says press Run and
    the other says the run happened and the metric produced NaN for all of it."""
    search_archive.save("dinov2", "all", {"evaluations": [ev("baseline", None)]},
                        base=env / "search_runs")

    never = client.get("/api/v1/search/runs/vit/all").json()
    assert never["saved"] is False and never["evaluations"] == 0

    empty = client.get("/api/v1/search/runs/dinov2/all").json()
    assert empty["saved"] is True and empty["evaluations"] == 1
    assert [c for c in client.get("/api/v1/search/runs").json()["cells"]
            if c["model"] == "dinov2" and c["scope"] == "all"][0]["scored"] == 0


# --- what a cell holds ------------------------------------------------------


def test_a_cell_holds_one_backbone_only(env) -> None:
    """A state file can legally hold two models - `--models a b` - and a cell that quietly
    carried another model's rows is the mixing bug the frame-count guard exists to stop,
    one level up."""
    state = {"evaluations": [ev("a", 0.9), ev("b", 0.8, model="vit")], "best": {}}
    search_archive.save("dinov2", "all", state, base=env / "search_runs")
    saved = search_archive.load("dinov2", "all", env / "search_runs")
    assert [e["model"] for e in saved["evaluations"]] == ["dinov2"]


def test_an_unknown_model_or_scope_is_refused_rather_than_sanitised(env) -> None:
    """Both halves reach the filesystem from a URL path. Rejecting anything that is not one
    of the six known values is the only version of this that cannot be talked into `../`."""
    for model, scope in [("../etc", "all"), ("dinov2", "../../x"), ("resnet", "all"),
                         ("dinov2", "250")]:
        with pytest.raises(ValueError):
            search_archive.cell(model, scope, env / "search_runs")


def test_a_saved_run_is_ranked_on_the_balanced_score_like_a_live_one(env, client) -> None:
    """The safeguard, not a presentation detail. `bought` has the higher recall and buys it
    by calling empty pitches a match; ranking it first is exactly the failure the false-play
    control was added to prevent."""
    search_archive.save("dinov2", "all", {"evaluations": [
        ev("baseline", 0.90, fp=0.02),
        ev("bought", 0.99, fp=0.80),
        ev("honest", 0.95, fp=0.03),
    ]}, base=env / "search_runs")
    r = client.get("/api/v1/search/runs/dinov2/all").json()
    assert [x["label"] for x in r["results"]] == ["honest", "baseline", "bought"]
    assert r["best_label"] == "honest"


def test_a_saved_subsampled_run_carries_the_same_warning_as_a_live_one(env, client) -> None:
    """The reason to recompute rather than store a summary: this sentence is written once."""
    search_archive.save(
        "vit", "500",
        {"evaluations": [ev("baseline", 0.9, model="vit", n_frames=500)]},
        base=env / "search_runs",
    )
    r = client.get("/api/v1/search/runs/vit/500").json()
    assert r["warning"] and "500 frames" in r["warning"]


# --- saving -----------------------------------------------------------------


def test_clearing_the_live_results_archives_them_first(env, client) -> None:
    """Clearing is the normal way to make room for the next of the six runs. Hours of
    evaluations must not depend on someone having pressed save beforehand."""
    write_state(env, [ev("baseline", 0.9), ev("gray", 0.93)])
    assert client.delete("/api/v1/search/preprocess").status_code == 200
    saved = search_archive.load("dinov2", "all", env / "search_runs")
    assert saved and len(saved["evaluations"]) == 2
    assert not (env / "preprocess_search.json").exists()


def test_a_state_file_of_mixed_frame_counts_is_not_archived_into_either_cell(env) -> None:
    """Which cell would it be? Scores from different frame counts are not comparable, so a
    file holding both belongs in neither - and a mislabelled cell is worse than no cell."""
    state = {"evaluations": [ev("a", 0.9, n_frames=1599), ev("b", 0.9, n_frames=500)]}
    assert search_control.archive_state(state) == []
    assert not (env / "search_runs").exists()


def test_the_scope_is_taken_from_what_was_scored_not_from_what_was_typed(env) -> None:
    """What a run *is* is what it scored on."""
    assert search_control.archive_state({"evaluations": [ev("a", 0.9, n_frames=500)]}) == [
        "dinov2__500"
    ]
    assert search_archive.scope_of(None) == "all"
    assert search_archive.scope_of(500) == "500"


def test_saving_the_current_run_by_hand_refuses_a_backbone_it_does_not_hold(
    env, client
) -> None:
    """Otherwise the cell fills with an empty list and reads as a finished run of nothing."""
    write_state(env, [ev("baseline", 0.9)])
    assert client.post("/api/v1/search/runs?model=vit&scope=all").status_code == 404
    assert client.post("/api/v1/search/runs?model=dinov2&scope=all").status_code == 200
    assert search_archive.load("dinov2", "all", env / "search_runs")


# --- the metric the grid is worth running for -------------------------------


def test_a_fold_with_no_play_frame_is_skipped_rather_than_scored_nan() -> None:
    """Why the six runs were worth nothing before 2026-09-23.

    `davinci_j_maint_outdoor` holds no active-play footage, so its recall was `nan`, and a
    mean over any array containing `nan` is `nan` - every evaluation of every backbone, at
    every frame count. The 43 stored convnextv2 evaluations are all unscored for exactly
    this reason. Recall is undefined where there is nothing to recall; that is not a reason
    to discard the folds that can measure it.
    """
    import numpy as np

    from experiments.preprocess_search import score
    from pitch_occupancy.data.splits import Split

    def row(name: str, cls: str):
        from pitch_occupancy.data.manifest import ManifestRow
        return ManifestRow(
            file=name, label=cls, class3=cls, venue="v", camera="c",
            slot_date="2026-01-01", slot_time="10:00", slot_id="s", t_s=0.0,
            source="test", labeled_by="test", lighting="day", quality="ok",
            split_role="dev",
        )

    play = [row(f"p{i}", "C2_ACTIVE_PLAY") for i in range(6)]
    empty = [row(f"e{i}", "C1_EMPTY") for i in range(6)]
    rows = play + empty
    rng = np.random.default_rng(0)
    # Separable by construction, so recall is 1.0 on any fold that can be scored at all.
    X = np.vstack([rng.normal(3.0, 0.1, (6, 8)), rng.normal(-3.0, 0.1, (6, 8))])

    train = tuple(rows)
    scorable = Split(name="lo_venue_out__a", train=train, test=tuple(play[:3]),
                     group_key="venue")
    # The fold that broke everything: a held-out venue with no active play in it.
    unscorable = Split(name="lo_venue_out__maint", train=train, test=tuple(empty[:3]),
                       group_key="venue")

    out = score(rows, X, [scorable, unscorable])
    assert not math.isnan(out["play_recall"]), "one empty fold still poisons the mean"
    assert out["folds_scored"] == 1
    assert out["folds_total"] == 2

    # And an evaluation where nothing could be scored is still NaN, not invented.
    none = score(rows, X, [unscorable])
    assert math.isnan(none["play_recall"]) and none["folds_scored"] == 0
