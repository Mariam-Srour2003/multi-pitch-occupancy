"""Start and monitor the preprocessing search from the browser (WP3-T8).

The search is a long subprocess, not a request handler: a full-size run takes hours, so it
is launched detached and the browser polls its state file. That also means a closed tab, a
reload, or a server restart does not interrupt a run in progress.

Concurrency is the thing this has to get right, because it has already gone wrong once -
three searches ran at the same time, each rewriting the shared state file, and the survivor
held evaluations scored on different frame counts with nothing to tell them apart. The
search script holds a lock for exactly this reason, and these endpoints refuse to start a
second run rather than discovering the collision afterwards.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pitch_occupancy.api import search_archive

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
STATE = RESULTS / "preprocess_search.json"
LOCK = RESULTS / ".preprocess_search.lock"
LOG = RESULTS / ".preprocess_search.log"
SCRIPT = ROOT / "experiments" / "preprocess_search.py"

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class StartRequest(BaseModel):
    model: str = Field(default="dinov2", description="Backbone to search with.")
    rounds: int = Field(default=3, ge=1, le=6)
    limit: int | None = Field(
        default=None,
        description=(
            "Subsample frames for speed. Leave unset for a result you intend to quote - "
            "subsampling shrinks the venue folds until recall can only take a few values."
        ),
    )


def expected_evaluations(rounds: int = 3) -> int:
    """How many evaluations a greedy run of `rounds` rounds will need.

    Round one tries every setting; each later round drops the switch just adopted. This is
    an upper bound - the search stops early when a round improves on nothing.
    """
    from pitch_occupancy.vision.preprocess import SWITCHES

    remaining = dict(SWITCHES)
    total = 0
    for _ in range(rounds):
        if not remaining:
            break
        total += sum(len(v) for v in remaining.values())
        remaining.pop(next(iter(remaining)))
    return total + 1  # + the baseline


class SearchStatus(BaseModel):
    running: bool
    pid: int | None = None
    evaluations: int = 0
    expected: int = 0
    progress: float = 0.0
    seconds_per_eval: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    n_frames: list[int] = []
    models: list[str] = []
    rounds_done: list[int] = []
    baseline: float | None = None
    best_label: str | None = None
    best_recall: float | None = None
    results: list[dict] = []
    warning: str | None = None
    rescore_warning: str | None = None


class SavedRun(BaseModel):
    """One cell of the saved grid, whether or not it was ever run.

    It carries the same result rows as a live run so the page can draw either with one
    renderer, plus `saved`, which is what tells "nobody has run this yet" apart from "this
    ran and scored nothing" - two states that look identical once both are empty lists and
    which mean opposite things about what to do next.
    """

    model: str
    model_label: str
    scope: str
    scope_label: str
    saved: bool
    generated: str = ""
    evaluations: int = 0
    n_frames: list[int] = []
    models: list[str] = []
    baseline: float | None = None
    best_label: str | None = None
    best_recall: float | None = None
    results: list[dict] = []
    warning: str | None = None
    rescore_warning: str | None = None


def _archive_dir() -> Path:
    """Where saved runs live, derived from `RESULTS` rather than fixed at import.

    `RESULTS` is redirected to a temporary directory by the tests, and an archive path that
    ignored that wrote fabricated cells into the repository's own `results/search_runs/`.
    """
    return RESULTS / "search_runs"


def _read_state() -> dict:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # the search writes the whole file each time; a poll can catch it mid-write
        return {}


def _pid_alive(pid: int) -> bool:
    try:
        subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"] if sys.platform == "win32"
            else ["kill", "-0", str(pid)],
            capture_output=True, check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def summarise(evals: list[dict]) -> dict:
    """Everything the page shows about a set of evaluations, live or saved.

    Split out of `status` when the saved-run grid was added, because the ranking here is
    not presentation - it is the safeguard. Recomputing it a second time for archived runs
    is how a stored ConvNeXtV2 chart ends up ranked on raw recall while the live one is
    ranked on the balanced score, and the whole point of the false-play control is that
    those two orders differ.
    """
    frames = sorted({e["n_frames"] for e in evals if e.get("n_frames")})
    # Which backbone these evaluations came from. The panel's dropdown is what the *next*
    # run would use, so without this a stored ConvNeXtV2 chart reads as whatever happens to
    # be selected there.
    models = sorted({e["model"] for e in evals if e.get("model")})
    baseline = next((e["play_recall"] for e in evals if e["label"] == "baseline"), None)

    # Rank on the balanced score, never on recall alone. Every cross-venue fold is 100%
    # active play, so recall is buyable by answering "playing" more often - which is the
    # entire reason the false-play control exists, and this endpoint was sorting by
    # `-play_recall` regardless of it.
    #
    # `"balanced" in e` is the only reliable test for whether an entry was scored by the
    # *repaired* control. Entries from before the repair carry `false_play: 0.0` for
    # everything, and 0.0 is not a low rate there - it is a broken measurement. It cannot
    # be detected from the value, because four DINOv2 configurations genuinely score 0.0.
    # In the current state file 45 of 88 entries are un-rescored, all convnextv2, and the
    # highest-recall row among them is a baseline at 0.9841 with a fabricated 0.0000.
    rescored = [e for e in evals if "balanced" in e]
    unscored = [e for e in evals if "balanced" not in e]
    ranked = sorted(rescored, key=lambda e: -e["balanced"]) + sorted(
        unscored, key=lambda e: -e["play_recall"]
    )
    best = ranked[0] if rescored else None

    # Pace is measured from this run rather than assumed: the same search is roughly three
    # times slower on 1,578 frames than on 500, and slower again on a heavier backbone.
    durations = sorted(e["seconds"] for e in evals if e.get("seconds"))
    per_eval = durations[len(durations) // 2] if durations else None
    expected = expected_evaluations()
    elapsed = float(sum(durations))
    # What is left to wait for at this pace. `status` blanks it when nothing is running,
    # and a saved run never shows it - the waiting is over by the time it is saved.
    eta = max(0.0, (expected - len(evals)) * per_eval) if per_eval and evals else None

    # Kept as a separate field rather than folded into `warning`. Mixed or subsampled
    # frame counts mean the numbers are not comparable *at all* - "discard this state file
    # and rerun" - and an un-rescored subset must not displace that from the one slot the
    # front end reads. The first draft of this used `elif` and buried the worse problem.
    rescore_warning = None
    if unscored:
        rescore_warning = (
            f"{len(unscored)} of {len(evals)} evaluations predate the repair of the "
            f"false-play control, which scored probes on their own training data and read "
            f"0.0000 for everything. They are listed after the scored ones, without a "
            f"false-play figure, and excluded from 'best' - a fabricated zero is not a low "
            f"rate. Re-running the search re-scores them from cache."
        )

    warning = None
    if len(frames) > 1:
        warning = (
            f"Evaluations are scored on {frames} frames. Scores from different frame "
            f"counts are not comparable - discard this state file and rerun."
        )
    elif frames and frames[0] < 1000:
        warning = (
            f"Scored on {frames[0]} frames. Subsampling shrinks the venue folds, so "
            f"recall takes only a few discrete values - fine for a smoke test, not for a "
            f"number you intend to quote."
        )

    return dict(
        evaluations=len(evals),
        expected=expected,
        progress=round(min(len(evals) / expected, 1.0), 4) if expected else 0.0,
        seconds_per_eval=round(per_eval, 1) if per_eval else None,
        elapsed_seconds=round(elapsed, 1),
        eta_seconds=round(eta, 1) if eta is not None else None,
        n_frames=frames,
        models=models,
        rounds_done=sorted({e["round"] for e in evals}),
        baseline=baseline,
        best_label=best["label"] if best else None,
        best_recall=best["play_recall"] if best else None,
        results=[
            {
                "label": e["label"],
                "recall": round(e["play_recall"], 4),
                "delta": round(e["play_recall"] - baseline, 4) if baseline else 0.0,
                "worst": round(e["worst_fold"], 4),
                # `None`, not 0.0, for an entry the repaired control never scored. Sending
                # the placeholder zero is what let the front end render 45 broken rows as
                # clean wins: its flag condition is `false_play > 0.001`, so a fabricated
                # 0.0 passed as the best possible result.
                "false_play": round(e["false_play"], 4) if "balanced" in e else None,
                "balanced": round(e["balanced"], 4) if "balanced" in e else None,
                "rescored": "balanced" in e,
                "round": e["round"],
                "describe": e.get("describe", ""),
            }
            for e in ranked
        ],
        warning=warning,
        rescore_warning=rescore_warning,
    )


@router.get("/preprocess", response_model=SearchStatus)
def status() -> SearchStatus:
    """Current state of the preprocessing search, however far it has got."""
    running = LOCK.exists()
    pid = None
    if running:
        try:
            pid = int(LOCK.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = None
    summary = summarise(_read_state().get("evaluations", []))
    if not running:
        summary["eta_seconds"] = None
    return SearchStatus(running=running, pid=pid, **summary)


@router.post("/preprocess", response_model=SearchStatus)
def start(req: StartRequest) -> SearchStatus:
    """Launch the search as a detached subprocess.

    Refuses if a run already holds the lock, and refuses to append to a state file scored
    on a different frame count - both failure modes that previously produced results which
    looked fine and were not.
    """
    if LOCK.exists():
        raise HTTPException(409, "A search is already running. Wait for it or stop it.")

    existing = {e.get("n_frames") for e in _read_state().get("evaluations", [])}
    existing.discard(None)
    if existing and existing != {req.limit} and not (req.limit is None and existing == {1578}):
        raise HTTPException(
            409,
            f"Existing results are scored on {sorted(existing)} frames; this run would use "
            f"{req.limit or 'all'}. Clear the results first - scores from different frame "
            f"counts cannot be compared.",
        )

    cmd = [sys.executable, str(SCRIPT), "--models", req.model, "--rounds", str(req.rounds)]
    if req.limit:
        cmd += ["--limit", str(req.limit)]

    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as log:
        subprocess.Popen(  # noqa: S603 - fixed command, no shell, no user strings
            cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
    return status()


@router.delete("/preprocess", response_model=SearchStatus)
def clear() -> SearchStatus:
    """Discard the current results so a run with different settings can start.

    Every backbone in the file is archived first. Clearing is the normal way to make room
    for the next of the six runs, and hours of evaluations should not depend on someone
    having pressed save before pressing clear.
    """
    if LOCK.exists():
        raise HTTPException(409, "A search is running; stop it before clearing results.")
    archive_state(_read_state())
    STATE.unlink(missing_ok=True)
    return status()


def archive_state(state: dict) -> list[str]:
    """Save each backbone in `state` into its cell, keyed by the frame count it used.

    The scope is read from the evaluations rather than from whatever was typed into the
    panel: what a run *is* is what it scored on, and a mislabelled cell is worse than an
    empty one.
    """
    evals = state.get("evaluations", [])
    saved = []
    for model in sorted({e.get("model") for e in evals if e.get("model")}):
        frames = sorted({e["n_frames"] for e in evals
                         if e.get("model") == model and e.get("n_frames")})
        if len(frames) != 1:
            continue  # mixed or unknown: the frame-count guard exists to prevent this
        scope = search_archive.scope_of(None if frames[0] > 500 else frames[0])
        try:
            search_archive.save(model, scope, state, base=_archive_dir())
        except ValueError:
            continue  # a backbone the grid does not offer
        saved.append(f"{model}__{scope}")
    return saved


# --- saved runs ---------------------------------------------------------------------
#
# Three backbones over two frame counts is six runs, and the live state file holds exactly
# one of them. Each finished run is copied into `results/search_runs/` and read back here,
# so the six can be compared without re-running any of them.


@router.get("/runs")
def runs() -> dict:
    """The whole grid: six cells, each saying whether it has been run and what it holds."""
    return {"cells": search_archive.index(_archive_dir())}


@router.get("/runs/{model}/{scope}", response_model=SavedRun)
def saved_run(model: str, scope: str) -> SavedRun:
    """One saved run, ranked and warned about by the same code as a live one."""
    try:
        run = search_archive.load(model, scope, _archive_dir())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    meta = next(
        c for c in search_archive.index(_archive_dir())
        if c["model"] == model and c["scope"] == scope
    )
    summary = summarise(run.get("evaluations", []) if run else [])
    return SavedRun(
        model=model,
        model_label=meta["model_label"],
        scope=scope,
        scope_label=meta["scope_label"],
        saved=run is not None,
        generated=run.get("generated", "") if run else "",
        evaluations=summary["evaluations"],
        n_frames=summary["n_frames"],
        models=summary["models"],
        baseline=summary["baseline"],
        best_label=summary["best_label"],
        best_recall=summary["best_recall"],
        results=summary["results"],
        warning=summary["warning"],
        rescore_warning=summary["rescore_warning"],
    )


@router.post("/runs", response_model=SavedRun)
def save_current(model: str, scope: str) -> SavedRun:
    """Copy what is in the live state file into one cell of the grid.

    Saving is also done by `experiments/search_all.py` as each run finishes; this is the
    hand-operated version, for a run that was started some other way.
    """
    state = _read_state()
    if not [e for e in state.get("evaluations", []) if e.get("model") == model]:
        raise HTTPException(404, f"the current results hold nothing for {model}")
    try:
        search_archive.save(model, scope, state, base=_archive_dir())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return saved_run(model, scope)

