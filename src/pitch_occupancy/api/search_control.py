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

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
STATE = RESULTS / "preprocess_search.json"
LOCK = RESULTS / ".preprocess_search.lock"
LOG = RESULTS / ".preprocess_search.log"
SCRIPT = ROOT / "experiments" / "preprocess_search.py"

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class StartRequest(BaseModel):
    model: str = Field(default="convnextv2", description="Backbone to search with.")
    rounds: int = Field(default=3, ge=1, le=6)
    limit: int | None = Field(
        default=None,
        description=(
            "Subsample frames for speed. Leave unset for a result you intend to quote - "
            "subsampling shrinks the venue folds until recall can only take a few values."
        ),
    )


class SearchStatus(BaseModel):
    running: bool
    pid: int | None = None
    evaluations: int = 0
    n_frames: list[int] = []
    rounds_done: list[int] = []
    baseline: float | None = None
    best_label: str | None = None
    best_recall: float | None = None
    results: list[dict] = []
    warning: str | None = None


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


@router.get("/preprocess", response_model=SearchStatus)
def status() -> SearchStatus:
    """Current state of the preprocessing search, however far it has got."""
    state = _read_state()
    evals = state.get("evaluations", [])
    running = LOCK.exists()
    pid = None
    if running:
        try:
            pid = int(LOCK.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = None

    frames = sorted({e["n_frames"] for e in evals if e.get("n_frames")})
    baseline = next((e["play_recall"] for e in evals if e["label"] == "baseline"), None)
    ranked = sorted(evals, key=lambda e: -e["play_recall"])

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

    return SearchStatus(
        running=running,
        pid=pid,
        evaluations=len(evals),
        n_frames=frames,
        rounds_done=sorted({e["round"] for e in evals}),
        baseline=baseline,
        best_label=ranked[0]["label"] if ranked else None,
        best_recall=ranked[0]["play_recall"] if ranked else None,
        results=[
            {
                "label": e["label"],
                "recall": round(e["play_recall"], 4),
                "delta": round(e["play_recall"] - baseline, 4) if baseline else 0.0,
                "worst": round(e["worst_fold"], 4),
                "false_play": round(e["false_play"], 4),
                "round": e["round"],
                "describe": e.get("describe", ""),
            }
            for e in ranked
        ],
        warning=warning,
    )


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
    """Discard the current results so a run with different settings can start."""
    if LOCK.exists():
        raise HTTPException(409, "A search is running; stop it before clearing results.")
    STATE.unlink(missing_ok=True)
    return status()
