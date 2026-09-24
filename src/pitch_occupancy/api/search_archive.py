"""Saved preprocessing searches, one file per backbone and frame count (WP3-T8).

The live search writes a *single* state file, `results/preprocess_search.json`, and refuses
to mix frame counts into it - which is correct, and it also means every finished run is
destroyed by the next one. Six runs are wanted here (three backbones over two frame
counts), so each finished run is copied into `results/search_runs/<model>__<scope>.json`
and the page reads those back.

An archive is a *snapshot*, not a second source of truth: it holds the evaluations exactly
as the search wrote them, and everything shown from it - the ranking, the deltas, the
warnings - is recomputed by the same code that renders a live run. Nothing is summarised on
the way in, because a summary written at save time is the copy that goes stale.

The filename is the key, so a repeated run overwrites its own cell rather than accumulating
runs nobody will compare. `scope` is what was *asked for* ("all" or "500") rather than the
row count that resulted: the development set grows, and a run asked for "all" belongs in
the same cell as the last one asked for "all" even when the manifest gained frames between
them. The row count is kept inside the file, where it is a fact about that run.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["MODELS", "SCOPES", "ARCHIVE", "cell", "save", "load", "index", "scope_of"]

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "results" / "search_runs"

#: The three backbones the panel offers, in the order it offers them.
MODELS = ("dinov2", "convnextv2", "vit")

#: The two frame counts, as (key, label, limit). `None` means the whole development set.
SCOPES = (("all", "All frames", None), ("500", "500 frames", 500))

_LABELS = {"dinov2": "DINOv2", "convnextv2": "ConvNeXtV2", "vit": "ViT"}


def scope_of(limit: int | None) -> str:
    """The archive key a run with this `--limit` belongs in."""
    return "500" if limit == 500 else "all" if limit is None else str(limit)


def cell(model: str, scope: str, base: Path | None = None) -> Path:
    """Where one (backbone, frame count) run is kept.

    Both parts are checked against the known values rather than sanitised. These reach the
    filesystem from a URL path, and a rejected unknown is the only version of this that
    cannot be talked into `../`.

    `base` exists so the directory can follow whatever `search_control.RESULTS` points at.
    Without it a test that redirects the results directory still archives into the real
    one - which is not a hypothetical: the first run of the suite after this module was
    written left a fabricated `convnextv2__all.json` in the repository.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if scope not in {s[0] for s in SCOPES}:
        raise ValueError(f"unknown scope {scope!r}")
    return (base or ARCHIVE) / f"{model}__{scope}.json"


def save(model: str, scope: str, state: dict, *, rounds: int | None = None,
         base: Path | None = None) -> Path:
    """Copy the evaluations of one backbone out of a finished run.

    Only that backbone's rows are taken. A state file can hold two models - the search
    accepts `--models a b` - and a cell that quietly contained another model's evaluations
    would be the same mixing bug the frame-count guard exists to prevent, one level up.
    """
    evals = [e for e in state.get("evaluations", []) if e.get("model") == model]
    frames = sorted({e["n_frames"] for e in evals if e.get("n_frames")})
    path = cell(model, scope, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": model,
                "scope": scope,
                "n_frames": frames[-1] if frames else None,
                "generated": state.get("generated", ""),
                "rounds": rounds,
                "evaluations": evals,
                "best": state.get("best", {}).get(model, {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load(model: str, scope: str, base: Path | None = None) -> dict | None:
    """One saved run, or `None` if that cell was never filled."""
    path = cell(model, scope, base)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def index(base: Path | None = None) -> list[dict]:
    """Every cell of the grid, filled or not.

    All six are listed either way. A selector that offered only the runs that exist would
    make a missing run invisible, and "which of these have I not run yet" is the question
    this grid is for.
    """
    out = []
    for model in MODELS:
        for scope, scope_label, _ in SCOPES:
            run = load(model, scope, base)
            evals = run.get("evaluations", []) if run else []
            scored = [e for e in evals if _is_number(e.get("play_recall"))]
            out.append({
                "model": model,
                "model_label": _LABELS.get(model, model),
                "scope": scope,
                "scope_label": scope_label,
                "saved": run is not None,
                "generated": run.get("generated", "") if run else "",
                "n_frames": run.get("n_frames") if run else None,
                "evaluations": len(evals),
                "scored": len(scored),
            })
    return out


def _is_number(v: object) -> bool:
    """NaN arrives from the state file as the float `nan`, and from JSON as `None`.

    `json.dumps` writes NaN out as the bare token `NaN`, which `json.loads` reads back as a
    float that fails every comparison silently. Both spellings mean the same thing here -
    this evaluation was never scored - and both have to be caught in the same place.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
