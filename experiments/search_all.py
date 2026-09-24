"""Fill the saved-run grid: every backbone over every frame count (WP3-T8).

`preprocess_search.py` searches one backbone at one frame count into one shared state file,
and refuses to mix frame counts into it - correctly, because scores from different counts
are not comparable. The page wants six runs side by side, so this drives the six in turn,
archiving each into `results/search_runs/` before the next one overwrites the state file.

    uv run python experiments/search_all.py                  # every empty cell, in order
    uv run python experiments/search_all.py --only dinov2    # one backbone, both counts
    uv run python experiments/search_all.py --force          # re-run cells already saved
    uv run python experiments/search_all.py --check          # what it would do, and why

**The full runs go first, and that is the whole reason this is a script.** Embeddings are
cached per (backbone, config) with a file index, so a cache built over all 1,599 frames
already answers any 500-frame subset of them - the 500-frame run after a full one costs
seconds instead of hours. In the other order, nothing is reused and the 500-frame cache
cannot serve the full run at all.

Each run is a subprocess, exactly as the browser launches it, so the lock, the resume from
cache and the state-file guards all behave the way they do in the live panel. A failure
stops the sequence rather than moving on: the usual cause is a stale lock or a state file
from a different frame count, and both affect every run after it.
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

#: Unbuffered, because this script's own lines share a pipe with its children's. The child
#: writes straight through and Python holds the parent's in a buffer, so a log tailed during
#: a run showed the search's output interleaved with *none* of the progress around it - the
#: run boundaries and the "saved" confirmations only appeared when the whole sequence ended,
#: which is the one moment they are no longer needed.
print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitch_occupancy.api import search_archive  # noqa: E402
from pitch_occupancy.api.search_control import _archive_dir, archive_state  # noqa: E402

RESULTS = ROOT / "results"
STATE = RESULTS / "preprocess_search.json"
LOCK = RESULTS / ".preprocess_search.lock"
SCRIPT = ROOT / "experiments" / "preprocess_search.py"

#: Full runs first so the 500-frame runs can read their embeddings out of the cache. Within
#: that, cheapest backbone first: if the machine is going to be taken away after two hours,
#: the two hours should have finished something.
ORDER = [
    (model, scope, limit)
    for scope, limit in (("all", None), ("500", 500))
    for model in ("convnextv2", "dinov2", "vit")
]


def _read_state() -> dict:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_one(model: str, scope: str, limit: int | None, rounds: int) -> bool:
    """One search, archived into its cell. False if it failed.

    The state file is archived and removed first rather than appended to. Two backbones can
    legally share it, but then a crash halfway through the second leaves a file that is
    neither run, and the frame-count guard rejects the whole thing the moment the scope
    changes. One run, one file, archived the moment it finishes.
    """
    if LOCK.exists():
        print(f"! a search already holds {LOCK.name} - stop it before running this")
        return False

    for cell in archive_state(_read_state()):
        print(f"  archived the previous run as {cell}")
    STATE.unlink(missing_ok=True)

    # Restore this cell's own evaluations so a re-run resumes instead of restarting. The
    # search matches cached entries on (model, hash, n_frames), so anything it already
    # scored under the same conditions is free.
    prior = search_archive.load(model, scope, _archive_dir())
    if prior and prior.get("evaluations"):
        STATE.write_text(
            json.dumps({"generated": prior.get("generated", ""),
                        "evaluations": prior["evaluations"],
                        "best": {model: prior.get("best", {})}}, indent=2),
            encoding="utf-8",
        )
        print(f"  resuming from {len(prior['evaluations'])} saved evaluations")

    cmd = [sys.executable, str(SCRIPT), "--models", model, "--rounds", str(rounds)]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"  {' '.join(cmd[1:])}")
    code = subprocess.run(cmd, cwd=ROOT, check=False).returncode  # noqa: S603
    if code != 0:
        print(f"! {model} {scope} exited {code}; stopping so the rest are not built on it")
        return False

    path = search_archive.save(model, scope, _read_state(), rounds=rounds,
                               base=_archive_dir())
    saved = search_archive.load(model, scope, _archive_dir()) or {}
    n = len(saved.get("evaluations", []))
    scored = sum(1 for e in saved.get("evaluations", []) if _is_scored(e))
    print(f"  saved {n} evaluations ({scored} scored) to {path.relative_to(ROOT)}")
    if n and not scored:
        print("  WARNING: nothing scored. Every fold that could measure recall was empty.")
    return True


def _is_scored(e: dict) -> bool:
    v = e.get("play_recall")
    return isinstance(v, (int, float)) and v == v


def main() -> int:
    ap = argparse.ArgumentParser(description="Run every cell of the saved-search grid.")
    ap.add_argument("--only", nargs="*", choices=sorted(search_archive.MODELS),
                    help="limit to these backbones")
    ap.add_argument("--scope", nargs="*", choices=["all", "500"],
                    help="limit to these frame counts")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="re-run cells already saved")
    ap.add_argument("--check", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    todo = [
        (m, sc, lim) for m, sc, lim in ORDER
        if (not args.only or m in args.only) and (not args.scope or sc in args.scope)
        and (args.force or not search_archive.cell(m, sc, _archive_dir()).exists())
    ]
    if not todo:
        print("every requested cell is already saved - pass --force to run them again")
        return 0

    print(f"{len(todo)} run(s), full-frame first so the 500-frame runs reuse the cache:")
    for m, sc, _ in todo:
        print(f"  {m} · {sc}")
    if args.check:
        return 0

    for i, (model, scope, limit) in enumerate(todo, 1):
        print(f"\n{'=' * 64}\n[{i}/{len(todo)}] {model} · {scope}\n{'=' * 64}")
        if not run_one(model, scope, limit, args.rounds):
            return 1
    print("\nall requested cells saved - the page reads them under 'View old results'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
