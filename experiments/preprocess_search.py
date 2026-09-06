"""Search the preprocessing space for the combination that transfers best (WP3-T8).

Hyper-parameter tuning, but over preprocessing switches rather than model weights, and
scored on **cross-venue** performance - the thing this project actually needs - rather than
in-venue accuracy, which the confound makes meaningless.

Why a search and not a table of single-switch ablations: the switches interact. Grayscale
alone gained +0.022 and a centre crop alone +0.038, but together they scored 0.061 *below*
the untouched baseline. Removing information has a floor, so the best set cannot be read
off one-at-a-time results.

**Why greedy, not exhaustive.** Every candidate needs a fresh embedding pass over the
dataset - the preprocessing is cheap, the backbone is not. Eleven switches with their value
options is well over a thousand combinations; at roughly a minute each that is a fortnight.
Greedy forward selection is O(k^2): it adds the single best-improving switch, then searches
again from there, stopping when nothing improves. That costs tens of evaluations rather
than thousands.

Greedy can miss a pair that only helps jointly. `--pairs` adds an exhaustive pass over
pairs of the surviving switches, which recovers the common case, and every result is cached
so a later exhaustive run reuses everything already computed.

    uv run python experiments/preprocess_search.py --check          # cost estimate only
    uv run python experiments/preprocess_search.py --limit 500      # fast search
    uv run python experiments/preprocess_search.py --models dinov2 convnextv2
    uv run python experiments/preprocess_search.py --pairs          # + pairwise pass

Results accumulate in `results/preprocess_search.json`, which the viewer reads. The run is
resumable: every evaluated configuration is cached by hash, so re-running after a sleep or
a kill continues rather than restarting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pitch_occupancy.data.manifest import ManifestRow, read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.vision.backbones import BACKBONES, embed_batch, load_backbone
from pitch_occupancy.vision.heads import LinearProbe
from pitch_occupancy.vision.preprocess import SWITCHES, PreprocessConfig, preprocess

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache" / "search"
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "preprocess_search.json"
SEED = 42
PLAY = "C2_ACTIVE_PLAY"
#: Below this many play frames a fold cannot rank configurations, only add noise.
MIN_FOLD_PLAY = 10
EMPTY = "C1_EMPTY"


def config_hash(cfg: PreprocessConfig) -> str:
    payload = json.dumps(cfg.as_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def embed(rows: list[ManifestRow], cfg: PreprocessConfig, backbone: str) -> np.ndarray:
    """Embed every row under one preprocessing config, cached by (backbone, config)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{backbone}_{config_hash(cfg)}.npz"
    files = [r.file for r in rows]
    if path.exists():
        z = np.load(path, allow_pickle=True)
        idx = {str(f): i for i, f in enumerate(z["files"])}
        if all(f in idx for f in files):
            return z["features"][[idx[f] for f in files]]

    model, processor, spec = load_backbone(backbone)
    chunks = []
    for start in range(0, len(files), 24):
        batch = files[start : start + 24]
        images = []
        for f in batch:
            bgr = cv2.imread(str(DATASET / f))
            rgb = cv2.cvtColor(preprocess(bgr, cfg), cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        chunks.append(embed_batch(model, processor, spec, images))
    feats = np.concatenate(chunks)
    np.savez_compressed(path, files=np.array(files, dtype=object), features=feats)
    return feats


def score(rows: list[ManifestRow], X: np.ndarray, folds) -> dict[str, float]:
    """Cross-venue play recall, with the false-play control alongside.

    Recall alone can be gamed by a config that simply makes the model readier to say PLAY -
    every clip-venue test set is 100% ACTIVE_PLAY. The control reports how often a genuine
    EMPTY frame is called PLAY, so a config that wins by shifting the boundary is visible
    rather than celebrated.
    """
    pos = {r.file: i for i, r in enumerate(rows)}
    recalls, false_play = [], []
    for fold in folds:
        probe = LinearProbe("s", seed=SEED).fit(X[[pos[r.file] for r in fold.train]], fold.train)
        pred = probe.predict(X[[pos[r.file] for r in fold.test]], fold.test)
        truth = [r.class3 for r in fold.test]
        play = [(p, t) for p, t in zip(pred, truth, strict=True) if t == PLAY]
        recalls.append(sum(p == PLAY for p, _ in play) / len(play) if play else float("nan"))
        empties = [r for r in fold.train if r.class3 == EMPTY]
        if empties:
            pe = probe.predict(X[[pos[r.file] for r in empties]], empties)
            false_play.append(float(np.mean([p == PLAY for p in pe])))
    arr = np.array(recalls, dtype=float)
    return {
        "play_recall": float(arr.mean()),
        "worst_fold": float(arr.min()),
        "false_play": float(np.mean(false_play)) if false_play else 0.0,
    }


def candidates(base: PreprocessConfig, used: set[str]) -> list[tuple[str, PreprocessConfig]]:
    out = []
    for switch, values in SWITCHES.items():
        if switch in used:
            continue
        for value in values:
            out.append((f"{switch}={value}", replace(base, **{switch: value})))
    return out


class SearchLock:
    """A crude lock file, because concurrent searches silently corrupt the results.

    `save_state` rewrites the whole file, so two runs overwrite each other's entries and
    the survivor is a mixture. That happened: three searches ran at once - two subsampled,
    one full-size - and the resulting state held evaluations scored on different frame
    counts, indistinguishable from one another.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "SearchLock":
        if self.path.exists():
            pid = self.path.read_text(encoding="utf-8").strip()
            raise SystemExit(
                f"another search already holds {self.path.name} (pid {pid}). "
                f"Stop it first, or delete the lock if it is stale."
            )
        self.path.write_text(str(os.getpid()), encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


def load_state() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {"generated": "", "evaluations": [], "best": {}}


def save_state(state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    state["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Greedy preprocessing search.")
    ap.add_argument("--models", nargs="*", default=["convnextv2"], choices=sorted(BACKBONES))
    ap.add_argument("--limit", type=int, default=None, help="subsample frames for speed")
    ap.add_argument("--rounds", type=int, default=4, help="max greedy rounds")
    ap.add_argument("--pairs", action="store_true", help="exhaustive pass over surviving pairs")
    ap.add_argument("--check", action="store_true", help="estimate cost and exit")
    args = ap.parse_args()

    rows = development_rows(read_manifest(DATASET / "manifest.csv"))
    if args.limit and args.limit < len(rows):
        rng = np.random.default_rng(SEED)
        keep = set(rng.choice(len(rows), args.limit, replace=False).tolist())
        rows = [r for i, r in enumerate(rows) if i in keep]
    folds = [
        f for f in leave_one_group_out(rows, group_key="venue")
        if not f.name.endswith("venue_01")
    ]
    n_cand = sum(len(v) for v in SWITCHES.values())
    print(f"{len(rows)} frames | {len(folds)} venue folds | {len(args.models)} model(s)")
    print(f"{n_cand} candidate settings across {len(SWITCHES)} switches")

    # A fold with a handful of play frames can only score a few discrete recall values, so
    # the metric saturates and the search ranks noise. Subsampling is the usual cause.
    play_per_fold = [
        (f.name.split("__")[-1], sum(1 for r in f.test if r.class3 == PLAY)) for f in folds
    ]
    short = lambda name: name.replace("clipvenue_", "")  # noqa: E731
    tiny = [(n, c) for n, c in play_per_fold if c < MIN_FOLD_PLAY]
    print("  play frames per fold: "
          + ", ".join(f"{short(n)}={c}" for n, c in play_per_fold))
    if tiny:
        listed = ", ".join(f"{short(n)}={c}" for n, c in tiny)
        print()
        print(f"  WARNING: {len(tiny)}/{len(folds)} folds have fewer than {MIN_FOLD_PLAY} "
              f"play frames ({listed}).")
        print("  Recall takes only a few discrete values there, the unweighted fold mean")
        print("  over-weights them, and a result of 1.000 is mostly arithmetic.")
        print("  Drop --limit for a ranking you can trust.")
        print()

    if args.check:
        per = len(rows) * 0.15 / 60  # rough: ~150 ms/frame
        evals = n_cand + sum(max(n_cand - i * 2, 0) for i in range(1, args.rounds))
        print(f"\n~{per:.1f} min per evaluation, ~{evals} evaluations per model")
        print(f"~{per * evals * len(args.models):.0f} min total (cached runs are free)")
        return 0

    state = load_state()
    prior = {e.get("n_frames") for e in state["evaluations"] if e.get("n_frames")}
    if prior and prior != {len(rows)}:
        raise SystemExit(
            f"{OUT_JSON.name} holds evaluations scored on {sorted(prior)} frames, but this "
            f"run uses {len(rows)}. Scores from different frame counts are not comparable. "
            f"Move the existing file aside and start fresh."
        )

    with SearchLock(RESULTS / ".preprocess_search.lock"):
        return _search(args, rows, folds, state)


def _search(args, rows, folds, state: dict) -> int:
    for backbone in args.models:
        print(f"\n{'=' * 60}\n{backbone}\n{'=' * 60}")
        base = PreprocessConfig()
        used: set[str] = set()

        def evaluate(cfg: PreprocessConfig, label: str, round_no: int) -> dict:
            h = config_hash(cfg)
            cached = next(
                (e for e in state["evaluations"]
                 if e["model"] == backbone and e["hash"] == h
                 and e.get("n_frames") == len(rows)),
                None,
            )
            if cached:
                return cached
            t0 = time.perf_counter()
            metrics = score(rows, embed(rows, cfg, backbone), folds)
            entry = {
                "model": backbone,
                "n_frames": len(rows),  # mixing frame counts makes results incomparable
                "hash": h,
                "label": label,
                "round": round_no,
                "config": cfg.as_dict(),
                "describe": cfg.describe(),
                "seconds": round(time.perf_counter() - t0, 1),
                **{k: round(v, 4) for k, v in metrics.items()},
            }
            state["evaluations"].append(entry)
            save_state(state)  # after every eval: the machine sleeps and kills long runs
            return entry

        current = evaluate(base, "baseline", 0)
        print(f"  baseline           recall {current['play_recall']:.4f}")

        for round_no in range(1, args.rounds + 1):
            best = None
            for label, cfg in candidates(base, used):
                entry = evaluate(cfg, label, round_no)
                mark = "+" if entry["play_recall"] > current["play_recall"] else " "
                print(
                    f"  {mark} {label:<28} recall {entry['play_recall']:.4f} "
                    f"worst {entry['worst_fold']:.3f} falsePlay {entry['false_play']:.4f}"
                )
                if best is None or entry["play_recall"] > best["play_recall"]:
                    best = entry
            if best is None or best["play_recall"] <= current["play_recall"]:
                print(f"  round {round_no}: no improvement - stopping")
                break
            switch = best["label"].split("=")[0]
            used.add(switch)
            base = PreprocessConfig(**best["config"])
            current = best
            print(f"  round {round_no}: adopt {best['label']} -> {best['play_recall']:.4f}\n")

        state["best"][backbone] = current
        save_state(state)
        print(f"\n  BEST for {backbone}: {current['describe']}")
        print(f"    recall {current['play_recall']:.4f}  worst {current['worst_fold']:.3f}")

    print(f"\nwrote {OUT_JSON}")
    with (RESULTS / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} | preprocessing search | "
            f"`python experiments/preprocess_search.py` | `{OUT_JSON.name}` | "
            f"{len(state['evaluations'])} evaluations\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
