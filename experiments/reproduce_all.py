"""Regenerate every result and figure from the dataset (WP0-T11).

The claim this backs is the one an examiner is entitled to test: *given the footage, every
number in the thesis can be recomputed from this repository.* Anything not reachable from
here is not reproducible, whatever the write-up says.

    uv run python experiments/reproduce_all.py --check   # what would run, and what is stale
    uv run python experiments/reproduce_all.py           # run everything missing
    uv run python experiments/reproduce_all.py --force   # ignore caches and rerun

Stages run in dependency order. Each declares what it produces, so ``--check`` can report
what is missing without doing any work - useful on a machine that sleeps and kills long
runs, since the expensive stages are resumable by simply running again.

**Stages are not silently skipped when their inputs are absent.** A missing feature cache
is reported as a blocked stage, not treated as success: a reproduction script that exits 0
having done nothing is worse than one that fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    command: list[str]
    produces: list[Path]
    requires: list[Path] = field(default_factory=list)
    note: str = ""
    minutes: int = 1

    def satisfied(self) -> bool:
        return all(p.exists() for p in self.produces)

    def blocked_by(self) -> list[Path]:
        return [p for p in self.requires if not p.exists()]


PY = [sys.executable]

STAGES: list[Stage] = [
    Stage(
        name="manifest",
        command=[*PY, "-m", "pitch_occupancy.cli", "manifest"],
        produces=[DATA / "processed" / "manifest.csv"],
        requires=[DATA / "processed"],
        note="index the labelled frames",
    ),
    Stage(
        name="coverage",
        command=[*PY, "-m", "pitch_occupancy.cli", "coverage"],
        produces=[RESULTS / "coverage.md"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="class x lighting x venue matrix",
    ),
    Stage(
        name="feature-cache",
        command=[*PY, "-m", "pitch_occupancy.cli", "cache"],
        produces=[DATA / "cache" / f"{k}.npz" for k in ("convnextv2", "dinov2", "vit")],
        requires=[DATA / "processed" / "manifest.csv"],
        note="embed every frame once per backbone - the expensive stage",
        minutes=25,
    ),
    Stage(
        name="h1-h2-baseline-floor",
        command=[*PY, str(ROOT / "experiments" / "h1_h2_baseline_floor.py")],
        produces=[RESULTS / "h1_h2_baseline_floor.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="split leakage and the trivial-baseline floor",
        minutes=3,
    ),
    Stage(
        name="h3-cross-venue",
        command=[*PY, str(ROOT / "experiments" / "h3_cross_venue_recall.py")],
        produces=[RESULTS / "h3_cross_venue_recall.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="cross-venue play recall",
        minutes=2,
    ),
    Stage(
        name="h3-sensitivity",
        command=[*PY, str(ROOT / "experiments" / "h3_sensitivity_merged_venues.py")],
        produces=[RESULTS / "h3_sensitivity_merged_venues.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="H3 with the two audited venues merged",
        minutes=2,
    ),
    Stage(
        name="label-efficiency",
        command=[*PY, str(ROOT / "experiments" / "label_efficiency.py")],
        produces=[RESULTS / "label_efficiency.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="macro-F1 against labelling budget",
        minutes=3,
    ),
    Stage(
        name="rq6-calibration",
        command=[*PY, str(ROOT / "experiments" / "rq6_calibration_riskcoverage.py")],
        produces=[RESULTS / "rq6_calibration.csv", RESULTS / "rq6_risk_coverage.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="calibration and risk-coverage (reports as blocked - see the log)",
        minutes=2,
    ),
    Stage(
        name="input-ablation",
        command=[*PY, str(ROOT / "experiments" / "input_ablation.py")],
        produces=[RESULTS / "input_ablation.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="what the model reads - re-embeds per variant",
        minutes=30,
    ),
    Stage(
        name="camera-health",
        command=[*PY, str(ROOT / "experiments" / "camera_health.py")],
        produces=[RESULTS / "camera_health.csv", RESULTS / "frame_quality.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="per-camera frame quality; a global threshold would flag one venue",
        minutes=4,
    ),
    Stage(
        name="near-duplicate-audit",
        command=[*PY, str(ROOT / "experiments" / "near_duplicate_audit.py")],
        produces=[RESULTS / "near_duplicates.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="quantifies H1's leakage: 37.1% of duplicate pairs straddle a random split",
        minutes=6,
    ),
    Stage(
        name="augmentation-grid",
        command=[*PY, str(ROOT / "experiments" / "augmentation_grid.py")],
        produces=[RESULTS / "figs" / "augmentation_grid.jpg"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="every augmentation preset over two real frames - the visual check",
    ),
    Stage(
        name="class-balancing",
        command=[*PY, str(ROOT / "experiments" / "class_balancing.py")],
        produces=[RESULTS / "class_balancing.csv"],
        # the input ablation writes this cache; class balancing only reads it
        requires=[DATA / "cache" / "ablate_dinov2_full.npz"],
        note="why balancing stays on: unweighted calls 46.5% of empty pitches a match",
        minutes=2,
    ),
    Stage(
        name="h3-false-play",
        command=[*PY, str(ROOT / "experiments" / "h3_with_false_play.py")],
        produces=[RESULTS / "h3_with_false_play.csv"],
        # needs the published table too - it reproduces that column before adding its own
        requires=[DATA / "cache" / "dinov2.npz", RESULTS / "h3_cross_venue_recall.csv"],
        note="H3 with the control it never had; ConvNeXtV2 calls 99.2% of empties a match",
        minutes=2,
    ),
    Stage(
        name="false-play-significance",
        command=[*PY, "-m", "experiments.false_play_significance"],
        produces=[RESULTS / "false_play_significance.csv"],
        requires=[RESULTS / "h3_with_false_play.csv"],
        note="CIs, McNemar and Holm - and the de-duplication that overturns them",
        minutes=3,
    ),
    Stage(
        name="effective-sample-audit",
        command=[*PY, "-m", "experiments.effective_sample_audit"],
        produces=[RESULTS / "effective_sample_audit.csv"],
        requires=[RESULTS / "h1_h2_baseline_floor.csv", DATA / "cache" / "dinov2.npz"],
        note="how many distinct scenes each reported test actually rests on",
        minutes=5,
    ),
    Stage(
        name="end-to-end-slots",
        command=[*PY, str(ROOT / "experiments" / "end_to_end_slots.py")],
        produces=[RESULTS / "end_to_end_slots.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="decision layer on the real slots",
    ),
    Stage(
        name="efficiency",
        command=[*PY, str(ROOT / "experiments" / "efficiency_latency.py")],
        produces=[RESULTS / "efficiency_latency.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="latency and 20-camera throughput - hardware-specific, rerun per machine",
        minutes=5,
    ),
    Stage(
        name="figures",
        command=[*PY, str(ROOT / "experiments" / "make_figures.py")],
        produces=[
            RESULTS / "figs" / f"{n}.png"
            for n in ("label_efficiency", "ranking_inversion", "cross_venue_recall")
        ],
        requires=[RESULTS / "label_efficiency.csv", RESULTS / "h3_cross_venue_recall.csv"],
        note="thesis figures, regenerated from the CSVs",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; run nothing")
    ap.add_argument("--force", action="store_true", help="rerun even if outputs exist")
    ap.add_argument("--only", nargs="*", help="run just these stages")
    args = ap.parse_args()

    stages = [s for s in STAGES if not args.only or s.name in args.only]
    if args.only:
        unknown = set(args.only) - {s.name for s in STAGES}
        if unknown:
            print(f"unknown stage(s): {', '.join(sorted(unknown))}")
            return 2

    print(f"{'stage':<24}{'state':<12}{'~min':>6}  note")
    print("-" * 96)
    todo: list[Stage] = []
    blocked: list[tuple[Stage, list[Path]]] = []
    for s in stages:
        missing = s.blocked_by()
        if missing:
            state = "BLOCKED"
            blocked.append((s, missing))
        elif s.satisfied() and not args.force:
            state = "done"
        else:
            state = "to run"
            todo.append(s)
        print(f"{s.name:<24}{state:<12}{s.minutes:>6}  {s.note}")

    if blocked:
        print("\nblocked stages and what they need:")
        for s, missing in blocked:
            for m in missing:
                print(f"  {s.name}: missing {m.relative_to(ROOT)}")

    print(f"\n{len(todo)} stage(s) to run, ~{sum(s.minutes for s in todo)} min")
    if args.check:
        # a blocked stage is a real failure of reproducibility, so say so in the exit code
        return 1 if blocked else 0
    if not todo:
        print("nothing to do - everything is already reproduced")
        return 1 if blocked else 0

    failed: list[str] = []
    for s in todo:
        print(f"\n=== {s.name} ===", flush=True)
        t0 = time.perf_counter()
        result = subprocess.run(s.command, cwd=ROOT)
        dt = (time.perf_counter() - t0) / 60
        if result.returncode != 0:
            print(f"  FAILED after {dt:.1f} min (exit {result.returncode})")
            failed.append(s.name)
        else:
            produced = [p for p in s.produces if p.exists()]
            if len(produced) != len(s.produces):
                # exit 0 with missing outputs has happened here before; do not trust it
                print(f"  exit 0 but only {len(produced)}/{len(s.produces)} outputs exist")
                failed.append(s.name)
            else:
                print(f"  ok in {dt:.1f} min")

    if failed:
        print(f"\n{len(failed)} stage(s) failed: {', '.join(failed)}")
        return 1
    print("\nall stages reproduced")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
