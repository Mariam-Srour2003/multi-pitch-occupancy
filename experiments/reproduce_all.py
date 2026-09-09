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
    #: Measures the machine rather than the data. `--force` skips these: a benchmark
    #: swept into a batch alongside other work reports the load, not the hardware, and
    #: overwriting a careful measurement with a contended one loses information. Naming
    #: it explicitly rather than remembering it, because it has already happened once -
    #: ConvNeXtV2 read 150.9 ms mid-batch against 101.2 ms idle, moving a ratio quoted in
    #: three documents. Run these with `--only`, deliberately, on an idle machine.
    machine_dependent: bool = False

    def satisfied(self) -> bool:
        return all(p.exists() for p in self.produces)

    def blocked_by(self) -> list[Path]:
        return [p for p in self.requires if not p.exists()]


PY = [sys.executable]


def claim_sources() -> list[Path]:
    """The result files `thesis/claims.toml` says it checks.

    Read from the ledger so the pipeline's dependency is the real one. A hand-written list
    here would be a second copy of the ledger and the copy that goes stale - which is the
    failure the ledger itself exists to catch, so duplicating it would be a poor joke.

    A malformed or absent ledger yields no requirements rather than breaking the listing:
    `--check` should still be able to report on every other stage.
    """
    import tomllib

    ledger = ROOT / "thesis" / "claims.toml"
    try:
        claims = tomllib.loads(ledger.read_text(encoding="utf-8"))["claim"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return []
    return sorted({ROOT / c["source"] for c in claims if c.get("source")})


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
        command=[*PY, "-m", "experiments.h1_h2_baseline_floor"],
        produces=[RESULTS / "h1_h2_baseline_floor.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="split leakage and the trivial-baseline floor",
        minutes=3,
    ),
    Stage(
        name="h3-cross-venue",
        command=[*PY, "-m", "experiments.h3_cross_venue_recall"],
        produces=[RESULTS / "h3_cross_venue_recall.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="cross-venue play recall",
        minutes=2,
    ),
    Stage(
        name="h3-sensitivity",
        command=[*PY, "-m", "experiments.h3_sensitivity_merged_venues"],
        produces=[RESULTS / "h3_sensitivity_merged_venues.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="H3 with the two audited venues merged",
        minutes=2,
    ),
    Stage(
        name="label-efficiency",
        command=[*PY, "-m", "experiments.label_efficiency"],
        produces=[RESULTS / "label_efficiency.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="macro-F1 against labelling budget",
        minutes=3,
    ),
    Stage(
        name="rq6-calibration",
        command=[*PY, "-m", "experiments.rq6_calibration_riskcoverage"],
        produces=[RESULTS / "rq6_calibration.csv", RESULTS / "rq6_risk_coverage.csv",
                  RESULTS / "rq6_reliability.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="calibration and risk-coverage (reports as blocked - see the log)",
        minutes=2,
    ),
    Stage(
        name="input-ablation",
        command=[*PY, "-m", "experiments.input_ablation"],
        produces=[RESULTS / "input_ablation.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="what the model reads - re-embeds per variant",
        minutes=30,
    ),
    Stage(
        name="camera-health",
        command=[*PY, "-m", "experiments.camera_health"],
        produces=[RESULTS / "camera_health.csv", RESULTS / "frame_quality.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="per-camera frame quality; a global threshold would flag one venue",
        minutes=4,
    ),
    Stage(
        name="near-duplicate-audit",
        command=[*PY, "-m", "experiments.near_duplicate_audit"],
        produces=[RESULTS / "near_duplicates.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="quantifies H1's leakage: 37.1% of duplicate pairs straddle a random split",
        minutes=6,
    ),
    Stage(
        name="camera-fingerprint",
        command=[*PY, "-m", "experiments.camera_fingerprint_audit"],
        produces=[RESULTS / "camera_fingerprint.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="proves the (1).mp4 camera suffix swaps between venue_01's two days",
        minutes=1,
    ),
    Stage(
        name="augmentation-grid",
        command=[*PY, "-m", "experiments.augmentation_grid"],
        produces=[RESULTS / "figs" / "augmentation_grid.jpg"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="every augmentation preset over two real frames - the visual check",
    ),
    Stage(
        name="class-balancing",
        command=[*PY, "-m", "experiments.class_balancing"],
        produces=[RESULTS / "class_balancing.csv"],
        # the input ablation writes this cache; class balancing only reads it
        requires=[DATA / "cache" / "ablate_dinov2_full.npz"],
        note="why balancing stays on: unweighted calls 46.5% of empty pitches a match",
        minutes=2,
    ),
    Stage(
        name="h3-false-play",
        command=[*PY, "-m", "experiments.h3_with_false_play"],
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
        command=[*PY, "-m", "experiments.end_to_end_slots"],
        produces=[RESULTS / "end_to_end_slots.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="decision layer on the real slots",
    ),
    Stage(
        name="efficiency",
        command=[*PY, "-m", "experiments.efficiency_latency"],
        produces=[RESULTS / "efficiency_latency.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="latency and 20-camera throughput - hardware-specific, run --only on an idle machine",
        machine_dependent=True,
        minutes=5,
    ),
    Stage(
        name="prompt-search",
        command=[*PY, "-m", "experiments.prompt_search"],
        produces=[RESULTS / "prompt_search.csv", RESULTS / "prompt_search_best.json"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="zero-shot prompt sweep - the RQ1 cold-start baseline",
        minutes=25,
    ),
    Stage(
        name="geometry-probe",
        command=[*PY, "-m", "experiments.geometry_convention_probe"],
        produces=[RESULTS / "geometry_convention_probe.csv"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="WP3-T3: keep the processor geometry; ConvNeXtV2 false-play 0.99 -> 0.02 letterboxed",
        minutes=50,
    ),
    Stage(
        name="benchmark-v2",
        command=[*PY, "-m", "experiments.benchmark_v2"],
        produces=[RESULTS / "benchmark_v2.csv", RESULTS / "benchmark_v2_protocols.json"],
        requires=[DATA / "cache" / "dinov2.npz", RESULTS / "h1_h2_baseline_floor.csv"],
        note="WP4-T1: four split protocols; a constant predictor scores 1.000 cross-venue",
        minutes=12,
    ),
    Stage(
        name="input-path-protocol",
        command=[*PY, "-m", "experiments.input_path_protocol"],
        produces=[RESULTS / "input_path_protocol.csv"],
        requires=[DATA / "cache" / "convnextv2.npz",
                  DATA / "cache" / "geom_probe" / "convnextv2.npz",
                  RESULTS / "geometry_convention_probe.csv"],
        note="WP3-T3(b): the input-path finding reverses under the camera swap - not established",
        minutes=6,
    ),
    Stage(
        name="h6-zero-shot",
        command=[*PY, "-m", "experiments.h6_zero_shot_gap"],
        produces=[RESULTS / "h6_zero_shot_gap.csv"],
        requires=[DATA / "cache" / "clip_image_features.npz", DATA / "cache" / "dinov2.npz",
                  RESULTS / "h1_h2_baseline_floor.csv", RESULTS / "prompt_search_best.json"],
        note="H6: inconclusive - the prompt matters more than the model (0.021-0.747 macro-F1)",
        minutes=9,
    ),
    Stage(
        name="h4-equivalence",
        command=[*PY, "-m", "experiments.h4_model_equivalence"],
        produces=[RESULTS / "h4_model_equivalence.csv"],
        requires=[DATA / "cache" / "vit.npz", RESULTS / "h1_h2_baseline_floor.csv",
                  RESULTS / "efficiency_latency.csv"],
        note="H4: ConvNeXtV2 == ViT exactly because neither ever predicts EMPTY; 1.83x concurrent",
        minutes=8,
    ),
    Stage(
        name="logit-average",
        command=[*PY, "-m", "experiments.logit_average_baseline"],
        produces=[RESULTS / "logit_average_baseline.csv"],
        requires=[DATA / "cache" / "dinov2.npz", RESULTS / "h3_cross_venue_recall.csv"],
        note="WP5-T9 on raw caches; its false-play half is refuted - see logit-average-preproc",
        minutes=6,
    ),
    Stage(
        name="logit-average-preproc",
        command=[*PY, "-m", "experiments.logit_average_baseline",
                 "--cache-dir", str(DATA / "cache" / "geom_probe"), "--suffix", "_preproc"],
        produces=[RESULTS / "logit_average_baseline_preproc.csv"],
        requires=[DATA / "cache" / "geom_probe" / "dinov2.npz"],
        note="WP5-T9 on letterboxed caches: refutes finding 2, strengthens finding 1 to 100%",
        minutes=6,
    ),
    Stage(
        name="fusion-head",
        command=[*PY, "-m", "experiments.fusion_head_ablation"],
        produces=[RESULTS / "fusion_head_ablation.csv",
                  RESULTS / "fusion_head_comparisons.csv",
                  RESULTS / "fusion_head_gate.csv"],
        requires=[DATA / "cache" / "dinov2.npz", RESULTS / "logit_average_baseline.csv"],
        note="WP5-T2: routing is worth -0.024 recall; and no model exceeds 0.165 EMPTY accuracy",
        minutes=9,
    ),
    Stage(
        name="stan-preliminary",
        command=[*PY, "-m", "experiments.stan_preliminary"],
        produces=[RESULTS / "stan_preliminary.csv"],
        requires=[DATA / "cache" / "dinov2.npz", DATA / "processed" / "manifest.csv"],
        note="WP5-T1 against four tuned baselines; preliminary by the WP5-T8 gate - 2 real slots",
        minutes=3,
    ),
    Stage(
        name="empty-recognition",
        command=[*PY, "-m", "experiments.empty_recognition"],
        produces=[RESULTS / "empty_recognition.csv"],
        requires=[DATA / "cache" / "dinov2.npz", RESULTS / "h3_with_false_play.csv"],
        note="WP4-T13: the false-play control is a camera-transfer test; one labelled frame fixes it",
        minutes=6,
    ),
    Stage(
        name="xai-figures",
        command=[*PY, "-m", "experiments.make_xai_figures"],
        produces=[RESULTS / "xai_evidence_focus.csv"],
        requires=[DATA / "cache" / "dinov2.npz", DATA / "processed" / "manifest.csv"],
        note="WP4-T5: exact linear evidence maps, redacted; ViT puts 1.02x evidence on players",
        minutes=5,
    ),
    Stage(
        name="preprocess-search",
        command=[*PY, "-m", "experiments.preprocess_search"],
        produces=[RESULTS / "preprocess_search.json"],
        requires=[DATA / "processed" / "manifest.csv"],
        note="740-min preprocessing sweep - named so it is reported, not omitted; run it deliberately with --only",
        minutes=740,
    ),
    Stage(
        name="search-resolution",
        command=[*PY, "-m", "experiments.search_resolution"],
        produces=[RESULTS / "search_resolution.csv", RESULTS / "search_resolution_gate.json"],
        requires=[RESULTS / "preprocess_search.json", DATA / "cache" / "search"],
        note="WP3-T8: floor 0.0119; weighting changes 3 of 6 winners; CLAHE gate fires on 78%",
        minutes=14,
    ),
    Stage(
        name="h5-preprocessing",
        command=[*PY, "-m", "experiments.h5_preprocessing_switches"],
        produces=[RESULTS / "h5_preprocessing_switches.csv"],
        requires=[RESULTS / "preprocess_search.json", DATA / "cache" / "search"],
        note="H5: ROI clause unrunnable (no polygon); CLAHE clause refuted - it hurts",
        minutes=9,
    ),
    Stage(
        name="rescore-false-play",
        command=[*PY, "-m", "experiments.rescore_false_play"],
        produces=[RESULTS / "false_play_rescored.csv"],
        requires=[RESULTS / "preprocess_search.json"],
        note="repairs the search's false-play control, which scored probes on their own training data",
        minutes=10,
    ),
    Stage(
        name="figures",
        command=[*PY, "-m", "experiments.make_figures"],
        produces=[
            RESULTS / "figs" / f"{n}.png"
            for n in ("label_efficiency", "ranking_inversion", "cross_venue_recall",
                      "risk_coverage_band", "baseline_floor", "accuracy_vs_latency",
                      "leakage_decomposition")
        ],
        requires=[RESULTS / "label_efficiency.csv", RESULTS / "h3_cross_venue_recall.csv",
                  RESULTS / "rq6_risk_coverage.csv", RESULTS / "benchmark_v2.csv",
                  RESULTS / "efficiency_latency.csv", RESULTS / "h3_with_false_play.csv"],
        note="thesis figures, regenerated from the CSVs",
    ),
    # --- three experiments the log quotes that this pipeline used not to reach ----------
    # The docstring's claim is "anything not reachable from here is not reproducible",
    # and the pipeline exited 0 while `prompt_search.csv`, `false_play_rescored.csv` and
    # `preprocess_search.json` - all cited in EXPERIMENT_LOG.md and all committed under
    # results/ - had no stage at all. A green run that silently omits a cited artefact is
    # the exact failure the header warns about, one level up.
    Stage(
        name="error-taxonomy",
        command=[*PY, "-m", "experiments.error_taxonomy"],
        produces=[RESULTS / "error_taxonomy.csv", RESULTS / "error_taxonomy_summary.csv"],
        requires=[DATA / "cache" / "dinov2.npz"],
        note="WP4-T6: 100% of leaky-split errors had a near-duplicate in training; 0% of honest ones",
        minutes=12,
    ),
    Stage(
        name="reconciliation-value",
        command=[*PY, "-m", "experiments.reconciliation_value"],
        produces=[RESULTS / "reconciliation_value.csv"],
        requires=[],
        note="WP6-T10: break-even flag precision ~95%; a EUR10 slot never pays at any precision",
        minutes=1,
    ),
    Stage(
        name="gate-check",
        command=[*PY, "-m", "experiments.gate_check"],
        produces=[ROOT / "results" / "gate_status.md", ROOT / "results" / "gate_status.json"],
        requires=[],
        note="WP2-T7/WP4-T8: milestone criteria checked against the artefacts, not ticked by hand",
        minutes=1,
    ),
    Stage(
        name="claims-ledger",
        command=[*PY, "-m", "experiments.verify_claims"],
        produces=[ROOT / "thesis" / "claims.md"],
        # The artefacts the ledger actually reads, taken from the ledger. Listing them by
        # hand would be a second copy of the ledger, and the copy that goes stale.
        requires=claim_sources(),
        note="WP8-T5: re-derives every quantitative claim from the artefact that produced it",
        minutes=1,
    ),
]

#: Stages too long to belong in a default run, but that must still be *named*: `--check`
#: reports them and the summary says the run was partial, so "everything reproduced"
#: is never printed over a gap. Run them explicitly with `--only preprocess-search`.
LONG_STAGES = {"preprocess-search"}


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

    # Deferred, not dropped. Without --only these are listed with their state and then
    # held back, and the summary below refuses to say "all stages reproduced" while any
    # remain outstanding.
    deferred = [s for s in stages if s.name in LONG_STAGES and not args.only]
    stages = [s for s in stages if s not in deferred]

    print(f"{'stage':<24}{'state':<12}{'~min':>6}  note")
    print("-" * 96)
    todo: list[Stage] = []
    held: list[str] = []
    blocked: list[tuple[Stage, list[Path]]] = []
    for s in stages:
        missing = s.blocked_by()
        if missing:
            state = "BLOCKED"
            blocked.append((s, missing))
        elif s.satisfied() and args.force and s.machine_dependent and not args.only:
            # --force means "distrust the cached output". For a benchmark the cached output
            # is the trustworthy one and the rerun is the suspect, so it is held back and
            # said so rather than silently skipped.
            state = "held (machine)"
            held.append(s.name)
        elif s.satisfied() and not args.force:
            state = "done"
        else:
            state = "to run"
            todo.append(s)
        print(f"{s.name:<24}{state:<12}{s.minutes:>6}  {s.note}")

    outstanding: list[str] = []
    for s in deferred:
        state = "done" if s.satisfied() and not args.force else "DEFERRED"
        if state == "DEFERRED":
            outstanding.append(s.name)
        print(f"{s.name:<24}{state:<12}{s.minutes:>6}  {s.note}")

    if blocked:
        print("\nblocked stages and what they need:")
        for s, missing in blocked:
            for m in missing:
                print(f"  {s.name}: missing {m.relative_to(ROOT)}")

    if held:
        print(
            "\nheld back from --force because they measure the machine, not the data:\n"
            + "\n".join(f"  --only {n}   (run on an idle machine)" for n in held)
        )

    if outstanding:
        print(
            "\ndeferred (too long for a default run, run deliberately):\n"
            + "\n".join(f"  --only {n}" for n in outstanding)
        )

    print(f"\n{len(todo)} stage(s) to run, ~{sum(s.minutes for s in todo)} min")
    if args.check:
        # a blocked stage is a real failure of reproducibility, so say so in the exit code
        return 1 if blocked or outstanding else 0
    if not todo:
        if outstanding:
            print(f"nothing to run here, but {len(outstanding)} deferred stage(s) are outstanding")
        else:
            print("nothing to do - everything is already reproduced")
        return 1 if blocked or outstanding else 0

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
    if blocked or outstanding:
        print(
            f"\nran every stage attempted, but the reproduction is PARTIAL: "
            f"{len(blocked)} blocked, {len(outstanding)} deferred"
        )
        return 1
    print("\nall stages reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
