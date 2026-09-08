"""The reproduction pipeline's structure.

The claim it backs - every number in the thesis is recomputable from this repo - fails
quietly if a stage stops being reachable, so the graph is checked rather than trusted."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "reproduce_all", ROOT / "experiments" / "reproduce_all.py"
)
assert spec and spec.loader
reproduce_all = importlib.util.module_from_spec(spec)
# must be registered before exec: @dataclass resolves cls.__module__ through sys.modules,
# and fails with an opaque AttributeError if the module is not there yet
sys.modules[spec.name] = reproduce_all
spec.loader.exec_module(reproduce_all)
STAGES = reproduce_all.STAGES


def test_stage_names_are_unique() -> None:
    names = [s.name for s in STAGES]
    assert len(names) == len(set(names))


def test_every_stage_declares_an_output() -> None:
    """A stage producing nothing can never be reported done, so it would rerun forever."""
    for s in STAGES:
        assert s.produces, s.name


def test_no_two_stages_claim_the_same_output() -> None:
    seen: dict[Path, str] = {}
    for s in STAGES:
        for p in s.produces:
            assert p not in seen, f"{s.name} and {seen[p]} both produce {p.name}"
            seen[p] = s.name


def test_requirements_are_met_by_an_earlier_stage_or_are_data() -> None:
    """Dependency order must hold: nothing may require an output produced later."""
    produced: set[Path] = set()
    for s in STAGES:
        for need in s.requires:
            is_data = "data" in need.parts and need not in produced
            later = any(need in later_stage.produces for later_stage in STAGES) and need not in produced
            assert is_data or need in produced, (
                f"{s.name} requires {need.name}, which is produced later"
                if later
                else f"{s.name} requires {need.name}, which nothing produces"
            )
        produced.update(s.produces)


def test_every_stage_command_points_at_a_real_target() -> None:
    """Every stage names a module file that exists.

    This used to test `command[-1].endswith(".py")` against a path. Once the stages moved
    to `-m experiments.<name>` no command ended in `.py` any more and the loop passed
    without asserting anything - so it checks the module now, and the next test checks
    that the module actually imports, which is the property that failed in practice.
    """
    for s in STAGES:
        if "-m" not in s.command:
            continue
        module = s.command[s.command.index("-m") + 1]
        if not module.startswith("experiments."):
            continue
        assert (ROOT / "experiments" / f"{module.split('.', 1)[1]}.py").exists(), s.name


@pytest.mark.slow
def test_every_experiment_imports_under_the_invocation_the_pipeline_uses() -> None:
    """The pipeline's claim is that every number is recomputable *from here*.

    `false_play_significance` and `rescore_false_play` import from a sibling experiment,
    and running them as `python experiments/x.py` puts `experiments/` on `sys.path` rather
    than the repository root - so both raised `ModuleNotFoundError` and neither could run
    at all. `--check` reported them done regardless, because their CSVs existed from an
    earlier run and nothing verified the command that produced them still worked.

    Import only: `-c "import experiments.x"` does not execute `main()`, so this costs
    seconds and still catches the failure.
    """
    modules = sorted(
        f"experiments.{p.stem}" for p in (ROOT / "experiments").glob("*.py")
        if p.stem != "__init__"
    )
    broken = []
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            last = [l for l in result.stderr.strip().splitlines() if l.strip()]
            broken.append(f"{module}: {last[-1] if last else 'no output'}")
    assert not broken, "these cannot be run by the pipeline: " + "; ".join(broken)


def test_expensive_stages_are_flagged() -> None:
    """The machine sleeps and kills long runs; anything slow must be visible up front."""
    slow = {s.name for s in STAGES if s.minutes >= 20}
    assert "feature-cache" in slow
    assert "input-ablation" in slow


def test_figures_depend_on_the_experiments_that_feed_them() -> None:
    figs = next(s for s in STAGES if s.name == "figures")
    names = {p.name for p in figs.requires}
    assert "label_efficiency.csv" in names
    assert "h3_cross_venue_recall.csv" in names


def test_satisfied_reports_false_when_an_output_is_missing(tmp_path: Path) -> None:
    stage = reproduce_all.Stage(
        name="x", command=["true"], produces=[tmp_path / "nope.csv"]
    )
    assert not stage.satisfied()


def test_blocked_by_names_the_missing_input(tmp_path: Path) -> None:
    missing = tmp_path / "absent.npz"
    stage = reproduce_all.Stage(
        name="x", command=["true"], produces=[tmp_path / "o.csv"], requires=[missing]
    )
    assert stage.blocked_by() == [missing]


def test_a_stage_with_all_inputs_present_is_not_blocked(tmp_path: Path) -> None:
    present = tmp_path / "there.npz"
    present.write_bytes(b"")
    stage = reproduce_all.Stage(
        name="x", command=["true"], produces=[tmp_path / "o.csv"], requires=[present]
    )
    assert stage.blocked_by() == []


@pytest.mark.parametrize("name", ["h1-h2-baseline-floor", "h3-cross-venue", "label-efficiency"])
def test_cache_dependent_stages_declare_the_cache(name: str) -> None:
    """These read cached features; without the requirement they would fail confusingly
    rather than reporting BLOCKED."""
    stage = next(s for s in STAGES if s.name == name)
    assert any("cache" in p.parts for p in stage.requires), name


def test_the_benchmark_stage_is_marked_machine_dependent() -> None:
    """`--force` means "distrust the cached output". For a stage that measures the machine
    the cached output is the trustworthy one and the rerun is the suspect: swept into a
    batch, ConvNeXtV2's median read 150.9 ms against 101.2 ms idle, moving a ratio quoted
    in three documents."""
    efficiency = next(s for s in STAGES if s.name == "efficiency")
    assert efficiency.machine_dependent


def test_only_stages_that_measure_the_machine_are_exempt_from_force() -> None:
    """The exemption weakens reproduction, so it must stay a short, deliberate list."""
    exempt = {s.name for s in STAGES if s.machine_dependent}
    assert exempt == {"efficiency"}, exempt


def test_a_machine_dependent_stage_still_runs_when_its_output_is_missing() -> None:
    """Held back from --force, never from a genuine reproduction: a missing CSV must still
    be produced or the pipeline would report success having skipped it."""
    stage = reproduce_all.Stage(
        name="bench", command=["true"], produces=[ROOT / "results" / "does_not_exist.csv"],
        machine_dependent=True,
    )
    assert not stage.satisfied()


def test_the_risk_coverage_figure_declares_the_band_it_draws() -> None:
    """It reads the worst/best columns, so it must wait for the experiment that writes them.

    Without the requirement the stage would run against an older two-column CSV and fail
    inside pandas rather than reporting BLOCKED - and the figure's whole point is the band,
    which those columns are.
    """
    figs = next(s for s in STAGES if s.name == "figures")
    assert any(p.name == "rq6_risk_coverage.csv" for p in figs.requires)
    assert any(p.name == "risk_coverage_band.png" for p in figs.produces)


def test_the_claims_ledger_depends_on_what_it_actually_reads() -> None:
    """Its requirements come from the ledger, not from a hand-written list.

    A second copy of the ledger inside the pipeline would be the copy that goes stale -
    which is precisely what the ledger exists to catch, so duplicating it would be a poor
    joke. This checks the dependency is derived: every source named in claims.toml is a
    requirement of the stage.
    """
    import tomllib

    ledger = ROOT / "thesis" / "claims.toml"
    if not ledger.exists():
        pytest.skip("ledger not present")
    named = {
        ROOT / c["source"]
        for c in tomllib.loads(ledger.read_text(encoding="utf-8"))["claim"]
        if c.get("source")
    }
    stage = next(s for s in STAGES if s.name == "claims-ledger")
    assert named, "the ledger checks nothing"
    assert named <= set(stage.requires), (
        "the stage does not require every artefact the ledger reads: "
        f"{sorted(p.name for p in named - set(stage.requires))}"
    )


def test_the_claims_ledger_runs_last() -> None:
    """It checks artefacts every other stage writes, so running it earlier would verify a
    previous run's numbers and report success on a pipeline that had not finished."""
    assert STAGES[-1].name == "claims-ledger"
