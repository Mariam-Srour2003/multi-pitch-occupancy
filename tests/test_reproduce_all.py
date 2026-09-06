"""The reproduction pipeline's structure.

The claim it backs - every number in the thesis is recomputable from this repo - fails
quietly if a stage stops being reachable, so the graph is checked rather than trusted."""

from __future__ import annotations

import importlib.util
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
    for s in STAGES:
        target = s.command[-1]
        if target.endswith(".py"):
            assert Path(target).exists(), f"{s.name} -> missing {target}"


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
