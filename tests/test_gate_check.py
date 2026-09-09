"""The milestone gate check (WP2-T7, WP4-T8).

Its job is to stop a hand-ticked tracker drifting from the repository, so the tests are about
whether it can be wrong in a *plausible* way — a check that reports a believable incorrect
number is worse than one that fails, because nobody looks twice at it.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
gc = importlib.import_module("experiments.gate_check")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return gc.evaluate()


def test_every_criterion_returns_one_of_the_three_states(rows) -> None:
    """Three, not two. "Needs a person" cannot be moved by anything in the repository, and
    reporting it as unmet would say the work was not done when nothing here could do it."""
    assert {r["status"] for r in rows} <= {gc.MET, gc.UNMET, gc.HUMAN}


def test_every_criterion_explains_itself(rows) -> None:
    """A bare tick is what the hand-maintained tracker already offered."""
    for r in rows:
        assert r["detail"], r["criterion"]


def test_a_gate_with_a_human_criterion_never_reads_as_passed(rows) -> None:
    """M1 waits on a supervisor and M7 on submission. If either ever read "passed" the
    checker would be claiming something it cannot know."""
    for gate in ("M1", "M6", "M7"):
        if any(r["status"] == gc.HUMAN for r in rows if r["gate"] == gate):
            assert gc.verdict(rows, gate) != "passed"


def test_an_unmet_criterion_beats_a_human_one_in_the_verdict() -> None:
    """A gate with real work outstanding is "not passed", not "waiting on a person" - the
    second reads as though the repository has done its part."""
    mixed = [
        {"gate": "X", "status": gc.HUMAN, "criterion": "a", "detail": "d"},
        {"gate": "X", "status": gc.UNMET, "criterion": "b", "detail": "d"},
    ]
    assert gc.verdict(mixed, "X") == "not passed"


def test_the_empty_concentration_reads_the_venue_table_not_the_lighting_one() -> None:
    """The bug this check had: the first row beginning `| EMPTY` is the class-by-*lighting*
    table earlier in the same file, so it reported "EMPTY appears in 2 venues" - meaning day
    and night. A plausible wrong number is worse than a failure."""
    if not (ROOT / "results" / "coverage.md").exists():
        pytest.skip("coverage not generated")
    status, detail = gc._empty_concentration_reported()
    assert status == gc.MET
    assert "exactly one venue" in detail, detail


def test_the_claims_criterion_runs_the_verifier_rather_than_grepping(tmp_path, monkeypatch) -> None:
    """Searching the generated page for a bad word would pass or fail on prose.

    Checked by behaviour, not by reading the source: an earlier version of this very test
    asserted the string "stale" was absent from the function, and failed on the docstring
    explaining why grepping is wrong. Feeding the criterion a page full of the word is the
    property that actually matters.
    """
    page = ROOT / "thesis" / "claims.md"
    if not page.exists():
        pytest.skip("claims ledger not generated")
    original = page.read_text(encoding="utf-8")
    try:
        page.write_text(original + "\n\nstale stale not checked stale\n", encoding="utf-8")
        status, _ = gc._claims_verified()
        assert status == gc.MET, "the criterion reacted to prose rather than to the claims"
    finally:
        page.write_text(original, encoding="utf-8")


def test_the_fusion_criterion_needs_the_ablation_not_just_the_module() -> None:
    """M4 asks for a module *ablated*, and "the file exists" would pass on one that was never
    compared to anything.

    Checked by hiding the comparison file: the criterion must fail while `fusion_head.py` is
    still sitting there. An earlier version keyed on the module's existence alone and passed
    on a multi-section CSV whose comparison rows `csv.DictReader` could not even see.
    """
    comparisons = ROOT / "results" / "fusion_head_comparisons.csv"
    if not comparisons.exists():
        pytest.skip("fusion ablation not generated")
    assert gc._fusion_ablated()[0] == gc.MET

    hidden = comparisons.with_suffix(".csv.hidden")
    comparisons.rename(hidden)
    try:
        status, detail = gc._fusion_ablated()
        assert status == gc.UNMET
        assert "has not been ablated" in detail, detail
    finally:
        hidden.rename(comparisons)


def test_the_stan_criterion_reads_the_caveat_rather_than_the_filename() -> None:
    """WP5-T8 forbids a headline below 30 real labelled slots, so "preliminary" is the
    criterion itself. A STAN that reported a headline off two slots would satisfy "the file
    exists" and violate exactly what the criterion protects.
    """
    results = ROOT / "results" / "stan_preliminary.csv"
    if not results.exists():
        pytest.skip("STAN not run")
    assert gc._stan_preliminary()[0] == gc.MET

    original = results.read_text(encoding="utf-8")
    try:
        results.write_text(original.replace("PRELIMINARY", "HEADLINE"), encoding="utf-8")
        status, detail = gc._stan_preliminary()
        assert status == gc.UNMET
        assert "no preliminary caveat" in detail, detail
    finally:
        results.write_text(original, encoding="utf-8")


def test_gate_names_and_weeks_match_the_tracker() -> None:
    """If a gate is renamed or re-cut in TODO.md and not here, the two disagree silently."""
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    for gate in gc.GATES:
        assert f"| {gate.name} | {gate.week} |" in todo, f"{gate.name} week {gate.week}"


def test_the_generated_status_is_current(rows) -> None:
    if not gc.OUT.exists():
        pytest.skip("gate status not generated")
    assert gc.markdown(rows) == gc.OUT.read_text(encoding="utf-8"), (
        "results/gate_status.md is stale; run `python -m experiments.gate_check`"
    )


def test_the_answer_does_not_depend_on_how_it_was_invoked() -> None:
    """A wrong gate status was committed because of this.

    `python experiments/gate_check.py` puts `experiments/` on the path rather than the repo
    root, so the claims criterion's import failed and it degraded quietly to "not met" -
    turning M3 from passed into not-passed depending on the command used to generate the
    file. Both invocations must agree.
    """
    import subprocess

    outs = []
    for command in (
        [sys.executable, "-m", "experiments.gate_check"],
        [sys.executable, str(ROOT / "experiments" / "gate_check.py")],
    ):
        result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr[-500:]
        outs.append([ln for ln in result.stdout.splitlines() if "passed (" in ln])
    assert outs[0] == outs[1], f"invocation changed the verdict: {outs}"
