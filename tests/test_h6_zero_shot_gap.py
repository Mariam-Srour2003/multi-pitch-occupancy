"""H6's comparison machinery.

The result it produces reads "zero-shot beats every trained probe", and the whole value of
the experiment is the two checks that stop that being the conclusion: a sweep over the
prompt space showing the declared set is a lucky one, and an effective-sample recount. What
is tested here is the parts that decide whether the headline is allowed to stand.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
h6 = importlib.import_module("experiments.h6_zero_shot_gap")

from pitch_occupancy.vision.zeroshot import DECLARED_PROMPT_SET, DESCRIPTORS, TEMPLATES

PLAY, EMPTY, C3 = "C2_ACTIVE_PLAY", "C1_EMPTY", "C3_MAINTENANCE_NON_SPORTING"


# --- the declared prompt set --------------------------------------------------


def test_the_declared_set_is_fixed_by_position_not_by_choice() -> None:
    """"Declared in advance" has to mean something checkable. Position in the descriptor
    list is checkable; "we picked a sensible one" is not."""
    for cls, options in DESCRIPTORS.items():
        assert DECLARED_PROMPT_SET.descriptors[cls] == options[0]
    assert DECLARED_PROMPT_SET.templates == tuple(TEMPLATES)


def test_the_declared_set_is_shared_not_copied() -> None:
    """Two experiments depend on this meaning the same thing, and a drift between two
    copies would be invisible - both would still be "a fixed prompt set"."""
    benchmark = importlib.import_module("experiments.benchmark_v2")
    assert benchmark.ZERO_SHOT_PROMPTS is DECLARED_PROMPT_SET


def test_the_declared_set_is_not_the_search_winner() -> None:
    import json

    path = ROOT / "results" / "prompt_search_best.json"
    if not path.exists():
        pytest.skip("prompt search results not present")
    searched = {
        k[len("desc_"):]: v
        for k, v in json.loads(path.read_text(encoding="utf-8"))["best"].items()
        if k.startswith("desc_")
    }
    declared = {c.name: d for c, d in DECLARED_PROMPT_SET.descriptors.items()}
    assert declared != searched


# --- the paired restriction ----------------------------------------------------


def test_both_predictions_are_cut_the_same_way_as_the_truth() -> None:
    """A paired bootstrap needs both models scored on identical frames. Restricting only
    one vector to the evaluable classes would silently unpair the comparison - which is the
    bug `paired_bootstrap_metric_diff` was written to make impossible in the first place."""
    truth = [PLAY, PLAY, EMPTY, EMPTY, C3]        # C3 has support 1 -> dropped
    a = [PLAY, EMPTY, EMPTY, PLAY, C3]
    b = [PLAY, PLAY, EMPTY, EMPTY, PLAY]
    yt, ya, yb = h6._evaluable_triple(truth, a, b)
    assert len(yt) == len(ya) == len(yb) == 4
    assert C3 not in yt
    assert ya == [PLAY, EMPTY, EMPTY, PLAY]
    assert yb == [PLAY, PLAY, EMPTY, EMPTY]


def test_nothing_is_dropped_when_every_class_is_evaluable() -> None:
    truth = [PLAY, PLAY, EMPTY, EMPTY]
    a, b = [PLAY] * 4, [EMPTY] * 4
    assert h6._evaluable_triple(truth, a, b) == (truth, a, b)


# --- the reproduction guard ----------------------------------------------------


def test_the_guard_aborts_when_a_probe_no_longer_reproduces(tmp_path, monkeypatch) -> None:
    published = tmp_path / "h1_h2_baseline_floor.csv"
    published.write_text(
        "split,model,accuracy,macro_f1\n"
        "grouped_slot_id_seed42,dinov2,0.9846,0.5794\n", encoding="utf-8"
    )
    monkeypatch.setattr(h6, "PUBLISHED", published)
    h6.check_reproduces({"dinov2": (0.5794, 0.98)})          # must not raise
    with pytest.raises(SystemExit, match="did not reproduce"):
        h6.check_reproduces({"dinov2": (0.5000, 0.98)})


def test_the_guard_only_reads_grouped_split_rows(tmp_path, monkeypatch) -> None:
    """The random split's macro-F1 for the same model is 0.9879. Comparing against it would
    abort on a number that was never meant to match."""
    published = tmp_path / "h1_h2_baseline_floor.csv"
    published.write_text(
        "split,model,accuracy,macro_f1\n"
        "random_seed42,dinov2,0.9873,0.9879\n"
        "grouped_slot_id_seed42,dinov2,0.9846,0.5794\n", encoding="utf-8"
    )
    monkeypatch.setattr(h6, "PUBLISHED", published)
    h6.check_reproduces({"dinov2": (0.5794, 0.98)})


def test_the_zero_shot_arm_is_not_checked_against_the_published_table(tmp_path, monkeypatch) -> None:
    """It is new; there is nothing published to check it against, and absence must not read
    as either agreement or failure."""
    published = tmp_path / "h1_h2_baseline_floor.csv"
    published.write_text("split,model,accuracy,macro_f1\n", encoding="utf-8")
    monkeypatch.setattr(h6, "PUBLISHED", published)
    h6.check_reproduces({h6.ZERO_SHOT: (0.6291, 0.7)})


# --- the committed result ------------------------------------------------------


def test_the_reported_verdict_is_not_read_off_the_declared_set_alone() -> None:
    """The declared set beats every probe on all five splits. Left there the verdict would
    be "refuted", and the prompt sweep is what makes it "inconclusive" instead - so the
    committed CSV must carry the sweep beside the comparisons."""
    import csv

    path = ROOT / "results" / "h6_zero_shot_gap.csv"
    if not path.exists():
        pytest.skip("H6 results not present")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert any(r["split"] == "PROMPT_SPACE" for r in rows), "no prompt-space sensitivity row"
    assert any(r["split"] == "SELECTION_BIAS" for r in rows), "no selection-bias row"
    comparisons = [r for r in rows if r["comparison"].startswith(h6.ZERO_SHOT + "_minus_")]
    assert comparisons, "no pairwise comparisons"
    assert all(r["p_holm"] for r in comparisons), "a comparison lacks its corrected p-value"
    assert all(r["cohens_g"] for r in comparisons), "a comparison lacks its effect size"
