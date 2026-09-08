"""WP4-T1's protocol comparison.

The table it produces is the thesis's first key figure, and its most quotable row is that a
constant predictor scores a perfect macro-F1 under the cross-venue protocol. That claim is
only worth anything if the diagnostics beside it are right, so what is checked here is the
machinery that decides whether a number is quotable: the test-set description, the split
inventory, the zero-shot control's independence from the split, and the guard that refuses
to extend a published table it cannot reproduce.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from pitch_occupancy.data.manifest import ManifestRow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
bm = importlib.import_module("experiments.benchmark_v2")

PLAY, EMPTY, C3 = "C2_ACTIVE_PLAY", "C1_EMPTY", "C3_MAINTENANCE_NON_SPORTING"


def row(name: str, *, venue="v1", slot="s1", cls=EMPTY, date="2026-07-11") -> ManifestRow:
    return ManifestRow(
        file=name, class4="1_empty", class3=cls, venue=venue, camera=f"{slot}_camA",
        slot_date=date, slot_time="10:00", slot_id=slot, t_s=0, source="regular",
        labeled_by="human", lighting="day", quality="unknown", split_role="",
    )


@pytest.fixture
def rows() -> list[ManifestRow]:
    out = []
    for v in ("v1", "v2", "v3"):
        for s in ("a", "b"):
            for i in range(10):
                out.append(row(f"{v}{s}{i}.jpg", venue=v, slot=f"{v}{s}",
                               cls=EMPTY if i < 5 else PLAY))
    return out


# --- the diagnostics that decide whether a score is quotable ------------------


def test_describe_reports_a_single_class_test_set_as_such() -> None:
    """The property behind the headline: a fold whose test side is one class cannot
    distinguish a model from a constant, and the row has to say so."""
    split = bm.Split(
        name="f", train=tuple(row(f"t{i}.jpg", cls=PLAY if i else EMPTY) for i in range(4)),
        test=tuple(row(f"e{i}.jpg", cls=PLAY) for i in range(6)), group_key="venue",
    )
    facts = bm.describe(split)
    assert facts["n_classes_test"] == 1
    assert facts["majority_share_test"] == 1.0
    assert facts["classes_in_test"] == PLAY


def test_describe_measures_the_majority_share_not_the_class_count() -> None:
    """Two classes present is not the same as balanced: 99-to-1 still makes accuracy
    uninformative, and only the share says so."""
    split = bm.Split(
        name="f", train=(), group_key="slot_id",
        test=tuple([row(f"p{i}.jpg", cls=PLAY) for i in range(99)] + [row("e.jpg", cls=EMPTY)]),
    )
    facts = bm.describe(split)
    assert facts["n_classes_test"] == 2
    assert facts["majority_share_test"] == pytest.approx(0.99)


def test_describe_carries_the_split_validator_warnings() -> None:
    split = bm.Split(
        name="f", group_key="slot_id",
        train=tuple(row(f"t{i}.jpg", cls=C3 if i == 0 else PLAY) for i in range(6)),
        test=tuple(row(f"e{i}.jpg", cls=PLAY) for i in range(6)),
    )
    assert "untestable" in bm.describe(split)["warnings"]


# --- the inventory of splits -------------------------------------------------


def test_every_protocol_is_represented(rows) -> None:
    names = {p for p, _ in bm.protocols(rows, [42, 43])}
    assert names == {"random", "grouped_slot", "lo_venue_out", "temporal"}


def test_the_replicated_protocols_get_one_split_per_seed(rows) -> None:
    """A single split is a sample of size one; the grouped protocol's spread across seeds
    turned out to be +-0.24 macro-F1, which is the reason this is not optional."""
    got = bm.protocols(rows, [42, 43, 44])
    assert sum(1 for p, _ in got if p == "random") == 3
    assert sum(1 for p, _ in got if p == "grouped_slot") == 3


def test_a_fold_that_cannot_train_is_skipped_rather_than_scored(capsys) -> None:
    """Leaving out the only venue with more than one class leaves a single-class training
    side. Fitting on it produces a number, and the number would be meaningless."""
    data = [row(f"a{i}.jpg", venue="va", slot=f"sa{i}", cls=PLAY) for i in range(10)]
    data += [row(f"b{i}.jpg", venue="vb", slot=f"sb{i}", cls=EMPTY if i < 5 else PLAY)
             for i in range(10)]
    folds = [s.name for p, s in bm.protocols(data, [42]) if p == "lo_venue_out"]
    assert not any(name.endswith("__vb") for name in folds)
    assert "single-class training side" in capsys.readouterr().out


def test_the_temporal_split_excludes_undated_frames(rows) -> None:
    """Undated frames used to be swept into train by string comparison, and here they are
    a different venue - which would make the "temporal" protocol a venue split."""
    data = rows + [row(f"nd{i}.jpg", venue="v9", slot="s9", date="") for i in range(4)]
    _, split = next((p, s) for p, s in bm.protocols(data, [42]) if p == "temporal")
    placed = {r.file for r in split.train} | {r.file for r in split.test}
    assert not any(f"nd{i}.jpg" in placed for i in range(4))


# --- the zero-shot control ---------------------------------------------------


def test_the_control_answers_the_same_way_whatever_the_split(rows) -> None:
    """This is the whole basis of the subtraction. If fitting could change its answers,
    its movement across protocols would no longer be pure test-set composition."""
    labels = {r.file: PLAY for r in rows}
    model = bm._FrozenLabels(labels)
    subset_a, subset_b = rows[:8], rows[8:16]
    before = model.predict(None, subset_a)
    model.fit(np.zeros((4, 1)), subset_b)   # a fit on entirely different data
    assert model.predict(None, subset_a) == before


def test_the_control_is_not_counted_among_the_trivial_baselines() -> None:
    """It reads the image; it simply never trains. Putting it in the floor would claim a
    model that ignores the pitch had matched the backbones when none had."""
    assert bm.ZERO_SHOT not in bm.TRIVIAL


def test_the_declared_prompt_set_is_not_the_searched_winner() -> None:
    """The search picked its winner on the folds it reports. Importing that choice into a
    table about protocols would import its selection bias with it."""
    import json

    path = ROOT / "results" / "prompt_search_best.json"
    if not path.exists():
        pytest.skip("prompt search results not present")
    best = json.loads(path.read_text(encoding="utf-8"))["best"]
    declared = {c.name: d for c, d in bm.ZERO_SHOT_PROMPTS.descriptors.items()}
    searched = {k[len("desc_"):]: v for k, v in best.items() if k.startswith("desc_")}
    assert declared != searched
    assert len(bm.ZERO_SHOT_PROMPTS.templates) == 5


# --- aggregation and the reproduction guard -----------------------------------


def test_summarise_reports_the_spread_and_the_count() -> None:
    records = [
        {"protocol": "grouped_slot", "model": "dinov2", "macro_f1": v}
        for v in (0.2, 0.4, 0.6)
    ]
    mean, sd, n = bm.summarise(records, "macro_f1")[("grouped_slot", "dinov2")]
    assert mean == pytest.approx(0.4)
    assert sd == pytest.approx(0.2)
    assert n == 3


def test_a_single_replicate_reports_zero_spread_not_nan() -> None:
    """`np.std(ddof=1)` on one value is nan, and a nan would print as a plausible blank."""
    records = [{"protocol": "temporal", "model": "vit", "macro_f1": 0.5}]
    mean, sd, n = bm.summarise(records, "macro_f1")[("temporal", "vit")]
    assert (mean, sd, n) == (0.5, 0.0, 1)


def test_the_reproduction_guard_aborts_when_a_published_number_moved(tmp_path, monkeypatch) -> None:
    published = tmp_path / "h1_h2_baseline_floor.csv"
    published.write_text(
        "split,model,accuracy,macro_f1\nrandom_seed42,dinov2,0.9873,0.9879\n", encoding="utf-8"
    )
    monkeypatch.setattr(bm, "PUBLISHED", published)
    ok = [{"replicate": "random_seed42", "model": "dinov2", "accuracy": 0.9873, "macro_f1": 0.9879}]
    bm.check_reproduces(ok)   # must not raise

    moved = [{"replicate": "random_seed42", "model": "dinov2", "accuracy": 0.9873, "macro_f1": 0.95}]
    with pytest.raises(SystemExit, match="did not reproduce"):
        bm.check_reproduces(moved)


def test_the_reproduction_guard_ignores_rows_the_published_table_lacks(tmp_path, monkeypatch) -> None:
    """Three of the four protocols are new, so most rows have nothing to check against;
    that must not be mistaken for agreement or for failure."""
    published = tmp_path / "h1_h2_baseline_floor.csv"
    published.write_text("split,model,accuracy,macro_f1\n", encoding="utf-8")
    monkeypatch.setattr(bm, "PUBLISHED", published)
    bm.check_reproduces([{"replicate": "temporal_2026-07-12", "model": "vit",
                          "accuracy": 0.1, "macro_f1": 0.1}])
