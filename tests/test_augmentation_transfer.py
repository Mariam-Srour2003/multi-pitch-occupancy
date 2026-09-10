"""Augmentation across the camera boundary (WP3-T6).

The experiment's headline is a large gain from one preset, which is exactly the shape of
result that deserves suspicion: augmentation multiplies the training set, and more rows can
flatter a model on their own. So the tests are mostly about the control that separates variety
from row count, and about the second finding - that turning everything on erases the gain -
since that one runs against instinct and will be the first thing an editor "corrects".
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
at = importlib.import_module("experiments.augmentation_transfer")

OUT = ROOT / "results" / "augmentation_transfer.csv"
SPREAD = ROOT / "results" / "augmentation_transfer_spread.csv"


def published() -> dict[str, dict]:
    """The published draw, one row per preset.

    The table carries several draws now, so this selects seed `at.SEED` explicitly rather
    than folding by preset - which would have silently handed every assertion below whichever
    draw happened to be written last.
    """
    if not OUT.exists():
        pytest.skip("augmentation transfer not run")
    rows = list(csv.DictReader(OUT.open(encoding="utf-8")))
    chosen = {}
    for row in rows:
        if row["preset"] not in chosen or str(row["seed"]) == str(at.SEED):
            chosen[row["preset"]] = row
    return chosen


# --- the control ------------------------------------------------------------------------


def test_the_duplicate_rows_control_is_run() -> None:
    """Without it, a gain from augmentation cannot be told from a gain from having more
    rows. It is the first entry in ORDER so the table reads control-first."""
    assert at.ORDER[0] == "none"
    assert "none" in published()


def test_the_control_adds_the_same_number_of_rows_as_a_real_preset() -> None:
    """It is only a control if it differs in variety alone."""
    rows = published()
    assert rows["none"]["n_train"] == rows["light"]["n_train"]


def test_the_control_still_rules_out_row_count_in_the_draw_that_gained() -> None:
    """The control's reasoning survives the retraction and is worth keeping separable from
    it. In the one draw that gained, duplication alone did *not* reproduce the gain - so
    whatever happened there was not about training-set size. What the draw was not is a
    different question from what it was, and only the second one was wrong."""
    rows = published()
    base = float(rows["baseline"]["macro_f1"])
    assert float(rows["none"]["macro_f1"]) <= base
    assert float(rows["light"]["macro_f1"]) > base + 0.2  # the seed-42 row, still itself


# --- the two findings -----------------------------------------------------------------------


def test_one_draw_recovered_empty_recall_from_zero() -> None:
    """What the seed-42 draw did, stated as what it is: one draw. The unaugmented probe calls
    none of the unseen camera's empty frames empty, and in this draw augmentation moved that
    to 0.687 - which the four other draws did not reproduce. Kept because the draw is real
    and reproducible; renamed because the old name asserted a property of the method."""
    rows = published()
    assert float(rows["baseline"]["empty_recall"]) == 0.0
    assert float(rows["light"]["empty_recall"]) > 0.5


def test_turning_everything_on_did_not_beat_the_one_draw_that_gained() -> None:
    """`full` contains every effect `light` has, and in the published draw it scored below
    it. That comparison is now between one draw of each, which is worth remembering before
    it is quoted as "match the augmentation to the shift": `full` has one draw, and `light`
    needed five to show that its own number moved by 0.5."""
    rows = published()
    assert float(rows["full"]["macro_f1"]) < float(rows["light"]["macro_f1"])


def test_it_does_not_reach_what_one_labelled_frame_reaches() -> None:
    """Augmentation is not a substitute for the five-frame recipe, and the deck says so. The
    comparison is against the onboarding table rather than a number typed here."""
    onboarding = ROOT / "results" / "onboarding_cost.csv"
    if not onboarding.exists():
        pytest.skip("onboarding curve not generated")
    one_frame = next(
        float(r["adapted_mean"]) for r in csv.DictReader(onboarding.open(encoding="utf-8"))
        if r["backbone"] == "dinov2" and r["k"] == "1"
    )
    assert float(published()["light"]["macro_f1"]) < one_frame


# --- the harness ------------------------------------------------------------------------------


def test_every_preset_in_the_order_is_a_real_one() -> None:
    from pitch_occupancy.vision.augment import AUGMENTATIONS

    assert set(at.ORDER) <= set(AUGMENTATIONS)


def test_the_table_is_written_after_every_preset(tmp_path, monkeypatch) -> None:
    """A run that keeps nothing until it finishes is one crash away from having done nothing,
    which is exactly what happened on the first attempt: it died during the fourth preset and
    lost the three already complete."""
    monkeypatch.setattr(at, "OUT", tmp_path / "partial.csv")
    at.save([{"preset": "none", "views": 4, "n_train": 10, "macro_f1": 0.5,
              "empty_recall": 0.1, "play_recall": 0.9}])
    written = list(csv.DictReader((tmp_path / "partial.csv").open(encoding="utf-8")))
    assert written and written[0]["preset"] == "none"


def test_the_stand_in_row_carries_a_label_into_the_probe() -> None:
    """Augmented views are not manifest rows; they only need to answer `class3`."""
    row = at.Row("aug_light_3", "C1_EMPTY")
    assert row.class3 == "C1_EMPTY"


def test_augmented_views_differ_from_each_other() -> None:
    """A preset that silently did nothing would produce identical views and a null result
    that looked like a measurement."""
    from pitch_occupancy.vision.augment import AUGMENTATIONS, augment

    rng = np.random.default_rng(0)
    image = rng.integers(40, 200, (64, 64, 3), dtype=np.uint8)
    views = [augment(image, AUGMENTATIONS["light"], rng=rng) for _ in range(6)]
    assert any(not np.array_equal(views[0], v) for v in views[1:])


def test_the_none_preset_really_changes_nothing() -> None:
    """Otherwise the control is not a control."""
    from pitch_occupancy.vision.augment import AUGMENTATIONS, augment

    rng = np.random.default_rng(1)
    image = rng.integers(40, 200, (32, 32, 3), dtype=np.uint8)
    assert np.array_equal(augment(image, AUGMENTATIONS["none"], rng=rng), image)


# --- more than one draw ---------------------------------------------------------------------


def spread() -> dict[str, dict]:
    if not SPREAD.exists():
        pytest.skip("augmentation transfer not run")
    return {r["preset"]: r for r in csv.DictReader(SPREAD.open(encoding="utf-8"))}


def test_the_seed_list_is_parsed_and_a_repeat_is_refused() -> None:
    """A repeated seed is not a second draw - it is the same draw counted twice, which would
    shrink the reported spread by pretending to independent evidence."""
    import argparse

    assert at.parse_seeds("42,7, 13") == [42, 7, 13]
    with pytest.raises(argparse.ArgumentTypeError):
        at.parse_seeds("42,42")
    with pytest.raises(argparse.ArgumentTypeError):
        at.parse_seeds("")


def test_the_control_is_the_only_seed_independent_preset() -> None:
    """`draws_nothing` decides which presets are worth re-drawing, and it reads the config
    rather than the name - so a preset quietly emptied of its effects is caught here rather
    than by wondering why four draws produced one number."""
    assert at.draws_nothing("none")
    assert not any(at.draws_nothing(p) for p in at.ORDER if p != "none")


def test_a_second_draw_of_the_control_would_be_identical() -> None:
    """Why `none` is run once and stands for every seed. If this ever fails, the control is
    consuming the generator and the runs are no longer comparable."""
    from pitch_occupancy.vision.augment import AUGMENTATIONS, augment

    image = np.random.default_rng(3).integers(40, 200, (32, 32, 3), dtype=np.uint8)
    first = augment(image, AUGMENTATIONS["none"], rng=np.random.default_rng(42))
    second = augment(image, AUGMENTATIONS["none"], rng=np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_a_second_draw_of_a_real_preset_is_not_identical() -> None:
    """The other half of the same argument: for `light`, the seed has to matter, or the extra
    draws measure nothing at all."""
    from pitch_occupancy.vision.augment import AUGMENTATIONS, augment

    image = np.random.default_rng(3).integers(40, 200, (64, 64, 3), dtype=np.uint8)
    first = augment(image, AUGMENTATIONS["light"], rng=np.random.default_rng(42))
    second = augment(image, AUGMENTATIONS["light"], rng=np.random.default_rng(7))
    assert not np.array_equal(first, second)


def test_a_row_written_before_seeds_existed_is_backfilled_not_dropped(tmp_path, monkeypatch):
    """The first run's rows have no `seed` column. They were all the published draw, and an
    hour of embedding is not worth discarding to keep a schema tidy."""
    old = tmp_path / "old.csv"
    old.write_text(
        "preset,views,n_train,macro_f1,empty_recall,play_recall\n"
        "baseline,0,775,0.4406,0.0000,1.0000\n"
        "light,4,3875,0.8550,0.6872,1.0000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(at, "OUT", old)
    rows = at.load_existing()
    assert {r["preset"]: r["seed"] for r in rows} == {"baseline": "", "light": str(at.SEED)}
    assert at.key(rows[1]) == ("light", str(at.SEED), "4")


def test_a_finished_draw_is_never_recomputed(tmp_path, monkeypatch) -> None:
    """The resume contract. Each draw costs about a quarter of an hour of embedding, so a
    second run that added a seed by redoing the first one would triple the cost of asking for
    one more."""
    monkeypatch.setattr(at, "OUT", tmp_path / "resume.csv")
    at.save([{"preset": "light", "seed": 7, "views": 4, "n_train": 3875,
              "macro_f1": 0.81, "empty_recall": 0.6, "play_recall": 1.0,
              "pred_empty": 40, "n_test": 521}])
    done = {at.key(r) for r in at.load_existing()}
    assert ("light", "7", "4") in done
    assert ("light", "13", "4") not in done


def test_the_control_stands_in_for_every_draw() -> None:
    """`row_for` falls back to the seed-independent control, because a per-draw comparison
    that quietly dropped it would compare the headline against nothing."""
    rows = [{"preset": "none", "seed": 42, "views": 4, "macro_f1": 0.3625},
            {"preset": "light", "seed": 42, "views": 4, "macro_f1": 0.8550}]
    assert at.row_for(rows, "none", 13, 4) is rows[0]
    assert at.row_for(rows, "light", 13, 4) is None


def test_the_summary_reports_the_range_and_not_only_the_mean() -> None:
    """Three draws do not support a confidence interval. The range is what a reader can check
    against the per-draw rows, so it is what gets written."""
    rows = [{"preset": "light", "seed": s, "views": 4, "macro_f1": f, "empty_recall": e}
            for s, f, e in ((42, 0.86, 0.70), (7, 0.80, 0.60), (13, 0.83, 0.65))]
    summary = at.summarise(rows, 4)[0]
    assert summary["n_seeds"] == 3
    assert summary["macro_f1_min"] == 0.80 and summary["macro_f1_max"] == 0.86
    assert abs(summary["macro_f1_mean"] - 0.83) < 1e-9
    assert summary["macro_f1_sd"] > 0


def test_a_single_draw_gets_a_blank_deviation_not_a_zero(tmp_path, monkeypatch) -> None:
    """0.0000 in that cell would say the metric does not move between draws, which is the
    opposite of what one draw establishes."""
    monkeypatch.setattr(at, "SPREAD", tmp_path / "spread.csv")
    at.save_spread(at.summarise([{"preset": "light", "seed": 42, "views": 4,
                                  "macro_f1": 0.855, "empty_recall": 0.687}], 4))
    written = list(csv.DictReader((tmp_path / "spread.csv").open(encoding="utf-8")))
    assert written[0]["macro_f1_sd"] == ""
    assert written[0]["n_seeds"] == "1"


def test_the_stability_check_answers_per_draw_and_can_refute() -> None:
    """The point of the extra draws: a claim that holds in one draw out of three is refuted,
    and a single draw cannot tell that apart from one that holds in three."""
    rows = [
        {"preset": "none", "seed": 42, "views": 4, "macro_f1": 0.36, "empty_recall": 0.0},
        {"preset": "colour", "seed": 42, "views": 4, "macro_f1": 0.35, "empty_recall": 0.0},
        {"preset": "light", "seed": 42, "views": 4, "macro_f1": 0.86, "empty_recall": 0.69},
        {"preset": "weather", "seed": 42, "views": 4, "macro_f1": 0.45, "empty_recall": 0.1},
        {"preset": "full", "seed": 42, "views": 4, "macro_f1": 0.35, "empty_recall": 0.0},
        {"preset": "colour", "seed": 7, "views": 4, "macro_f1": 0.35, "empty_recall": 0.0},
        {"preset": "light", "seed": 7, "views": 4, "macro_f1": 0.44, "empty_recall": 0.0},
        {"preset": "weather", "seed": 7, "views": 4, "macro_f1": 0.45, "empty_recall": 0.1},
        {"preset": "full", "seed": 7, "views": 4, "macro_f1": 0.35, "empty_recall": 0.0},
    ]
    checks = at.stability(rows, [42, 7], 4, baseline_f1=0.4406)
    assert [c["seed"] for c in checks] == [42, 7]
    assert checks[0][at.CLAIM_KEYS[0]] and not checks[1][at.CLAIM_KEYS[0]]
    assert checks[0][at.CLAIM_KEYS[4]] and not checks[1][at.CLAIM_KEYS[4]]


def test_an_incomplete_draw_is_not_scored() -> None:
    """A draw missing a preset would answer "the best preset is light" out of whatever
    happened to finish, which is how a crash becomes a finding."""
    rows = [{"preset": "light", "seed": 13, "views": 4, "macro_f1": 0.9, "empty_recall": 0.8}]
    assert at.stability(rows, [13], 4, baseline_f1=0.44) == []

def test_two_runs_cannot_overlap(tmp_path, monkeypatch) -> None:
    """Resuming made a concurrent run *data-losing*, not merely wasteful: each process reads
    the CSV once at startup and rewrites the whole file after every preset, so whichever
    finishes second erases the other's rows. A quarter of an hour of embedding, deleted by a
    five-second re-run, with nothing to say it had happened."""
    monkeypatch.setattr(at, "LOCK", tmp_path / ".lock")
    with at.only_one_run():
        assert (tmp_path / ".lock").exists()
        with pytest.raises(SystemExit, match="another run holds"):
            with at.only_one_run():
                pass
    assert not (tmp_path / ".lock").exists(), "the lock outlived the run that took it"


def test_the_lock_is_released_when_the_run_raises(tmp_path, monkeypatch) -> None:
    """Otherwise the first crash leaves a lock nobody can explain, and the next person's fix
    is to delete locks reflexively - which is the same as not having one."""
    monkeypatch.setattr(at, "LOCK", tmp_path / ".lock")
    with pytest.raises(ValueError), at.only_one_run():
        raise ValueError("the fourth preset died")
    assert not (tmp_path / ".lock").exists()


def test_an_unknown_preset_is_refused_before_anything_loads() -> None:
    """Extra draws are worth spending on the number in dispute, which is what `--presets` is
    for. A typo has to stop the run: a filter that matched nothing would embed nothing,
    print a table of the rows already on disk, and look exactly like a finished run."""
    import argparse

    args = argparse.Namespace(views=4, seeds="42", presets="ligth")
    with pytest.raises(SystemExit, match="unknown preset"):
        at.run(args)

# --- the retraction ------------------------------------------------------------------------
#
# The headline these tests were written around - `light` recovers empty-pitch recall from
# 0.000 to 0.687 - was one augmentation draw. Re-drawn four more times with the same rows,
# the same probe seed and the same test set, it does not hold. The tests below pin the
# retraction rather than the retracted number, and one of them fails if the spread ever
# narrows enough to make the original claim quotable again, so putting it back has to be a
# deliberate act with new evidence behind it.
#
# Neither of the original two could have caught this. Both read the seed-42 row, which is
# still exactly what it always was. A test that fixes a value detects a change in that value,
# and nothing changed.


def test_the_headline_gain_does_not_survive_a_re_draw() -> None:
    """The retraction itself. If this fails, either the spread has narrowed - in which case
    the write-up needs rewriting *back*, deliberately - or the draws have been thinned out
    until only the flattering one is left."""
    row = spread()["light"]
    assert int(row["n_seeds"]) >= 4, "the retraction rests on the draws; do not delete them"
    assert float(row["macro_f1_min"]) <= 0.4406, (
        "no draw scores at or below the unaugmented baseline any more - the finding has "
        "changed and the retraction in EXPERIMENT_LOG.md is stale"
    )
    assert float(row["macro_f1_max"]) > 0.8, "the published draw should still be in the table"


def test_the_spread_is_wide_enough_to_forbid_quoting_the_mean() -> None:
    """A standard deviation of this size on a metric bounded in [0, 1] is the finding. Quoting
    the mean of these draws as though it described the method is what the retraction exists to
    prevent."""
    row = spread()["light"]
    assert float(row["macro_f1_sd"]) > 0.15


def test_empty_recall_is_zero_in_at_least_one_draw() -> None:
    """The failure that matters operationally: on some draws the probe calls **none** of the
    unseen camera's empty frames empty, which is the state the augmentation was supposed to
    fix."""
    assert float(spread()["light"]["empty_recall_min"]) == 0.0


def test_a_collapsed_preset_predicts_a_single_class() -> None:
    """Why several presets score exactly 0.3479: that is the macro-F1 of answering
    ACTIVE_PLAY to every frame of this test set, not a measurement of the augmentation. The
    `pred_empty` column exists because the repeated value looked wrong."""
    rows = [r for r in csv.DictReader(OUT.open(encoding="utf-8"))
            if r.get("pred_empty") not in (None, "")]
    collapsed = [r for r in rows if abs(float(r["macro_f1"]) - 0.3479) < 5e-4]
    assert collapsed, "no collapsed rows recorded - was pred_empty dropped?"
    for row in collapsed:
        assert int(row["pred_empty"]) == 0, row
        assert float(row["empty_recall"]) == 0.0, row
