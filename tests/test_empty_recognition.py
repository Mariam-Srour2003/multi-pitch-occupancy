"""What the false-play control measures (WP4-T13).

The experiment's claim is that a published number has been read wrongly, so the tests are
about the two things that would make that claim itself wrong: a leakage count that does not
count leakage, and a scoring function that repeats the conflation it is meant to expose.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
er = importlib.import_module("experiments.empty_recognition")


def test_accuracy_and_false_play_are_reported_separately() -> None:
    """The whole finding is that these two are not complements. If `scores` ever derived one
    from the other, the experiment would be asserting its own conclusion."""
    maint = "C3_MAINTENANCE_NON_SPORTING"
    predicted = ["C1_EMPTY", "C2_ACTIVE_PLAY", maint, maint]
    s = er.scores(predicted)
    assert s["empty_accuracy"] == pytest.approx(0.25)
    assert s["false_play_rate"] == pytest.approx(0.25)
    assert s["maintenance_rate"] == pytest.approx(0.5)
    assert s["empty_accuracy"] + s["false_play_rate"] != pytest.approx(1.0), (
        "the third class must be able to absorb the difference"
    )


def test_a_model_can_score_zero_false_play_with_zero_accuracy() -> None:
    """The exact case that started this: 243 frames answered MAINTENANCE."""
    s = er.scores(["C3_MAINTENANCE_NON_SPORTING"] * 243)
    assert s["false_play_rate"] == 0.0
    assert s["empty_accuracy"] == 0.0


class Row:
    def __init__(self, file: str) -> None:
        self.file = file


def test_crossing_pairs_counts_pairs_across_the_boundary_only() -> None:
    """A leak count that also counted within-train pairs would make every protocol look
    leaky and the comparison between them meaningless."""
    cache = {"a": 0b0000, "b": 0b0000, "c": 0b1111_1111}
    train, test = [Row("a")], [Row("b"), Row("c")]
    assert er.crossing_pairs(train, test, cache, threshold=0) == 1


def test_crossing_pairs_is_zero_when_nothing_is_similar() -> None:
    cache = {"a": 0, "b": 0xFFFF_FFFF_FFFF_FFFF}
    assert er.crossing_pairs([Row("a")], [Row("b")], cache, threshold=6) == 0


def test_frames_missing_from_the_hash_cache_are_skipped_not_counted() -> None:
    """An unreadable frame must not silently become a hash of 0, which would collide with
    every dark frame and invent leakage that is not there."""
    cache = {"a": 0}
    assert er.crossing_pairs([Row("a")], [Row("absent")], cache, threshold=6) == 0


def test_the_class_configurations_cover_both_axes() -> None:
    """Dropping C3 and turning off class weighting are different interventions, and the
    experiment's first finding is that neither one rescues EMPTY accuracy. Both must be
    present or the finding is untested."""
    names = {name for name, _, _ in er.CLASS_CONFIGS}
    assert names == {"3class_balanced", "3class_unbalanced",
                     "2class_balanced", "2class_unbalanced"}


def test_the_published_rows_still_reproduce() -> None:
    """The experiment claims to reinterpret published numbers, so it must first produce
    them. If this fails, the reinterpretation is of something else."""
    out = ROOT / "results" / "empty_recognition.csv"
    published = ROOT / "results" / "h3_with_false_play.csv"
    if not out.exists() or not published.exists():
        pytest.skip("results not generated")

    import csv

    reference = {
        r["model"]: float(r["false_play_rate"])
        for r in csv.DictReader(published.open(encoding="utf-8"))
        if r["held_out_venue"] == "MEAN_ACROSS_FOLDS" and r.get("false_play_rate")
    }
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    baseline = {
        r["backbone"]: float(r["false_play_rate"]) for r in rows
        if r["axis"] == "class_configuration" and r["configuration"] == "3class_balanced"
    }
    for model, want in reference.items():
        if model in baseline:
            assert abs(baseline[model] - want) < 0.005, f"{model}: {baseline[model]} vs {want}"


def test_one_labelled_frame_beats_the_published_protocol_for_every_backbone() -> None:
    """The headline. Stated as a comparison rather than as a threshold, so it survives the
    numbers moving; what must not change is the direction."""
    out = ROOT / "results" / "empty_recognition.csv"
    if not out.exists():
        pytest.skip("results not generated")

    import csv

    rows = [r for r in csv.DictReader(out.open(encoding="utf-8"))
            if r["axis"] == "camera_representation"]
    by = {(r["backbone"], r["configuration"]): float(r["empty_accuracy"]) for r in rows}
    backbones = {b for b, _ in by}
    assert backbones, "no camera-representation rows"
    for backbone in backbones:
        published = by[(backbone, "A_only__published")]
        one_shot = by[(backbone, "A_plus_B_play_plus_01_empties")]
        assert one_shot > published + 0.5, (
            f"{backbone}: one labelled frame gave {one_shot:.4f} against {published:.4f}"
        )


def test_every_k_shot_row_carries_its_leakage_count() -> None:
    """0.979 without the 1,400 crossing pairs beside it is the flattering half of the
    finding. The count is what stops the k-shot rows being drawn as a learning curve."""
    out = ROOT / "results" / "empty_recognition.csv"
    if not out.exists():
        pytest.skip("results not generated")

    import csv

    for r in csv.DictReader(out.open(encoding="utf-8")):
        if r["axis"] == "camera_representation":
            assert r["crossing_pairs"] != "", r["configuration"]
            assert r["n_distinct_scenes"], r["configuration"]
