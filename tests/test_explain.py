"""XAI: the exact decomposition, and the redaction (WP4-T5).

Two things here could be wrong in a way nobody would notice. A saliency map that does not
correspond to the model's actual score looks exactly like one that does - it is a plausible
picture either way - so the decomposition is tested by reconstruction rather than by eye. And
a redaction step that silently fails produces an unredacted frame that the pipeline reports as
redacted, which is the worst failure in this file because it is irreversible once committed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.vision.explain import (
    SUPPORTS_ATTENTION,
    class_evidence_map,
    evidence_on_people,
    overlay_heatmap,
    pixelate_boxes,
    probe_weights,
    redact_people,
)
from pitch_occupancy.vision.heads import LinearProbe


def rows_and_features(n: int = 80, dim: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    labels = ["C1_EMPTY" if v < 0 else "C2_ACTIVE_PLAY" for v in X[:, 0]]
    rows = [
        ManifestRow(
            file=f"f{i}.jpg", class4="x", class3=labels[i], venue="v", camera="c",
            slot_date="2026-07-11", slot_time="10:00", slot_id="s", t_s=i,
            source="regular", labeled_by="human", lighting="day", quality="ok",
            split_role="",
        )
        for i in range(n)
    ]
    return rows, X


# --- the exactness claim ------------------------------------------------------------------


def test_the_evidence_map_reconstructs_the_score_exactly() -> None:
    """The whole reason this module does not use Grad-CAM. The head is linear over a mean
    pool, so the map is the summands of the score and must add back up to it.

    A tolerance of 1e-9 rather than a loose one: this is float64 over a handful of positions,
    and anything larger is a wrong decomposition rather than accumulated error.
    """
    rows, X = rows_and_features()
    probe = LinearProbe("t").fit(X, rows)
    weights, constant = probe_weights(probe, "C2_ACTIVE_PLAY")

    # Four "positions" whose mean is the pooled feature the probe actually saw.
    rng = np.random.default_rng(1)
    for i in range(5):
        positions = rng.normal(size=(4, X.shape[1]))
        pooled = positions.mean(axis=0)[None, :]
        _, score = class_evidence_map(positions, weights, constant, (2, 2), drop_first=False)
        classes = [str(c) for c in probe.classes_]
        direct = probe._model.decision_function(pooled)
        want = float(direct[0]) if direct.ndim == 1 else float(
            direct[0][classes.index("C2_ACTIVE_PLAY")]
        )
        assert score == pytest.approx(want, abs=1e-9), f"case {i}"


def test_the_map_has_the_grid_shape_it_was_given() -> None:
    rows, X = rows_and_features()
    probe = LinearProbe("t").fit(X, rows)
    weights, constant = probe_weights(probe, "C1_EMPTY")
    positions = np.random.default_rng(2).normal(size=(9, X.shape[1]))
    evidence, _ = class_evidence_map(positions, weights, constant, (3, 3), drop_first=False)
    assert evidence.shape == (3, 3)


def test_dropping_the_cls_token_shrinks_the_picture_not_the_score() -> None:
    """CLS is pooled into the feature the probe saw, so it belongs in the arithmetic. It has
    no position on the image, so it must not appear in the picture. Both at once."""
    rows, X = rows_and_features()
    probe = LinearProbe("t").fit(X, rows)
    weights, constant = probe_weights(probe, "C2_ACTIVE_PLAY")
    positions = np.random.default_rng(3).normal(size=(5, X.shape[1]))  # CLS + 2x2
    with_cls, score_a = class_evidence_map(positions, weights, constant, (2, 2), drop_first=True)
    assert with_cls.shape == (2, 2)
    _, score_b = class_evidence_map(positions, weights, constant, (5, 1), drop_first=False)
    assert score_a == pytest.approx(score_b), "the score must not depend on the picture"


def test_an_unfitted_probe_is_refused() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        probe_weights(LinearProbe("t"), "C1_EMPTY")


def test_an_unknown_class_is_named() -> None:
    rows, X = rows_and_features()
    probe = LinearProbe("t").fit(X, rows)
    with pytest.raises(KeyError, match="C3_MAINTENANCE"):
        probe_weights(probe, "C3_MAINTENANCE_NON_SPORTING")


# --- evidence on people --------------------------------------------------------------------


def test_evidence_concentrated_on_a_box_reads_above_its_area() -> None:
    evidence = np.zeros((8, 8))
    evidence[0:2, 0:2] = 1.0
    share, area = evidence_on_people(evidence, [(0, 0, 25, 25)], (100, 100))
    assert share > area
    assert share / area > 2.0


def test_uniform_evidence_reads_at_exactly_the_area_fraction() -> None:
    """The null this measurement is against: a model whose evidence ignores the people scores
    a ratio of 1. ViT sits at 1.02 on real frames, which is why this must be exact."""
    evidence = np.ones((8, 8))
    share, area = evidence_on_people(evidence, [(0, 0, 50, 100)], (100, 100))
    assert share / area == pytest.approx(1.0, abs=0.02)


def test_negative_evidence_cannot_cancel_positive_evidence() -> None:
    """The map is signed. Summing it raw would let a region arguing *against* the class hide a
    region arguing for it, and report an informative frame as empty of evidence."""
    evidence = np.zeros((8, 8))
    evidence[0:4, :] = 1.0
    evidence[4:8, :] = -1.0
    share, _ = evidence_on_people(evidence, [(0, 0, 100, 50)], (100, 100))
    assert share == pytest.approx(1.0, abs=0.05)


def test_no_boxes_gives_no_share() -> None:
    share, area = evidence_on_people(np.ones((4, 4)), [], (40, 40))
    assert (share, area) == (0.0, 0.0)


# --- redaction -------------------------------------------------------------------------------


def test_pixelation_changes_the_pixels_it_covers_and_nothing_else() -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    out = pixelate_boxes(image, [(0, 0, 30, 30)], blocks=3)
    assert not np.array_equal(out[:30, :30], image[:30, :30])
    assert np.array_equal(out[30:, 30:], image[30:, 30:])


def test_pixelation_actually_destroys_detail() -> None:
    """A pixelation that merely resamples would leave a face recognisable. Variance inside the
    covered region must collapse."""
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    out = pixelate_boxes(image, [(0, 0, 60, 60)], blocks=2)
    assert out.std() < image.std() / 2


def test_a_detector_that_cannot_load_reports_minus_one_rather_than_success() -> None:
    """Failing open here writes an unredacted frame that the pipeline calls redacted. The
    count is the only signal the caller gets, so a failure must be distinguishable from
    'found nobody'."""
    image = np.zeros((32, 32, 3), np.uint8)
    out, found = redact_people(image, model_name="a-model-that-does-not-exist.pt")
    assert found == -1
    assert np.array_equal(out, image)


def test_zero_detections_and_a_failed_detector_are_different_values() -> None:
    """0 means checked and found none; -1 means not checked. If these ever collapse, a
    silently broken detector reads as a frame with no people in it."""
    from pitch_occupancy.vision.explain import detect_people

    assert detect_people(np.zeros((32, 32, 3), np.uint8),
                         model_name="a-model-that-does-not-exist.pt") is None


# --- display ------------------------------------------------------------------------------------


def test_the_overlay_keeps_the_frame_size() -> None:
    image = np.zeros((40, 70, 3), np.uint8)
    assert overlay_heatmap(image, np.random.default_rng(6).normal(size=(4, 4))).shape == image.shape


def test_a_flat_map_does_not_divide_by_zero() -> None:
    image = np.zeros((20, 20, 3), np.uint8)
    assert np.isfinite(overlay_heatmap(image, np.full((4, 4), 3.0))).all()


def test_only_the_transformers_claim_attention() -> None:
    """ConvNeXtV2 has none to roll out, which is architecture rather than an unfinished job."""
    assert set(SUPPORTS_ATTENTION) == {"vit", "dinov2"}


# --- the commitment, across every script that publishes a real frame (2026-09-10) ---------
#
# `thesis/ethics.md` says any frame reproduced in the thesis or slides has faces blurred, and
# this module's docstring says the redaction "is not optional and not a flag". Both were true
# of `make_xai_figures.py` and neither was true of `augmentation_grid.py`, which read real
# night-match frames, drew them into a sheet, committed it to git and served it on the thesis
# site with six unpixelated players in the first tile.
#
# So the commitment is checked against the scripts rather than trusted to them.


def _figure_scripts() -> list[Path]:
    """Experiment scripts that read a frame and write an image into `results/`."""
    root = Path(__file__).resolve().parents[1]
    out = []
    for path in sorted((root / "experiments").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "imread" in src and "imwrite" in src:
            out.append(path)
    return out


def test_every_script_that_publishes_a_real_frame_redacts_it() -> None:
    """A frame written by any of these goes into git and onto the served site. The guard is
    that the script calls the redaction, because nothing downstream can tell whether a
    committed JPEG was redacted or merely looked small enough not to matter."""
    scripts = _figure_scripts()
    assert scripts, "no frame-publishing scripts found - has the glob stopped matching?"
    missing = [p.name for p in scripts
               if "redact" not in p.read_text(encoding="utf-8")]
    assert not missing, (
        "these publish real frames without redacting them: " + ", ".join(missing)
        + ". thesis/ethics.md commits to blurring faces in any published figure."
    )


def test_the_redaction_refuses_rather_than_degrading_when_the_detector_is_missing() -> None:
    """`redact_people` reports -1 when the model will not load, and a caller that treated
    that as "no people found" would publish an unredacted frame that looked checked. The
    contract exists so callers can refuse; this pins the contract itself."""
    import numpy as np

    from pitch_occupancy.vision.explain import redact_people

    frame = np.zeros((32, 32, 3), np.uint8)
    out, n = redact_people(frame, model_name="definitely-not-a-model-file.pt")
    assert n == -1
    assert out.shape == frame.shape
