"""The gated fusion head (WP5-T2).

The module's one job is to make "does the gate help?" answerable, so the tests are about the
ways it could fail to answer that: rungs of the ladder that do not actually differ, a gate
that silently ignores its input, a weight that scales the wrong coordinates, and a fit that
does not reproduce. A wrong number here would be an unfalsifiable modelling claim rather than
a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch_occupancy.slots.fusion_head import (
    GATE_MODES,
    GATE_STATISTIC_NAMES,
    FusionHead,
    build_gate_statistics,
    gate_statistics,
)

DIM = 12
N = 240


def toy(seed: int = 0):
    """Two backbones that disagree, with the disagreement flipping on the gate statistic.

    Getting this fixture right took two attempts and the first failure is the instructive
    one. Making backbone `a` informative on bright frames and `b` on dark ones is *not*
    enough: the shared head reads the concatenation, so it can put weight on both blocks'
    useful coordinate at once and reach 1.000 with no gate at all. Every rung scored 1.000
    and the test proved nothing.

    So here the two blocks carry *opposite* labels, and which one is right flips with the
    gate column: on bright frames the label is the sign of ``a[:, 0]``, on dark frames the
    sign of ``b[:, 0]``, and the other block says the reverse each time. Any fixed weighting
    computes ``(w_a - w_b) * signal`` on one group and its negation on the other, so it can
    win one group only by losing the other. Per-frame routing can win both, and nothing else
    in this architecture can.
    """
    rng = np.random.default_rng(seed)
    bright = rng.random(N) < 0.5
    y = rng.integers(0, 2, N)
    signal = np.where(y == 1, 1.0, -1.0)
    a = rng.normal(size=(N, DIM)) * 0.3
    b = rng.normal(size=(N, DIM)) * 0.3
    flip = np.where(bright, 1.0, -1.0)
    a[:, 0] += 4 * signal * flip
    b[:, 0] -= 4 * signal * flip
    gate = np.column_stack([bright.astype(float), rng.normal(size=N), rng.normal(size=N)])
    return {"a": a, "b": b}, gate, [f"C{v}" for v in y]


# --- the ladder ---------------------------------------------------------------------


def test_the_three_rungs_are_the_documented_ones() -> None:
    assert GATE_MODES == ("uniform", "constant", "mlp")


def test_a_bad_rung_is_rejected_rather_than_silently_treated_as_off() -> None:
    """Defaulting an unknown mode to "no gate" would report an ablation that never ran."""
    with pytest.raises(ValueError, match="unknown gate"):
        FusionHead(["a"], gate="gated")


def test_the_uniform_rung_weights_every_backbone_equally() -> None:
    X, gate, y = toy()
    head = FusionHead(["a", "b"], gate="uniform", epochs=20).fit(X, gate, y)
    assert np.allclose(head.gate_weights(gate), 0.5)
    assert head.report(gate).is_degenerate()


def test_the_constant_rung_learns_one_weighting_and_uses_it_everywhere() -> None:
    """It must differ from uniform - otherwise the middle rung is not a rung - while still
    giving every frame the same weights, which is what separates it from routing."""
    X, gate, y = toy()
    head = FusionHead(["a", "b"], gate="constant", epochs=200).fit(X, gate, y)
    w = head.gate_weights(gate)
    assert np.allclose(w, w[0]), "a constant gate must not vary between frames"
    assert head.report(gate).is_degenerate()
    assert abs(w[0, 0] - 0.5) > 1e-4, "it never moved off the uniform initialisation"


def test_the_mlp_rung_routes_on_the_statistic_it_is_given() -> None:
    """The fixture hides the label in a different backbone depending on the first gate
    column, so a gate that reads its input separates the two groups and one that ignores it
    cannot. Without this the ablation could report a null result caused by a gate that was
    never wired to its input at all."""
    X, gate, y = toy()
    head = FusionHead(["a", "b"], gate="mlp", epochs=400).fit(X, gate, y)
    w = head.gate_weights(gate)[:, 0]
    bright = gate[:, 0] > 0.5
    assert not head.report(gate).is_degenerate()
    assert w[bright].mean() > w[~bright].mean() + 0.05, (
        f"the gate does not separate the two groups: {w[bright].mean():.3f} vs "
        f"{w[~bright].mean():.3f}"
    )


def test_routing_beats_a_constant_weighting_where_routing_is_the_answer() -> None:
    """The positive control for the whole experiment. On data built so that a router wins,
    the mlp rung must beat the constant rung - otherwise a null result on the real data
    would be uninformative about routing and informative only about this implementation."""
    X, gate, y = toy()
    fits = {
        mode: FusionHead(["a", "b"], gate=mode, epochs=400).fit(X, gate, y)
        for mode in GATE_MODES
    }
    scores = {m: np.mean(np.array(h.predict(X, gate)) == np.array(y)) for m, h in fits.items()}
    assert scores["mlp"] > scores["constant"] + 0.02, scores


# --- the wiring that could be quietly wrong -------------------------------------------


def test_the_weights_scale_the_right_block() -> None:
    """`z.view(n, K, d)` assumes the concatenation is block-major. A transposed reshape would
    scale interleaved coordinates instead, still train, and still produce a plausible table -
    so which coordinates a weight touches is pinned rather than assumed.

    Driving one backbone's weight to zero must make that backbone's features irrelevant: two
    inputs differing only in block `b` must then score identically.
    """
    import torch

    from pitch_occupancy.slots.fusion_head import _build_gated_module

    model = _build_gated_module(
        n_backbones=2, block_dim=DIM, gate_dim=1, hidden=4, n_classes=2, gate="constant"
    )
    with torch.no_grad():
        model.logits.copy_(torch.tensor([20.0, -20.0]))  # all weight on block 0
    z = torch.zeros(2, 2 * DIM)
    z[1, DIM:] = 9.0  # differs only inside block 1
    g = torch.zeros(2, 1)
    out = model(z, g)
    assert torch.allclose(out[0], out[1], atol=1e-5), (
        "changing the zero-weighted block changed the output, so the weights are not "
        "scaling contiguous blocks"
    )


def test_a_mismatched_gate_matrix_is_rejected() -> None:
    """The lighting ablation passes one column where the statistics gate passes three. If the
    wrong matrix could be handed over silently, two variants would quietly become one."""
    X, gate, y = toy()
    head = FusionHead(["a", "b"], gate_dim=1)
    with pytest.raises(ValueError, match="expected"):
        head.fit(X, gate, y)


def test_frames_and_gate_rows_must_agree() -> None:
    X, gate, y = toy()
    with pytest.raises(ValueError, match="how many frames"):
        FusionHead(["a", "b"]).fit(X, gate[:10], y)


def test_a_missing_backbone_is_named() -> None:
    X, gate, y = toy()
    with pytest.raises(KeyError, match="c"):
        FusionHead(["a", "c"]).fit(X, gate, y)


def test_a_single_class_training_fold_is_refused() -> None:
    """The venue_01 fold trains on clip venues alone, all ACTIVE_PLAY. A head fitted there
    would predict one class everywhere and score a perfect recall on a play-only test set."""
    X, gate, y = toy()
    with pytest.raises(ValueError, match="only one class"):
        FusionHead(["a", "b"]).fit(X, gate, ["C1"] * len(y))


def test_predicting_before_fitting_says_so() -> None:
    with pytest.raises(RuntimeError, match="was not fitted"):
        FusionHead(["a"]).predict_proba({"a": np.zeros((2, DIM))}, np.zeros((2, 3)))


def test_a_constant_feature_does_not_become_nan() -> None:
    """Dividing by a zero standard deviation would poison the whole head, and the symptom is
    a table of nan rather than an exception."""
    X, gate, y = toy()
    X["a"][:, 3] = 7.0
    head = FusionHead(["a", "b"], epochs=20).fit(X, gate, y)
    assert np.isfinite(head.predict_proba(X, gate)).all()


# --- reproducibility ------------------------------------------------------------------


def test_two_fits_with_the_same_seed_agree_exactly() -> None:
    """Full-batch training has no sampling, so "reproducible" here means bit-identical, not
    close. The grouped-split defect this project already fixed was exactly a point estimate
    that agreed while everything downstream of it did not."""
    X, gate, y = toy()
    a = FusionHead(["a", "b"], epochs=50, seed=7).fit(X, gate, y).predict_proba(X, gate)
    b = FusionHead(["a", "b"], epochs=50, seed=7).fit(X, gate, y).predict_proba(X, gate)
    assert np.array_equal(a, b)


def test_a_different_seed_gives_a_different_fit() -> None:
    """Otherwise the seed is decoration and the previous test proves nothing."""
    X, gate, y = toy()
    a = FusionHead(["a", "b"], epochs=50, seed=7).fit(X, gate, y).predict_proba(X, gate)
    b = FusionHead(["a", "b"], epochs=50, seed=8).fit(X, gate, y).predict_proba(X, gate)
    assert not np.array_equal(a, b)


def test_the_loss_falls():
    X, gate, y = toy()
    losses = FusionHead(["a", "b"], epochs=200).fit(X, gate, y).training_loss
    assert losses[-1] < losses[0] / 2, "300 epochs of Adam should be visibly training"


def test_the_head_stays_small() -> None:
    """The plan caps STAN at 100k parameters and this is a far simpler model. A head that
    grew past that would be overfitting 1,296 frames rather than fusing."""
    X, gate, y = toy()
    head = FusionHead(["a", "b"], epochs=5).fit(X, gate, y)
    assert head.n_parameters < 100_000


# --- the gate statistics --------------------------------------------------------------


def test_the_statistics_are_the_three_the_plan_names() -> None:
    assert GATE_STATISTIC_NAMES == ("brightness", "contrast", "edge_density")


def test_brightness_and_contrast_behave_as_named() -> None:
    dark = np.zeros((16, 16, 3), np.uint8)
    bright = np.full((16, 16, 3), 250, np.uint8)
    noisy = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    assert gate_statistics(dark)[0] < gate_statistics(bright)[0]
    assert gate_statistics(bright)[1] < gate_statistics(noisy)[1]


def test_edge_density_separates_a_flat_frame_from_a_striped_one() -> None:
    flat = np.full((32, 32, 3), 120, np.uint8)
    striped = flat.copy()
    striped[::2] = 0
    assert gate_statistics(striped)[2] > gate_statistics(flat)[2] + 0.05


def test_an_unreadable_frame_is_raised_rather_than_zero_filled(tmp_path) -> None:
    """A zero row is a dark, flat, edgeless frame - the signature of a night shot - so a
    missing file would not look missing to the gate, it would look like night."""
    with pytest.raises(OSError, match="cannot read"):
        build_gate_statistics(["absent.jpg"], tmp_path)
