"""The ensemble's arithmetic, pinned (WP5-T9).

The experiment itself needs feature caches and minutes of probe fitting, so what is tested
here is the part that would fail *silently*: class alignment and the two averaging rules.
An ensemble that averages column 0 of one model with column 0 of another is only meaningful
if both columns mean the same class, and nothing in a shape check would notice if they did
not - the result would just be quietly wrong, which is the failure mode this project has
already met twice in other guises.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "logit_average_baseline", ROOT / "experiments" / "logit_average_baseline.py"
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


lab = _module()
ORDER = ["C1_EMPTY", "C2_ACTIVE_PLAY"]


# --- class alignment -------------------------------------------------------


def test_align_reorders_columns_onto_the_shared_order() -> None:
    p = np.array([[0.2, 0.8]])
    out = lab._align(p, ["C2_ACTIVE_PLAY", "C1_EMPTY"], ORDER)
    assert out.tolist() == [[0.8, 0.2]]


def test_align_is_identity_when_the_order_already_matches() -> None:
    p = np.array([[0.3, 0.7], [0.9, 0.1]])
    assert lab._align(p, ORDER, ORDER).tolist() == p.tolist()


def test_align_raises_rather_than_guessing_on_an_unknown_class() -> None:
    with pytest.raises(KeyError):
        lab._align(np.array([[0.5, 0.5]]), ["C1_EMPTY", "SOMETHING_ELSE"], ORDER)


def test_misaligned_probes_would_otherwise_average_to_the_wrong_class() -> None:
    """The bug the alignment exists to prevent, demonstrated.

    Two probes that agree the frame is EMPTY, reported in opposite column orders. Averaged
    without alignment they produce ACTIVE_PLAY - a confident, wrong answer from two models
    that both got it right.
    """
    a = np.array([[0.60, 0.40]])  # order C1, C2 -> EMPTY
    b = np.array([[0.05, 0.95]])  # order C2, C1 -> EMPTY, strongly

    naive = np.mean([a, b], axis=0)  # [0.325, 0.675]
    assert ORDER[int(naive.argmax())] == "C2_ACTIVE_PLAY", "the bug should be visible"

    aligned = np.mean(
        [lab._align(a, ORDER, ORDER), lab._align(b, ["C2_ACTIVE_PLAY", "C1_EMPTY"], ORDER)],
        axis=0,
    )  # [0.775, 0.225]
    assert ORDER[int(aligned.argmax())] == "C1_EMPTY"


# --- the two averaging rules ----------------------------------------------


def _probs(**kw: np.ndarray) -> dict[str, np.ndarray]:
    return kw


def test_a_unanimous_ensemble_agrees_with_its_members() -> None:
    p = np.array([[0.05, 0.95]])
    preds = lab.predictions(_probs(convnextv2=p, dinov2=p, vit=p), ORDER)
    for name in ("convnextv2", "ens_convnextv2_dinov2", "ens_convnextv2_dinov2_logit"):
        assert preds[name] == ["C2_ACTIVE_PLAY"]


#: One member is almost certain the pitch is EMPTY; the other two are fairly sure it is
#: PLAY. The arithmetic mean goes with the majority (0.60 vs 0.40); the geometric mean does
#: not, because 0.001 for PLAY drags that class's product down far more than two votes of
#: 0.9 lift it. Worked out numerically rather than guessed - the first version of these two
#: tests asserted a difference on an input where the two rules in fact agree.
_VETO = np.array([[0.999, 0.001]])
_LEAN_PLAY = np.array([[0.1, 0.9]])


def test_mean_probability_follows_the_majority() -> None:
    """Arithmetic mean: two votes of 0.9 for PLAY outweigh one near-certainty for EMPTY."""
    preds = lab.predictions(
        _probs(convnextv2=_VETO, dinov2=_LEAN_PLAY, vit=_LEAN_PLAY), ORDER
    )
    assert preds["ens_all_three"] == ["C2_ACTIVE_PLAY"]


def test_mean_log_probability_lets_a_confident_member_veto_instead() -> None:
    """Geometric mean on the *same* input reaches the opposite verdict.

    This is why both are reported rather than one. They differ exactly where the backbones
    disagree, which is the case the whole fusion question is about - and on this data the
    disagreement axis is empty-vs-play, where ConvNeXtV2 calls 99.2% of held-out empty
    pitches a match and DINOv2 30.9%. Which rule is right is an empirical question, so the
    experiment reports both rather than picking one in advance.
    """
    preds = lab.predictions(
        _probs(convnextv2=_VETO, dinov2=_LEAN_PLAY, vit=_LEAN_PLAY), ORDER
    )
    assert preds["ens_all_three_logit"] == ["C1_EMPTY"]
    assert preds["ens_all_three"] != preds["ens_all_three_logit"]


def test_log_averaging_survives_a_zero_probability() -> None:
    """log(0) is -inf and would poison the mean; the clip is what stops a NaN verdict."""
    zero = np.array([[0.0, 1.0]])
    other = np.array([[0.7, 0.3]])
    preds = lab.predictions(_probs(convnextv2=zero, dinov2=other, vit=other), ORDER)
    assert preds["ens_all_three_logit"][0] in ORDER


def test_every_member_and_ensemble_is_reported() -> None:
    p = np.array([[0.4, 0.6]])
    preds = lab.predictions(_probs(convnextv2=p, dinov2=p, vit=p), ORDER)
    for k in ("convnextv2", "dinov2", "vit"):
        assert k in preds
    for name in lab.ENSEMBLES:
        assert name in preds and f"{name}_logit" in preds


def test_the_declared_ensembles_only_name_real_backbones() -> None:
    for members in lab.ENSEMBLES.values():
        assert set(members) <= set(lab.CACHES), members
