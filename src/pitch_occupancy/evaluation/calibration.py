"""Calibration and selective prediction (WP4-T5, WP4-T9, RQ6).

The system routes uncertain frames to a human via the REVIEW band, so its confidence
scores are not decoration - they decide how much manual work the facility is billed for. A
model that is 95% confident should be right 95% of the time; if it is right 80% of the
time, every threshold downstream means something other than it says.

Two things live here:

* **Calibration** - expected calibration error and reliability bins, plus temperature
  scaling, which rescales the logits with a single learned parameter and so cannot change
  any prediction, only its confidence.
* **Risk-coverage** - the selective-prediction view. Sort by confidence, answer only the
  most confident fraction, defer the rest. This turns "how accurate is it?" into the
  question a manager actually asks: *how much human review buys a given reliability?*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ReliabilityBin",
    "expected_calibration_error",
    "reliability_bins",
    "fit_temperature",
    "apply_temperature",
    "risk_coverage_curve",
    "risk_coverage_band",
    "confidence_ties",
    "coverage_for_target_accuracy",
]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Positive means over-confident."""
        return self.mean_confidence - self.accuracy


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, *, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Equal-width confidence bins. Empty bins are omitted rather than reported as zero."""
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if conf.shape != ok.shape:
        raise ValueError(f"inputs must align: {conf.shape} vs {ok.shape}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        out.append(
            ReliabilityBin(
                lo=float(lo), hi=float(hi), n=int(mask.sum()),
                mean_confidence=float(conf[mask].mean()),
                accuracy=float(ok[mask].mean()),
            )
        )
    return out


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, *, n_bins: int = 10
) -> float:
    """ECE: bin-size-weighted mean gap between confidence and accuracy. 0 is perfect."""
    bins = reliability_bins(confidence, correct, n_bins=n_bins)
    n = sum(b.n for b in bins)
    return 0.0 if n == 0 else sum(b.n * abs(b.gap) for b in bins) / n


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax of ``logits / temperature``. T > 1 softens, T < 1 sharpens."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(
    logits: np.ndarray,
    y_true_idx: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> float:
    """Temperature minimising NLL on held-out data, by grid search.

    Grid search rather than gradient descent: one parameter, a well-behaved objective, and
    no optimiser state to get wrong. Because it only rescales logits, the argmax - and so
    every prediction and every accuracy figure - is unchanged. Only confidence moves.
    """
    grid = grid if grid is not None else np.concatenate(
        [np.arange(0.05, 1.0, 0.05), np.arange(1.0, 30.01, 0.1)]
    )
    idx = np.asarray(y_true_idx, dtype=int)
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        p = apply_temperature(logits, float(t))
        nll = -np.log(np.clip(p[np.arange(len(idx)), idx], 1e-12, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)

    # A temperature pinned to either end of the grid means the optimum lies outside it,
    # which in practice means the calibration slice does not resemble what the model will
    # see. Warn rather than return a number that looks like a fitted parameter.
    if best_t in (float(grid[0]), float(grid[-1])):
        import warnings

        warnings.warn(
            f"temperature hit the grid boundary at {best_t:g}; the calibration set is "
            f"probably not representative of the evaluation distribution, and this value "
            f"should not be reported as a fitted parameter",
            RuntimeWarning,
            stacklevel=2,
        )
    return best_t


def risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray, *, points: int = 50
) -> list[tuple[float, float, float]]:
    """``(coverage, accuracy_on_answered, review_rate)`` as the threshold sweeps.

    Coverage is the fraction answered automatically; review rate is its complement - the
    share of slots a human has to look at.

    **Read :func:`risk_coverage_band` before quoting a point on this curve.** "The most
    confident k" is only defined when the k-th and (k+1)-th confidences differ. Where they
    tie, this function returns *one* of the orderings that confidence permits, and which one
    depends on the order the arrays arrived in. On this project's DINOv2 probe the fitted
    temperature hits its grid boundary and **890 of 907 calibrated confidences collapse to
    exactly 1.0**, so almost the whole curve was an arbitrary choice among tied frames -
    and it moved by up to 0.125 between runs for that reason alone.
    """
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    # Stable sort so a given input order always gives the same answer. It does not make the
    # answer *identified* - that is what the band is for - only reproducible.
    order = np.argsort(-conf, kind="stable")  # most confident first
    ok_sorted = ok[order]

    n = len(conf)
    out: list[tuple[float, float, float]] = []
    for k in np.unique(np.linspace(1, n, points).astype(int)):
        acc = float(ok_sorted[:k].mean())
        cov = k / n
        out.append((cov, acc, 1.0 - cov))
    return out


def confidence_ties(confidence: np.ndarray) -> tuple[int, int]:
    """``(n_tied, largest_group)`` - how much of a confidence vector is not orderable.

    A diagnostic to run *before* a risk-coverage curve is plotted. ``n_tied`` counts the
    scores sharing their exact value with at least one other, so it is the number of frames
    whose position in the curve is decided by something other than the model.
    """
    conf = np.asarray(confidence, dtype=float)
    if conf.size == 0:
        return 0, 0
    _, counts = np.unique(conf, return_counts=True)
    return int(counts[counts > 1].sum()), int(counts.max())


def risk_coverage_band(
    confidence: np.ndarray, correct: np.ndarray, *, points: int = 50
) -> list[tuple[float, float, float, float]]:
    """``(coverage, accuracy_worst, accuracy_best, review_rate)`` - the curve as a band.

    The honest form of :func:`risk_coverage_curve`. At a coverage whose boundary falls
    inside a group of equally confident frames, the model does not say which of them to
    answer, so accuracy there is not a number but an interval: best case answers the
    correct members of the tie group first, worst case answers the incorrect ones first.
    Where confidences are distinct the two bounds coincide and the band is the curve.

    **Report the lower bound against any target.** A promise of "99% precision at 40%
    coverage" that only holds for a favourable ordering of indistinguishable frames is not
    a promise about the model.
    """
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if conf.shape != ok.shape:
        raise ValueError(f"paired inputs must align: {conf.shape} vs {ok.shape}")
    n = conf.size
    if n == 0:
        raise ValueError("cannot build a risk-coverage band from an empty sample")

    order = np.argsort(-conf, kind="stable")
    conf_sorted, ok_sorted = conf[order], ok[order]

    # Within each run of equal confidence, sort correct-first for the best case and
    # incorrect-first for the worst. Every prefix of the result is then the most (and least)
    # favourable answer set of that size that confidence alone permits.
    best = ok_sorted.copy()
    worst = ok_sorted.copy()
    start = 0
    for end in [*np.flatnonzero(np.diff(conf_sorted) != 0) + 1, n]:
        group = ok_sorted[start:end]
        n_ok = int(group.sum())
        best[start:end] = [True] * n_ok + [False] * (len(group) - n_ok)
        worst[start:end] = [False] * (len(group) - n_ok) + [True] * n_ok
        start = end

    cum_best, cum_worst = np.cumsum(best), np.cumsum(worst)
    out: list[tuple[float, float, float, float]] = []
    for k in np.unique(np.linspace(1, n, points).astype(int)):
        out.append((k / n, float(cum_worst[k - 1] / k), float(cum_best[k - 1] / k), 1.0 - k / n))
    return out


def coverage_for_target_accuracy(
    confidence: np.ndarray, correct: np.ndarray, target: float = 0.99
) -> tuple[float, float] | None:
    """Largest coverage whose automated answers still reach ``target`` accuracy.

    Returns ``(coverage, review_rate)``, or ``None`` if even the single most confident
    prediction misses the target - which is itself a reportable result.

    **Judged on the band's lower bound**, so the answer is one the model can be held to
    however tied frames happen to be ordered. This is the number that sets how much human
    review the facility is billed for; taking the favourable ordering of indistinguishable
    frames would quote an operating point that does not exist.
    """
    band = risk_coverage_band(confidence, correct, points=200)
    feasible = [(cov, rev) for cov, acc_lo, _, rev in band if acc_lo >= target]
    return max(feasible, key=lambda t: t[0]) if feasible else None
