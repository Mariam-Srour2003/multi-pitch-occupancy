"""Uncertainty and significance (WP0-T6).

Every comparison the thesis reports goes through here, because a bare accuracy delta on a
few hundred correlated frames is not evidence. Three things are always attached:

* a **confidence interval**, from bootstrap resampling;
* a **paired test** — McNemar at frame level, since the two models saw identical frames
  and an unpaired test throws that pairing away;
* an **effect size**, so "significant" is never mistaken for "large".

And because WP4 and WP5 run dozens of pairwise comparisons, :func:`holm_bonferroni`
corrects the family. Uncorrected p-values across thirty comparisons produce roughly one
false positive by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "BootstrapCI",
    "McNemarResult",
    "bootstrap_ci",
    "bootstrap_metric_ci",
    "paired_bootstrap_diff",
    "paired_bootstrap_metric_diff",
    "mcnemar",
    "holm_bonferroni",
    "cohens_g",
]

DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    estimate: float
    low: float
    high: float
    level: float = 0.95

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"


@dataclass(frozen=True, slots=True)
class McNemarResult:
    n_only_a_correct: int
    n_only_b_correct: int
    statistic: float
    p_value: float
    effect_size: float  # Cohen's g: how lopsided the disagreements are

    @property
    def n_discordant(self) -> int:
        return self.n_only_a_correct + self.n_only_b_correct


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI for any statistic of a sample.

    For accuracy, pass the per-item correctness vector; for macro-F1 use
    :func:`bootstrap_metric_ci`, which resamples items and recomputes the metric.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    dist = np.array([statistic(arr[i]) for i in idx])
    lo, hi = np.percentile(dist, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return BootstrapCI(float(statistic(arr)), float(lo), float(hi), level)


def bootstrap_metric_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """CI for a metric that needs both label vectors, e.g. macro-F1."""
    t = np.asarray(y_true, dtype=object)
    p = np.asarray(y_pred, dtype=object)
    if t.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    dist = np.empty(resamples)
    for r in range(resamples):
        i = rng.integers(0, t.size, t.size)
        dist[r] = metric(t[i], p[i])
    lo, hi = np.percentile(dist, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return BootstrapCI(float(metric(t, p)), float(lo), float(hi), level)


def paired_bootstrap_diff(
    correct_a: Sequence[bool] | np.ndarray,
    correct_b: Sequence[bool] | np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """CI for the difference in accuracy, resampling *items* so the pairing is kept.

    Used at slot level, where McNemar's frame-level independence assumption does not hold.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align: {a.shape} vs {b.shape}")
    return bootstrap_ci(a - b, np.mean, resamples=resamples, level=level, seed=seed)


def paired_bootstrap_metric_diff(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """CI for ``metric(A) - metric(B)``, resampling items **once** per draw.

    The missing primitive that made a confirmed defect possible. :func:`mcnemar` and
    :func:`paired_bootstrap_diff` both work on per-frame *correctness*, so they answer a
    question about accuracy - but this project's primary metric is macro-F1, which is not a
    mean over frames and cannot be recovered from a correctness vector. H2 was consequently
    reported with a macro-F1 delta beside a p-value computed on accuracy, and for one pair
    the two pointed in opposite directions.

    Drawing one index set per resample and scoring both models on it is what keeps the
    comparison paired: the two models then face the same frames in the same proportions, so
    the difference distribution reflects disagreement between models rather than variation
    between samples.

    **Read the interval, not a p-value.** For an equivalence claim - "these two models are
    indistinguishable" - a non-significant difference is absence of evidence, while an
    interval that lies entirely inside a stated margin is evidence of absence. Only the
    second supports the claim, so callers testing equivalence should compare this interval
    against a margin they declared in advance.
    """
    t = np.asarray(y_true, dtype=object)
    a = np.asarray(pred_a, dtype=object)
    b = np.asarray(pred_b, dtype=object)
    if not (t.shape == a.shape == b.shape):
        raise ValueError(f"paired inputs must align: {t.shape} vs {a.shape} vs {b.shape}")
    if t.size == 0:
        raise ValueError("cannot bootstrap an empty sample")

    rng = np.random.default_rng(seed)
    dist = np.empty(resamples)
    for r in range(resamples):
        i = rng.integers(0, t.size, t.size)
        dist[r] = metric(t[i], a[i]) - metric(t[i], b[i])
    lo, hi = np.percentile(dist, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    observed = metric(t, a) - metric(t, b)
    return BootstrapCI(float(observed), float(lo), float(hi), level)


def cohens_g(n_a: int, n_b: int) -> float:
    """Effect size for a paired binary comparison: |proportion - 0.5| of discordants.

    0 means the two models err on each other's cases equally often; 0.5 means one is
    strictly better wherever they disagree.
    """
    n = n_a + n_b
    return 0.0 if n == 0 else abs(n_a / n - 0.5)


def mcnemar(
    correct_a: Sequence[bool] | np.ndarray,
    correct_b: Sequence[bool] | np.ndarray,
    *,
    exact_below: int = 25,
) -> McNemarResult:
    """Paired test on which model is right where they disagree.

    Only the discordant pairs carry information. Below ``exact_below`` of them the exact
    binomial test is used; above it the chi-square approximation with continuity
    correction, which is what makes the two regimes agree closely.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align: {a.shape} vs {b.shape}")

    n01 = int(np.sum(a & ~b))  # only A right
    n10 = int(np.sum(~a & b))  # only B right
    n = n01 + n10

    if n == 0:
        return McNemarResult(n01, n10, 0.0, 1.0, 0.0)

    if n < exact_below:
        from scipy.stats import binomtest

        p = float(binomtest(min(n01, n10), n, 0.5).pvalue)
        stat = float(min(n01, n10))
    else:
        from scipy.stats import chi2

        stat = (abs(n01 - n10) - 1) ** 2 / n
        p = float(chi2.sf(stat, df=1))
    return McNemarResult(n01, n10, float(stat), min(p, 1.0), cohens_g(n01, n10))


def holm_bonferroni(
    p_values: Sequence[float], *, alpha: float = 0.05
) -> tuple[list[float], list[bool]]:
    """Holm–Bonferroni step-down correction for a family of tests.

    Returns the adjusted p-values in the input order and whether each is rejected at
    ``alpha``. Uniformly more powerful than plain Bonferroni, and it is what stops "we ran
    thirty comparisons and three were significant" from being a finding.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return [], []
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist(), (adjusted <= alpha).tolist()
