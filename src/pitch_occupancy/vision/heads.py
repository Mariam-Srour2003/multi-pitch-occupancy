"""Classifiers on top of frozen features, plus the rule-based floors.

Everything here exposes the same tiny interface — ``fit(X, rows)`` then ``predict(X, rows)``
— so an experiment can put a logistic-regression probe on DINOv2 features and a rule that
reads the clock into the same comparison table without special-casing either.

Rows are passed alongside the feature matrix because some predictors use metadata rather
than pixels. That is the point: :class:`ClockRule` deliberately ignores ``X`` entirely, and
if it keeps up with the probes then the probes are not doing what their name suggests.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.data.taxonomy import Class3

__all__ = ["Predictor", "MajorityClass", "ClockRule", "LinearProbe", "make_predictor"]


@runtime_checkable
class Predictor(Protocol):
    name: str

    def fit(self, X: np.ndarray, rows: Sequence[ManifestRow]) -> "Predictor": ...
    def predict(self, X: np.ndarray, rows: Sequence[ManifestRow]) -> list[str]: ...


class MajorityClass:
    """Predicts the most frequent training label. The true zero point."""

    name = "majority"

    def __init__(self) -> None:
        self._label = Class3.EMPTY.value

    def fit(self, X: np.ndarray, rows: Sequence[ManifestRow]) -> "MajorityClass":
        self._label = Counter(r.class3 for r in rows).most_common(1)[0][0]
        return self

    def predict(self, X: np.ndarray, rows: Sequence[ManifestRow]) -> list[str]:
        return [self._label] * len(rows)


class ClockRule:
    """Predicts from the lighting condition alone — no pixels involved.

    For each lighting value it learns the majority training class, which on the pilot data
    reduces to "night means a match, day means an empty pitch". It scored 98.4% on the
    labelled frames, matching the trained backbones. Its role in the thesis is to make that
    impossible to overlook.
    """

    name = "clock_rule"

    def __init__(self) -> None:
        self._by_lighting: dict[str, str] = {}
        self._fallback = Class3.EMPTY.value

    def fit(self, X: np.ndarray, rows: Sequence[ManifestRow]) -> "ClockRule":
        self._fallback = Counter(r.class3 for r in rows).most_common(1)[0][0]
        groups: dict[str, Counter] = {}
        for r in rows:
            groups.setdefault(r.lighting, Counter())[r.class3] += 1
        self._by_lighting = {k: c.most_common(1)[0][0] for k, c in groups.items()}
        return self

    def predict(self, X: np.ndarray, rows: Sequence[ManifestRow]) -> list[str]:
        return [self._by_lighting.get(r.lighting, self._fallback) for r in rows]


class LinearProbe:
    """Logistic regression on frozen features — the workhorse.

    Class weights are balanced by default: C3 has 6 frames in the whole dataset, and an
    unweighted fit simply never predicts it.
    """

    def __init__(self, name: str = "probe", *, balanced: bool = True, seed: int = 42) -> None:
        self.name = name
        self._balanced = balanced
        self._seed = seed
        self._model = None

    def fit(self, X: np.ndarray, rows: Sequence[ManifestRow]) -> "LinearProbe":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        from sklearn.pipeline import make_pipeline

        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced" if self._balanced else None,
                random_state=self._seed,
            ),
        )
        self._model.fit(X, [r.class3 for r in rows])
        return self

    def predict(self, X: np.ndarray, rows: Sequence[ManifestRow]) -> list[str]:
        if self._model is None:
            raise RuntimeError(f"{self.name} was not fitted")
        return list(self._model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name} was not fitted")
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self._model.classes_  # type: ignore[union-attr]


def make_predictor(kind: str, **kwargs) -> Predictor:
    if kind == "majority":
        return MajorityClass()
    if kind == "clock_rule":
        return ClockRule()
    return LinearProbe(name=kind, **kwargs)
