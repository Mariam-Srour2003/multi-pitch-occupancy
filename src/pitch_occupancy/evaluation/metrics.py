"""Classification metrics reported over the 3-class taxonomy.

Macro-F1 leads and accuracy accompanies it. On this dataset accuracy is dominated by the
majority class — a rule that reads only the clock scores 98.4% — so an accuracy figure on
its own says almost nothing about whether occupancy is being classified.

**Classes with zero support are excluded from the macro average and reported as absent.**
Averaging an undefined F1 as 0.0 drags the headline number down for a reason unrelated to
model quality; averaging it as 1.0 inflates it. Either way the number stops meaning what
its name says, and C3 currently has 6 frames in the entire dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pitch_occupancy.data.taxonomy import CLASS3_ORDER

__all__ = ["ClassScore", "Report", "evaluate", "confusion_matrix"]


@dataclass(frozen=True, slots=True)
class ClassScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class Report:
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    per_class: tuple[ClassScore, ...]
    absent_classes: tuple[str, ...] = field(default=())
    n: int = 0

    def __str__(self) -> str:  # pragma: no cover - display only
        lines = [
            f"n={self.n}  accuracy={self.accuracy:.4f}  macro_f1={self.macro_f1:.4f}  "
            f"balanced_acc={self.balanced_accuracy:.4f}",
            f"  {'class':<30}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>7}",
        ]
        for c in self.per_class:
            lines.append(
                f"  {c.label:<30}{c.precision:>8.3f}{c.recall:>8.3f}{c.f1:>8.3f}{c.support:>7}"
            )
        if self.absent_classes:
            lines.append(
                f"  absent from this split (excluded from macro): "
                f"{', '.join(self.absent_classes)}"
            )
        return "\n".join(lines)


def _labels(classes: Sequence[str] | None) -> list[str]:
    return list(classes) if classes else [c.value for c in CLASS3_ORDER]


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], *, classes: Sequence[str] | None = None
) -> np.ndarray:
    """Rows are true classes, columns predicted, in ``classes`` order."""
    labels = _labels(classes)
    index = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        if t in index and p in index:
            m[index[t], index[p]] += 1
    return m


def evaluate(
    y_true: Sequence[str], y_pred: Sequence[str], *, classes: Sequence[str] | None = None
) -> Report:
    """Full report. Classes absent from ``y_true`` are named, not silently averaged."""
    labels = _labels(classes)
    cm = confusion_matrix(y_true, y_pred, classes=labels)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)
    total = int(cm.sum())

    scores: list[ClassScore] = []
    absent: list[str] = []
    for i, label in enumerate(labels):
        if support[i] == 0:
            absent.append(label)
            continue
        p = float(correct[i] / predicted[i]) if predicted[i] else 0.0
        r = float(correct[i] / support[i])
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        scores.append(ClassScore(label, p, r, f1, int(support[i])))

    macro = float(np.mean([s.f1 for s in scores])) if scores else 0.0
    bal = float(np.mean([s.recall for s in scores])) if scores else 0.0
    return Report(
        accuracy=float(correct.sum() / total) if total else 0.0,
        macro_f1=macro,
        balanced_accuracy=bal,
        per_class=tuple(scores),
        absent_classes=tuple(absent),
        n=total,
    )
