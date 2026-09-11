"""Step through a clip one frame at a time, showing what the model looked at (WP4-T5).

`clip_analysis` answers *what* the model decided and when. This answers *why*, frame by
frame, and yields each step as soon as it is computed so a caller can render it while the
next one is still being worked out. Watching it arrive at one frame a second is the point:
a timeline that appears all at once says "trust me", and this says "here is the frame, here
is where the evidence was, here is the score it added up to".

**One forward pass, not two.** The obvious implementation calls the classifier for the
prediction and `spatial_features` for the explanation, which runs the backbone twice and
doubles the cost of every step. The probe is a logistic regression on *mean-pooled* features,
so the pooled vector the classifier would have used is exactly ``features.mean(axis=0)`` -
the same numbers, one pass. Checked rather than assumed: `test_walkthrough.py` asserts the
derived prediction matches the deployed classifier, and it agrees to about 3e-9.

**The evidence map is exact, and says so in its own output.** `class_evidence_map` is not a
saliency heuristic - the score is the mean of the per-position contributions plus a constant,
so the map *sums to the score*. Every step carries ``score_from_map``, ``score_direct`` and
the error between them, because a decomposition claiming to be exact should be checkable by
the person reading it rather than trusted on the strength of a docstring.

**Redaction is off by default here, and that is deliberate rather than careless.** The rule
in `explain.py` - faces pixelated, not optional, not a flag - governs frames *written to
disk*, which in this project means into git history, permanently, under a data-release
decision that is still open. Nothing here is written anywhere: these frames are streamed to
an operator who is already authorised to watch this footage, which is the same category as
`api/routes.evidence_image` serving unredacted evidence to the same person. Turn it on with
``redact=True`` for a screen share, a thesis figure, or anything that leaves the room - it
costs about 640 ms a frame, which is why it is not simply always on.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["Step", "walk_clip", "EXPLAIN_ALL"]

#: ``explain_n=EXPLAIN_ALL`` explains every sampled frame. Explaining costs roughly twice a
#: bare prediction, so a caller that wants the first few steps illustrated and the rest merely
#: classified passes a small number instead.
EXPLAIN_ALL: int = -1


@dataclass(frozen=True, slots=True)
class Step:
    """One sampled frame, and what the model made of it."""

    index: int
    t_s: float
    predicted: str
    confidence: float
    elapsed_ms: float
    #: BGR frame as decoded, redacted only if the caller asked. None once explaining stops.
    frame_bgr: np.ndarray | None = None
    #: The per-position contribution map, in score units, at the backbone's patch grid.
    evidence: np.ndarray | None = None
    grid: tuple[int, int] | None = None
    score_from_map: float | None = None
    score_direct: float | None = None
    n_people: int | None = None
    evidence_on_people: float | None = None
    people_area: float | None = None

    @property
    def explained(self) -> bool:
        return self.evidence is not None

    @property
    def reconstruction_error(self) -> float | None:
        if self.score_from_map is None or self.score_direct is None:
            return None
        return abs(self.score_from_map - self.score_direct)

    @property
    def focus_ratio(self) -> float | None:
        """Positive evidence on people, over the area they occupy.

        The number that says whether the model is reading *players* rather than the scene
        around them: far above 1 means the evidence concentrates on people, near 1 means it
        is spread as though they were not there. None when nobody was detected, because a
        ratio over zero area is not a small number - it is not a number.
        """
        if not self.people_area:
            return None
        return (self.evidence_on_people or 0.0) / self.people_area


def walk_clip(
    path: Path | str,
    classifier,
    *,
    interval_s: float = 10.0,
    explain_n: int = EXPLAIN_ALL,
    redact: bool = False,
    max_samples: int = 360,
) -> Iterator[Step]:
    """Yield one :class:`Step` per sampled frame, as soon as each is computed.

    ``classifier`` is a `vision.classifier.ProbeClassifier` - this needs its backbone and
    probe, not merely something callable, because the explanation decomposes the probe's own
    weights. ``explain_n`` bounds how many frames get the full treatment; the rest are
    predicted and yielded without an evidence map, which is about twice as fast.

    A frame that cannot be decoded is skipped rather than yielded as a guess, exactly as
    `clip_analysis.analyse_clip` records a gap instead of inventing one.
    """
    import time

    import cv2
    import numpy as np

    from pitch_occupancy.vision import explain as X

    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open {source.name} as video")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration_s = frames / fps if fps > 0 and frames > 0 else 0.0
        if duration_s > 0 and duration_s / interval_s > max_samples:
            interval_s = duration_s / max_samples

        classes = list(classifier.probe.classes_)
        index, t = 0, 0.0
        while duration_s <= 0 or t < duration_s:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                if duration_s <= 0:
                    break
                t += interval_s
                continue

            began = time.perf_counter()
            explaining = explain_n == EXPLAIN_ALL or index < explain_n

            if not explaining:
                state, confidence = classifier(frame)
                yield Step(index=index, t_s=t, predicted=str(state), confidence=confidence,
                           elapsed_ms=(time.perf_counter() - began) * 1000)
                index += 1
                t += interval_s
                continue

            # One pass. The pooled vector below is what `embed_batch` would have produced,
            # so the prediction is the deployed one rather than a lookalike.
            feats, grid = X.spatial_features(
                classifier.model, classifier.processor, classifier.spec, frame
            )
            pooled = feats.mean(axis=0, keepdims=True)
            proba = classifier.probe.predict_proba(pooled)[0]
            predicted = classes[int(proba.argmax())]
            confidence = float(proba.max())

            weights, constant = X.probe_weights(classifier.probe, predicted)
            evidence, score_from_map = X.class_evidence_map(
                feats, weights, constant, grid, drop_first=classifier.spec.kind != "convnet"
            )
            direct = classifier.probe._model.decision_function(pooled)[0]
            score_direct = float(
                direct[classes.index(predicted)] if np.ndim(direct) else direct
            )

            shown = frame
            n_people: int | None = None
            on_people = area = None
            boxes = X.detect_people(frame)
            n_people = len(boxes)
            if boxes:
                on_people, area = X.evidence_on_people(evidence, boxes, frame.shape[:2])
            if redact:
                shown = X.pixelate_boxes(frame.copy(), boxes)

            yield Step(
                index=index, t_s=t, predicted=str(predicted), confidence=confidence,
                elapsed_ms=(time.perf_counter() - began) * 1000,
                frame_bgr=shown, evidence=evidence, grid=grid,
                score_from_map=score_from_map, score_direct=score_direct,
                n_people=n_people, evidence_on_people=on_people, people_area=area,
            )
            index += 1
            t += interval_s
            if index >= max_samples:
                break
    finally:
        capture.release()
