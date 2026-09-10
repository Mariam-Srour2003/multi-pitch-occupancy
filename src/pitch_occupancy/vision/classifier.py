"""The production classifier: one frame in, one class and a confidence out (WP6-T2).

`worker.run_slot` has taken a ``classify`` callable since it was written, and
`scheduler.run_due` passes one straight through. Until this module existed **nothing ever
supplied a real one**. Every test hands the seam a stub, `experiments/end_to_end_slots.py`
rebuilds each slot's per-minute sequence from the *labels* rather than from a model, and
`worker.main` raised `NotImplementedError` naming exactly this gap. So the sampling, fusion,
aggregation, evidence selection and reconciliation had all been exercised end to end without
a single frame ever having been classified by the system itself.

That is a quiet kind of hole. Nothing was broken and no test failed; the pipeline simply had
no head on it, and the missing piece was small enough to keep not noticing.

**The input path is the whole risk here.** A classifier that embeds frames even slightly
differently from the way the feature cache was built is a model evaluated on one
distribution and deployed on another - and this project has already been bitten once by
exactly that (`docs/PM_REVIEW_2026-09-08.md`: every published number comes from raw frames
handed to the HF processor, which keeps roughly the middle half of a 16:9 pitch, and
letterboxing instead moved ConvNeXtV2's false-play rate from 99.2% to 2.1%). So this module
does not implement its own preprocessing. It converts BGR to RGB, wraps the array in a PIL
image and calls the same :func:`embed_batch` under the same processor geometry that
`data/feature_cache.py` uses, and a test embeds a real frame both ways and asserts the two
vectors agree.

**The probe is fitted at construction, not loaded from a pickle.** It takes about a second
on cached features. A serialised sklearn pipeline would be a second artefact that can drift
from the manifest it claims to come from, that breaks on a scikit-learn upgrade, and that
nothing in the repository regenerates - three failure modes bought in exchange for saving a
second. Refitting means the deployed model is always the one the current cache and manifest
imply, and `pitch info` can say which rows it came from.

**It trains on development rows only.** :func:`development_rows` drops the locked venues, so
the final test set stays unevaluated (WP0-T4) even though a deployment would have every
reason to train on everything it has. The lock is worth more than the frames: reaching those
rows requires ``final_test_rows(..., i_have_finished_all_development=True)``, and a
production path that quietly trained on them would make the one honest number in the thesis
unquotable.

**The confidence is not calibrated.** It is the probe's maximum class probability, and RQ6
found these are badly behaved for some backbones - one model's calibrated confidences are
890/907 identical, so a risk-coverage "curve" over them is a band. `fuse` uses it to weigh
two cameras against each other and `aggregate_slot` never reads it, so nothing here turns a
confidence into a verdict on its own. Anything that wants a *threshold* on it should go
through `evaluation/calibration.py` first and say which temperature it fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pitch_occupancy.data.taxonomy import Class3

__all__ = ["ProbeClassifier", "load_classifier"]

#: The probe's seed. Fixed so that two deployments built from the same cache agree, and the
#: same value the benchmark uses so a deployed prediction is reproducible from the tables.
SEED = 42


@dataclass(slots=True)
class ProbeClassifier:
    """A frozen backbone and a fitted linear head, behind `worker.Classifier`.

    Callable with one BGR frame, as OpenCV and `frame_source` produce them, because that is
    what `run_slot` has to hand. Batch use goes through :meth:`classify_batch`, which is the
    same path with one forward pass - at one frame per camera per minute the difference does
    not matter operationally, but it matters a great deal when replaying a recorded slot.
    """

    backbone: str
    model: object
    processor: object
    spec: object
    probe: object
    n_train: int

    def __call__(self, image_bgr: np.ndarray) -> tuple[Class3, float]:
        (state, confidence), = self.classify_batch([image_bgr])
        return state, confidence

    def classify_batch(self, images_bgr: list[np.ndarray]) -> list[tuple[Class3, float]]:
        """Classify several frames in one forward pass.

        Empty input returns an empty list rather than calling the model with nothing: a
        zero-length batch is a legitimate minute in which every camera was down, and the
        processor raises on it.
        """
        if not images_bgr:
            return []
        from PIL import Image

        from pitch_occupancy.vision.backbones import embed_batch

        pil = [Image.fromarray(np.asarray(im)[:, :, ::-1]) for im in images_bgr]
        features = embed_batch(self.model, self.processor, self.spec, pil)
        proba = self.probe.predict_proba(features)
        classes = list(self.probe.classes_)
        return [
            (Class3(classes[int(row.argmax())]), float(row.max()))
            for row in proba
        ]


def load_classifier(
    model_key: str | None = None,
    *,
    cache_dir: Path | None = None,
    dataset_dir: Path | None = None,
) -> ProbeClassifier:
    """Build the deployed classifier: cached features in, fitted probe out.

    Raises rather than degrading if the cache is missing or does not cover the manifest.
    A production classifier that silently trained on whichever rows happened to be cached
    would be a different model from the one the tables describe, and the failure would show
    up as slightly wrong verdicts rather than as an error - the most expensive shape a bug
    can have here.
    """
    from pitch_occupancy.config import settings
    from pitch_occupancy.data.feature_cache import features_for, load_cache
    from pitch_occupancy.data.manifest import read_manifest
    from pitch_occupancy.data.splits import development_rows, load_final_venues
    from pitch_occupancy.vision.backbones import load_backbone
    from pitch_occupancy.vision.heads import LinearProbe

    key = model_key or settings.default_model_key
    cache_dir = cache_dir or settings.feature_cache_dir
    dataset_dir = dataset_dir or settings.dataset_dir

    cached = load_cache(key, cache_dir)
    rows = development_rows(read_manifest(dataset_dir / "manifest.csv"))
    features, kept = features_for(cached, rows)
    if not kept:
        raise RuntimeError(
            f"the {key} cache covers none of the {len(rows)} development rows; "
            f"run `uv run pitch cache --model {key}` before deploying"
        )
    if len(kept) != len(rows):
        raise RuntimeError(
            f"the {key} cache covers {len(kept)} of {len(rows)} development rows. "
            f"Fit on a subset and the deployed model is not the one any table describes; "
            f"run `uv run pitch cache --model {key}` to bring it up to date"
        )

    locked = load_final_venues()
    leaked = sorted({r.venue for r in kept} & set(locked))
    if leaked:  # belt and braces: development_rows already drops these
        raise RuntimeError(f"locked venues reached the production fit: {leaked}")

    model, processor, spec = load_backbone(key)
    probe = LinearProbe(key, seed=SEED).fit(features, kept)
    return ProbeClassifier(
        backbone=key, model=model, processor=processor, spec=spec,
        probe=probe, n_train=len(kept),
    )
