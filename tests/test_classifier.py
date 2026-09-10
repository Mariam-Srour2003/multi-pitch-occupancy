"""The production classifier (WP6-T2).

`run_slot` and `run_due` have taken a ``classify`` callable since they were written, and
until `vision/classifier.py` nothing outside a test ever supplied one: every scheduler test
hands the seam a stub, `end_to_end_slots.py` rebuilds each slot from the *labels*, and
`worker.main` raised rather than running. The pipeline had no head on it and nothing failed,
which is what makes it worth a file of tests rather than a line in a changelog.

Two of these matter more than the rest:

* **the input path** - the deployed classifier must embed a frame the way the cache was
  built, or the model is evaluated on one distribution and deployed on another. This project
  has already been bitten by exactly that (`docs/PM_REVIEW_2026-09-08.md`), so the check is
  against real cached features rather than against a description of them;
* **the lock** - a production fit that reached the held-out venues would make the one
  honest number in the thesis unquotable, and it would do so silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch_occupancy.data.taxonomy import Class3
from pitch_occupancy.vision.classifier import ProbeClassifier, load_classifier

ROOT = Path(__file__).resolve().parents[1]


class FakeProbe:
    """Two classes, so an argmax has something to choose between."""

    classes_ = np.array(["C1_EMPTY", "C2_ACTIVE_PLAY"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # first feature decides, so a test can steer the answer with a pixel value
        empty = (X[:, 0] > 0).astype(float)
        return np.stack([empty * 0.8 + 0.1, (1 - empty) * 0.8 + 0.1], axis=1)


def fake_classifier(monkeypatch) -> ProbeClassifier:
    """A classifier with the model replaced, so the seam can be tested without a backbone."""
    import pitch_occupancy.vision.backbones as backbones

    def embed_batch(model, processor, spec, images, **kwargs):
        # one feature per image: positive when the image is bright
        return np.array([[float(np.asarray(im).mean()) - 100] for im in images], np.float32)

    monkeypatch.setattr(backbones, "embed_batch", embed_batch)
    return ProbeClassifier(backbone="dinov2", model=None, processor=None, spec=None,
                           probe=FakeProbe(), n_train=7)


# --- the seam ---------------------------------------------------------------------------


def test_it_satisfies_the_classifier_protocol(monkeypatch) -> None:
    """`run_slot` types its parameter as `worker.Classifier`; the check is that a real
    instance actually matches it, not that the annotation reads well."""
    import inspect

    from pitch_occupancy.worker import Classifier

    clf = fake_classifier(monkeypatch)
    assert isinstance(clf, Classifier)
    # isinstance only sees that a __call__ exists, so pin the shape of it too
    declared = inspect.signature(Classifier.__call__)
    actual = inspect.signature(type(clf).__call__)
    assert list(actual.parameters)[1:] == list(declared.parameters)[1:]
    assert actual.return_annotation == declared.return_annotation


def test_one_frame_in_one_class_and_confidence_out(monkeypatch) -> None:
    clf = fake_classifier(monkeypatch)
    state, confidence = clf(np.full((8, 8, 3), 200, np.uint8))
    assert isinstance(state, Class3)
    assert 0.0 <= confidence <= 1.0


def test_the_frame_is_read_as_BGR(monkeypatch) -> None:
    """`frame_source` and OpenCV both produce BGR, and the cache was built from RGB. A
    classifier that skipped the swap would score a blue pitch on a model trained on a green
    one - and would never raise."""
    seen: list[np.ndarray] = []
    import pitch_occupancy.vision.backbones as backbones

    def embed_batch(model, processor, spec, images, **kwargs):
        seen.extend(np.asarray(im) for im in images)
        return np.zeros((len(images), 1), np.float32)

    monkeypatch.setattr(backbones, "embed_batch", embed_batch)
    clf = ProbeClassifier(backbone="dinov2", model=None, processor=None, spec=None,
                          probe=FakeProbe(), n_train=1)
    frame = np.zeros((4, 4, 3), np.uint8)
    frame[:, :, 0] = 255  # blue in BGR
    clf(frame)
    assert (seen[0][:, :, 2] == 255).all(), "the channels were not swapped to RGB"


def test_an_empty_batch_is_not_sent_to_the_model(monkeypatch) -> None:
    """A minute in which every camera was down is a legitimate input. The processor raises
    on a zero-length batch, and a crash there would take down a slot over an absence the
    rest of the pipeline handles by design."""
    import pitch_occupancy.vision.backbones as backbones

    def explode(*a, **k):
        raise AssertionError("the model was called with nothing to embed")

    monkeypatch.setattr(backbones, "embed_batch", explode)
    clf = ProbeClassifier(backbone="dinov2", model=None, processor=None, spec=None,
                          probe=FakeProbe(), n_train=1)
    assert clf.classify_batch([]) == []


def test_a_batch_is_one_forward_pass(monkeypatch) -> None:
    """Replaying a recorded slot is 120 frames; one call per frame is 120 model invocations
    where one is needed."""
    calls: list[int] = []
    import pitch_occupancy.vision.backbones as backbones

    def embed_batch(model, processor, spec, images, **kwargs):
        calls.append(len(images))
        return np.zeros((len(images), 1), np.float32)

    monkeypatch.setattr(backbones, "embed_batch", embed_batch)
    clf = ProbeClassifier(backbone="dinov2", model=None, processor=None, spec=None,
                          probe=FakeProbe(), n_train=1)
    out = clf.classify_batch([np.zeros((4, 4, 3), np.uint8)] * 5)
    assert len(out) == 5
    assert calls == [5]


def test_the_confidence_is_the_winning_class_probability(monkeypatch) -> None:
    """Not the margin, not a logit. `fuse` weighs two cameras with it, so the scale has to
    be the one `fusion.py` documents."""
    clf = fake_classifier(monkeypatch)
    _, confidence = clf(np.full((8, 8, 3), 200, np.uint8))
    assert confidence == pytest.approx(0.9)


# --- the lock ---------------------------------------------------------------------------


def test_the_production_fit_refuses_a_locked_venue(monkeypatch, tmp_path) -> None:
    """The guard behind the guard. `development_rows` already drops these, so this fires
    only if that changes - which is precisely when it is needed."""
    import pitch_occupancy.vision.classifier as mod

    class Row:
        file, class3, venue = "a.jpg", "C1_EMPTY", "locked_venue"

    monkeypatch.setattr(mod, "__doc__", mod.__doc__)  # keep the module import explicit
    from pitch_occupancy.data import feature_cache, manifest, splits

    monkeypatch.setattr(manifest, "read_manifest", lambda p: [Row()])
    monkeypatch.setattr(splits, "development_rows", lambda rows, **k: list(rows))
    monkeypatch.setattr(splits, "load_final_venues", lambda *a, **k: frozenset({"locked_venue"}))
    monkeypatch.setattr(feature_cache, "load_cache", lambda *a, **k: object())
    monkeypatch.setattr(feature_cache, "features_for",
                        lambda cached, rows: (np.zeros((1, 3), np.float32), list(rows)))
    with pytest.raises(RuntimeError, match="locked venues"):
        load_classifier("dinov2", cache_dir=tmp_path, dataset_dir=tmp_path)


def test_a_partial_cache_raises_rather_than_fitting_on_what_it_has(monkeypatch, tmp_path):
    """Fitting on whichever rows happen to be cached gives a model no table describes, and
    the symptom is slightly wrong verdicts rather than an error."""
    from pitch_occupancy.data import feature_cache, manifest, splits

    class Row:
        def __init__(self, name):
            self.file, self.class3, self.venue = name, "C1_EMPTY", "venue_01"

    rows = [Row("a.jpg"), Row("b.jpg")]
    monkeypatch.setattr(manifest, "read_manifest", lambda p: rows)
    monkeypatch.setattr(splits, "development_rows", lambda r, **k: list(r))
    monkeypatch.setattr(splits, "load_final_venues", lambda *a, **k: frozenset())
    monkeypatch.setattr(feature_cache, "load_cache", lambda *a, **k: object())
    monkeypatch.setattr(feature_cache, "features_for",
                        lambda cached, r: (np.zeros((1, 3), np.float32), r[:1]))
    with pytest.raises(RuntimeError, match="1 of 2"):
        load_classifier("dinov2", cache_dir=tmp_path, dataset_dir=tmp_path)


# --- against the real cache -------------------------------------------------------------


@pytest.mark.slow
def test_the_deployed_input_path_reproduces_the_cached_features() -> None:
    """The one that would have caught the input-path problem.

    The cache opens each file with PIL; a deployment receives a BGR array that OpenCV
    decoded from the same file. Those are two decoders and two colour orders, and if they
    disagree then every published number describes a model that is not the one running. They
    agree exactly here, which is worth pinning: it is the assumption `classifier.py` is
    built on, and nothing else in the repository would notice it breaking.
    """
    cv2 = pytest.importorskip("cv2")
    from PIL import Image

    from pitch_occupancy.config import settings
    from pitch_occupancy.data.feature_cache import load_cache
    from pitch_occupancy.data.manifest import read_manifest
    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    cache_file = settings.feature_cache_dir / "dinov2.npz"
    if not cache_file.exists():
        pytest.skip("no dinov2 cache")
    cached = load_cache("dinov2", settings.feature_cache_dir)
    index = cached.index()
    rows = [r for r in read_manifest(settings.dataset_dir / "manifest.csv") if r.file in index]
    if len(rows) < 8:
        pytest.skip("cache too small to compare")
    sample = [rows[i] for i in np.random.default_rng(0).choice(len(rows), 8, replace=False)]

    model, processor, spec = load_backbone("dinov2")
    frames = [cv2.imread(str(settings.dataset_dir / r.file)) for r in sample]
    deployed = embed_batch(model, processor, spec,
                           [Image.fromarray(f[:, :, ::-1]) for f in frames])
    stored = cached.features[[index[r.file] for r in sample]]
    assert np.abs(deployed - stored).max() < 1e-4


@pytest.mark.slow
def test_it_classifies_a_real_frame_as_its_label() -> None:
    """Not an accuracy measurement - the benchmark does that, honestly, on held-out data.
    This asks only whether the assembled thing works at all on a frame it was trained on,
    which is the failure mode of a pipeline that has never been run."""
    cv2 = pytest.importorskip("cv2")
    from pitch_occupancy.config import settings

    if not (settings.feature_cache_dir / "dinov2.npz").exists():
        pytest.skip("no dinov2 cache")
    from pitch_occupancy.data.manifest import read_manifest

    rows = read_manifest(settings.dataset_dir / "manifest.csv")
    frame_row = next(r for r in rows if r.class3 == "C1_EMPTY")
    clf = load_classifier("dinov2")
    state, confidence = clf(cv2.imread(str(settings.dataset_dir / frame_row.file)))
    assert state is Class3.EMPTY
    assert confidence > 0.5
