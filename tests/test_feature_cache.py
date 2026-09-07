"""The feature cache is the one place where a silent error is most expensive: every
experiment reads from it, and wrong-but-plausible vectors still train a head that
converges. The guards below are what stop the pilot's pooling bug recurring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch_occupancy.data.feature_cache import (
    CachedFeatures,
    build_cache,
    cache_path,
    features_for,
    load_cache,
    preprocessing_hash,
)
from pitch_occupancy.data.manifest import ManifestRow
from pitch_occupancy.vision.backbones import BACKBONES, POOLING_STAMP


def row(file: str, venue: str = "v1") -> ManifestRow:
    return ManifestRow(
        file=file, class4="2_playing", class3="C2_ACTIVE_PLAY", venue=venue,
        camera="c", slot_date="2026-07-11", slot_time="10:00", slot_id="s1", t_s=0,
        source="regular", labeled_by="human", lighting="day", quality="unknown",
        split_role="",
    )


@pytest.fixture
def cached(tmp_path: Path) -> CachedFeatures:
    files = np.array([f"2_playing/f{i}.jpg" for i in range(5)], dtype=object)
    feats = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    c = CachedFeatures("vit", POOLING_STAMP, "abc123", files, feats)
    np.savez_compressed(
        cache_path("vit", tmp_path), files=c.files, features=c.features,
        backbone=c.backbone, pooling=c.pooling, preproc_hash=c.preproc_hash,
    )
    return c


# --- fingerprinting ---------------------------------------------------------


def test_preprocessing_hash_is_stable_and_order_independent() -> None:
    a = preprocessing_hash(roi=True, resize=224)
    b = preprocessing_hash(resize=224, roi=True)
    assert a == b


def test_preprocessing_hash_changes_with_settings() -> None:
    assert preprocessing_hash(roi=True) != preprocessing_hash(roi=False)


# --- loading guards ---------------------------------------------------------


def test_loads_a_matching_cache(cached, tmp_path: Path) -> None:
    got = load_cache("vit", tmp_path)
    assert got.backbone == "vit"
    assert got.dim == 4
    assert len(got) == 5


def test_missing_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no feature cache"):
        load_cache("vit", tmp_path)


def test_wrong_pooling_is_refused(tmp_path: Path) -> None:
    """The exact failure that produced the pilot's false 38%."""
    np.savez_compressed(
        cache_path("vit", tmp_path),
        files=np.array(["a.jpg"], dtype=object), features=np.zeros((1, 4), np.float32),
        backbone="vit", pooling="pooler_output", preproc_hash="abc123",
    )
    with pytest.raises(ValueError, match="pooling"):
        load_cache("vit", tmp_path)


def test_mismatched_preprocessing_is_refused(cached, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="preprocessing"):
        load_cache("vit", tmp_path, expect_preproc_hash="different")


def test_matching_preprocessing_is_accepted(cached, tmp_path: Path) -> None:
    assert load_cache("vit", tmp_path, expect_preproc_hash="abc123").dim == 4


# --- alignment --------------------------------------------------------------


def test_features_are_returned_in_row_order(cached) -> None:
    rows = [row("2_playing/f3.jpg"), row("2_playing/f0.jpg")]
    X, kept = features_for(cached, rows)
    assert [r.file for r in kept] == ["2_playing/f3.jpg", "2_playing/f0.jpg"]
    assert np.array_equal(X[0], cached.features[3])
    assert np.array_equal(X[1], cached.features[0])


def test_uncached_frames_are_dropped_not_zero_filled(cached) -> None:
    """A head trained on zero vectors still converges - to nonsense."""
    rows = [row("2_playing/f1.jpg"), row("2_playing/NEW.jpg")]
    X, kept = features_for(cached, rows)
    assert len(kept) == 1
    assert X.shape == (1, 4)


def test_no_overlap_returns_empty_not_an_error(cached) -> None:
    X, kept = features_for(cached, [row("2_playing/unknown.jpg")])
    assert kept == []
    assert X.shape == (0, 4)


# --- the real thing ---------------------------------------------------------


@pytest.mark.slow
def test_build_cache_round_trips_on_real_images(tmp_path: Path) -> None:
    import cv2

    ds = tmp_path / "ds"
    (ds / "2_playing").mkdir(parents=True)
    rows = []
    for i in range(3):
        name = f"2_playing/f{i}.jpg"
        cv2.imwrite(str(ds / name), np.full((64, 64, 3), 40 * i + 20, np.uint8))
        rows.append(row(name))

    built = build_cache(rows, "convnextv2", ds, tmp_path / "cache", batch_size=2, progress=False)
    assert built.pooling == POOLING_STAMP
    assert len(built) == 3
    reloaded = load_cache("convnextv2", tmp_path / "cache")
    assert np.allclose(reloaded.features, built.features)


def test_every_registered_backbone_declares_its_pooling_kind() -> None:
    assert {b.kind for b in BACKBONES.values()} <= {"transformer", "convnet"}


# --- the processor-geometry convention ------------------------------------
#
# WP3-T3 found that ConvNeXtV2 and DINOv2 resize to 256 and centre-crop back to 224 *after*
# preprocess.py has produced a 224x224 frame, discarding 23.4% of it. Which convention to
# use is an open decision to be settled by measurement, so both have to be cacheable at
# once. `thesis/protocol.md` and `EXPERIMENT_LOG.md` both stated the flag was already part
# of the cache fingerprint; it was neither in the fingerprint nor a parameter of
# build_cache, so the alternative convention could not be cached at all.


def test_the_two_geometry_conventions_get_different_fingerprints() -> None:
    """Otherwise they collide on one cache key and mix silently."""
    hf = BACKBONES["vit"].hf_id
    on = preprocessing_hash(backbone=hf, processor_geometry=True, roi=False)
    off = preprocessing_hash(backbone=hf, processor_geometry=False, roi=False)
    assert on != off


def test_the_two_geometry_conventions_get_different_files() -> None:
    """And different filenames, or building the second overwrites the first."""
    d = Path("cache")
    assert cache_path("vit", d) != cache_path("vit", d, processor_geometry=False)
    assert cache_path("vit", d).name == "vit.npz", "the default path must not move"
    assert cache_path("vit", d, processor_geometry=False).name == "vit_nogeom.npz"


def test_the_default_convention_keeps_the_original_filename() -> None:
    """Existing caches stay findable; only the alternative convention is new."""
    assert cache_path("dinov2", Path("c"), processor_geometry=True).name == "dinov2.npz"


def test_load_cache_reads_the_convention_it_is_asked_for(tmp_path: Path) -> None:
    """A cache written under one convention is never returned for the other."""
    for geom, val in ((True, 1.0), (False, 2.0)):
        np.savez_compressed(
            cache_path("vit", tmp_path, processor_geometry=geom),
            files=np.array(["a.jpg"], dtype=object),
            features=np.full((1, 4), val, dtype=np.float32),
            backbone="vit",
            pooling=POOLING_STAMP,
            preproc_hash=f"h_{geom}",
        )
    assert load_cache("vit", tmp_path).features[0][0] == 1.0
    assert load_cache("vit", tmp_path, processor_geometry=False).features[0][0] == 2.0


def test_asking_for_a_convention_that_was_never_built_raises(tmp_path: Path) -> None:
    np.savez_compressed(
        cache_path("vit", tmp_path),
        files=np.array(["a.jpg"], dtype=object),
        features=np.zeros((1, 4), dtype=np.float32),
        backbone="vit", pooling=POOLING_STAMP, preproc_hash="h",
    )
    load_cache("vit", tmp_path)  # the default one exists
    with pytest.raises(FileNotFoundError):
        load_cache("vit", tmp_path, processor_geometry=False)
