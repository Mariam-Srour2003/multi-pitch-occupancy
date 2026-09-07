"""The preprocessing search's false-play control.

Every cross-venue test fold is 100% ACTIVE_PLAY, so recall can be bought outright by making
the probe readier to say PLAY. The control is the only thing standing between that and a
recommendation.

It was broken for 52 evaluations: it scored each probe on frames it had just been fitted on,
read 0.0000 every time, and ranked nothing. These tests pin the two properties that failure
violated - the control must use held-out frames, and it must be able to tell configurations
apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.preprocess_search import false_play_rate
from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.db.seed import PHYSICAL_CAMERA

EMPTY, PLAY = "C1_EMPTY", "C2_ACTIVE_PLAY"
CACHE = settings.feature_cache_dir / "ablate_dinov2_full.npz"


@pytest.fixture(scope="module")
def rows():
    manifest = settings.dataset_dir / "manifest.csv"
    if not manifest.exists():
        pytest.skip("dataset manifest not present")
    return development_rows(read_manifest(manifest))


def camera(row) -> str:
    return PHYSICAL_CAMERA.get(row.camera, row.camera)


def test_the_frames_it_scores_on_are_not_the_frames_it_trains_on(rows) -> None:
    """The bug, stated as a property. Camera A trains, camera B is scored, and the two share
    no frames - so a probe cannot score well here by memorising."""
    train = {r.file for r in rows if r.venue == "venue_01" and camera(r) == "camera_A"}
    scored = {
        r.file for r in rows
        if r.venue == "venue_01" and camera(r) == "camera_B" and r.class3 == EMPTY
    }
    assert scored, "no held-out EMPTY frames; the control would have nothing to measure"
    assert not (train & scored)


def test_the_clip_folds_have_no_held_out_empty_frames(rows) -> None:
    """Why the bug survived: there was nothing on the held-out side of a cross-venue fold to
    measure, so the code reached for the training side. The gap is real and still here."""
    from pitch_occupancy.data.splits import leave_one_group_out

    for fold in leave_one_group_out(rows):
        empty_in_test = sum(1 for r in fold.test if r.class3 == EMPTY)
        if fold.name.endswith("venue_01"):
            assert empty_in_test > 0
        else:
            assert empty_in_test == 0, "a clip venue gained EMPTY frames - revisit the control"


@pytest.mark.slow
def test_the_control_discriminates_between_configurations(rows) -> None:
    """A control that returns the same number for everything is not a control. On the real
    cached features the two extremes are far apart."""
    if not CACHE.exists():
        pytest.skip("cached features not present")
    data = np.load(CACHE, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(data["files"])}
    kept = [r for r in rows if r.file in index]
    X = data["features"][[index[r.file] for r in kept]]
    pos = {r.file: i for i, r in enumerate(kept)}

    rate = false_play_rate(kept, X, pos)
    assert 0.0 <= rate <= 1.0
    assert not np.isnan(rate)
    # the measured value for untouched DINOv2 features; a control pinned at 0 or 1 is dead
    assert 0.01 < rate < 0.99


@pytest.mark.slow
def test_adding_the_all_play_clip_venues_destroys_empty_detection(rows) -> None:
    """The larger finding, kept as a test so it cannot quietly stop being true.

    All 282 clip-venue development frames are ACTIVE_PLAY and none are EMPTY. Training on
    them takes the probe from calling 23% of unseen empty pitches a match to calling all of
    them."""
    if not CACHE.exists():
        pytest.skip("cached features not present")
    from pitch_occupancy.vision.heads import LinearProbe

    data = np.load(CACHE, allow_pickle=True)
    index = {str(f): i for i, f in enumerate(data["files"])}
    kept = [r for r in rows if r.file in index]
    X = data["features"][[index[r.file] for r in kept]]
    pos = {r.file: i for i, r in enumerate(kept)}

    clips = [r for r in kept if r.venue != "venue_01"]
    assert clips and all(r.class3 == PLAY for r in clips), "clip venues are single-class"

    venue_a = [r for r in kept if r.venue == "venue_01" and camera(r) == "camera_A"]
    empties = [
        r for r in kept
        if r.venue == "venue_01" and camera(r) == "camera_B" and r.class3 == EMPTY
    ]

    def rate(train):
        probe = LinearProbe("t", seed=42).fit(X[[pos[r.file] for r in train]], train)
        pred = probe.predict(X[[pos[r.file] for r in empties]], empties)
        return float(np.mean([p == PLAY for p in pred]))

    assert rate(venue_a) < 0.5
    assert rate(venue_a + clips) > 0.9


@pytest.mark.slow
def test_h3_recall_still_reproduces_the_published_table(rows) -> None:
    """The h3_with_false_play harness earns the right to add a column by first reproducing
    the one already published. If this drifts, neither column can be trusted."""
    import csv as _csv

    from experiments.h3_with_false_play import CACHES, cross_venue_recall, load
    import numpy as _np

    published = {
        r["model"]: float(r["play_recall"])
        for r in _csv.DictReader((settings.results_dir / "h3_cross_venue_recall.csv")
                                 .open(encoding="utf-8"))
        if r["held_out_venue"] == "MEAN_ACROSS_FOLDS"
    }
    if not published:
        pytest.skip("published H3 table not present")

    for model, cache in CACHES.items():
        if not (settings.feature_cache_dir / cache).exists() or model not in published:
            continue
        kept, X = load(cache, rows)
        mean = float(_np.mean([r for _, r, _ in cross_venue_recall(model, kept, X)]))
        assert abs(mean - published[model]) < 0.005, (
            f"{model}: {mean:.4f} against published {published[model]:.4f}"
        )


@pytest.mark.slow
def test_the_effective_sample_is_reported_not_assumed(rows) -> None:
    """The 243 held-out empty frames are consecutive views of one camera. Counting them as
    243 observations gave p-values down to 8e-53; counting distinct scenes leaves three to
    ten, and no comparison survives. The check that catches this must stay."""
    import cv2

    from pitch_occupancy.data.dedup import DEFAULT_THRESHOLD, dhash, hamming
    from pitch_occupancy.db.seed import PHYSICAL_CAMERA

    cam = lambda r: PHYSICAL_CAMERA.get(r.camera, r.camera)  # noqa: E731
    empties = [
        r for r in rows
        if r.venue == "venue_01" and cam(r) == "camera_B" and r.class3 == EMPTY
    ]
    hashes = {}
    for row in empties:
        img = cv2.imread(str(settings.dataset_dir / row.file))
        if img is not None:
            hashes[row.file] = dhash(img)
    if len(hashes) < 50:
        pytest.skip("frames not present")

    keep: list[str] = []
    for f, h in hashes.items():
        if all(hamming(h, hashes[k]) > DEFAULT_THRESHOLD for k in keep):
            keep.append(f)
    assert len(keep) < 25, (
        f"{len(keep)} distinct scenes from {len(hashes)} frames - if this rises a lot, the "
        f"held-out set has genuinely diversified and the significance entry needs revisiting"
    )
