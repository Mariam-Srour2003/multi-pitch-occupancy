"""Two independent descriptors must agree that the `(1).mp4` suffix swaps cameras.

`vision/camera_id.py` and `vision/fingerprint.py` were written separately and answer the
camera-identity question by different means:

* `camera_id` takes the median background, equalises it, and keeps the **gradient
  magnitude** downscaled to a 48x27 grid - a picture of where the edges are.
* `fingerprint` takes the same median background but builds a grid of **gradient
  orientation** histograms plus **rg-chromaticity** histograms, and masks the watermark.

They share only the median-background step. Everything after it differs: magnitude versus
orientation, no colour versus colour, unmasked versus masked. So when both conclude that
the morning slot's `file0` is the same physical camera as the evening slot's `file1`, that
is two measurements agreeing rather than one repeated - which is the reason the duplicated
work was kept instead of deleted.

The claim under test is the one `db/seed.py` depends on: its PHYSICAL_CAMERA mapping sends
`slot_20260711_1000_camA` and `slot_20260712_2030_camB` to the same camera. If a future
change to either descriptor breaks that agreement, the mapping is no longer supported by
the evidence and this test fails rather than letting fusion quietly swap pitch halves.

Skipped when the venue_01 recordings are absent - they are not distributed with the repo.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pitch_occupancy.config import settings
from pitch_occupancy.frame_source import discover_slots
from pitch_occupancy.vision import camera_id, fingerprint

#: What `db/seed.py` asserts: these two exports are the same physical camera, despite
#: carrying opposite suffixes.
EXPECTED_SAME_CAMERA = (
    ("slot_20260711_1000", "file0"),
    ("slot_20260712_2030", "file1"),
)

N_FRAMES = 7


def _slot_videos():
    venue = settings.raw_dir / "venue_01"
    if not venue.is_dir():
        pytest.skip("venue_01 recordings not present (footage is not in the repo)")
    slots = discover_slots(venue)
    needed = {slot for slot, _ in EXPECTED_SAME_CAMERA}
    if not needed.issubset(slots):
        pytest.skip("venue_01 recordings not present (footage is not in the repo)")
    return slots


def _sample_frames(path, n=N_FRAMES):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        pytest.skip(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        pytest.skip(f"no readable frames in {path}")
    return frames


def _four_views(slots):
    """The four venue_01 recordings as (tag, path), tag being slot + file key."""
    views = []
    for slot, cams in sorted(slots.items()):
        if slot not in {s for s, _ in EXPECTED_SAME_CAMERA}:
            continue
        for key, path in sorted(cams.items()):
            views.append((f"{slot}::{key}", path))
    return views


@pytest.fixture(scope="module")
def views():
    return _four_views(_slot_videos())


@pytest.fixture(scope="module")
def frames(views):
    """Sampled frames per recording, decoded once for the whole module.

    Without this the six tests below re-seek four ~600 MB files about twenty times
    between them, which took a minute. A test that slow gets skipped rather than run.
    """
    return {tag: _sample_frames(path) for tag, path in views}


@pytest.fixture(scope="module")
def magnitude_descriptors(views):
    """`camera_id` descriptors, one decode per recording."""
    return {tag: camera_id.describe_view(path, n_frames=N_FRAMES) for tag, path in views}


@pytest.fixture(scope="module")
def orientation_prints(frames):
    """`fingerprint` descriptors, reusing the decoded frames."""
    return [
        fingerprint.fingerprint_view(tag, fs, venue="venue_01")
        for tag, fs in frames.items()
    ]


def _expected_tags():
    return {f"{slot}::{key}" for slot, key in EXPECTED_SAME_CAMERA}


def test_the_four_recordings_are_discoverable(views):
    assert len(views) == 4, f"expected 4 venue_01 recordings, found {[t for t, _ in views]}"
    assert _expected_tags().issubset({t for t, _ in views})


def test_camera_id_pairs_the_opposite_suffixes(magnitude_descriptors):
    """The gradient-magnitude descriptor: cross-day, opposite-suffix is the best match."""
    descs = magnitude_descriptors
    want = _expected_tags()
    a, b = sorted(want)

    cross = camera_id.similarity(descs[a], descs[b])
    others = [
        camera_id.similarity(descs[a], descs[t]) for t in descs if t not in (a, b)
    ]
    assert cross > max(others), (
        f"camera_id: {a} should match {b} ({cross:.3f}) above all others ({others})"
    )


def test_fingerprint_pairs_the_opposite_suffixes(orientation_prints):
    """The orientation+chroma descriptor must reach the same pairing, independently."""
    pairs = fingerprint.mutual_nearest_pairs(orientation_prints)
    assert pairs, "fingerprint found no mutual nearest pair among the four recordings"

    closest = {pairs[0][0], pairs[0][1]}
    assert closest == _expected_tags(), (
        f"fingerprint's closest mutual pair is {sorted(closest)}, "
        f"expected {sorted(_expected_tags())}"
    )


def test_both_descriptors_agree_and_seed_encodes_that_agreement(
    magnitude_descriptors, orientation_prints
):
    """The two methods agree, and `db/seed.py` maps the agreed pair to one camera."""
    from pitch_occupancy.db.seed import PHYSICAL_CAMERA

    # descriptor 1: greedy match of the evening recordings against the morning as reference
    descs = magnitude_descriptors
    morning = {t: d for t, d in descs.items() if t.startswith("slot_20260711_1000")}
    evening = {t: d for t, d in descs.items() if t.startswith("slot_20260712_2030")}
    matched = camera_id.match_cameras(evening, morning)

    a, b = sorted(_expected_tags())  # a = morning file0, b = evening file1
    assert matched[b] == a, f"camera_id greedy match sent {b} to {matched[b]}, expected {a}"

    # descriptor 2: the same pairing, from orientation + chroma
    top = fingerprint.mutual_nearest_pairs(orientation_prints)[0]
    assert {top[0], top[1]} == {a, b}

    # and the mapping the rest of the system relies on says the same thing
    same_camera = {
        tag for tag, cam in PHYSICAL_CAMERA.items()
        if cam == PHYSICAL_CAMERA["slot_20260711_1000_camA"]
    }
    assert same_camera == {"slot_20260711_1000_camA", "slot_20260712_2030_camB"}, (
        "db/seed.py PHYSICAL_CAMERA no longer encodes the measured swap"
    )


def test_same_day_halves_stay_far_apart(orientation_prints):
    """Sanity: the two cameras of one slot look at opposite ends and must not pair."""
    by_tag = {p.camera: p for p in orientation_prints}
    for slot in {s for s, _ in EXPECTED_SAME_CAMERA}:
        halves = sorted(t for t in by_tag if t.startswith(slot))
        if len(halves) != 2:
            continue
        d = fingerprint.distance(by_tag[halves[0]].descriptor, by_tag[halves[1]].descriptor)
        cross_day = fingerprint.distance(
            by_tag[sorted(_expected_tags())[0]].descriptor,
            by_tag[sorted(_expected_tags())[1]].descriptor,
        )
        assert d > cross_day, (
            f"{slot}'s two halves ({d:.3f}) are closer than the cross-day same-camera "
            f"pair ({cross_day:.3f}) - the descriptor is not separating views"
        )


def test_the_two_descriptors_are_actually_different_measurements(frames):
    """Guard the premise: if the descriptors converge, the agreement stops being evidence."""
    background = fingerprint.median_background(next(iter(frames.values())))

    magnitude = camera_id._structure(background, (48, 27))
    orientation = fingerprint.structure_descriptor(background)

    assert magnitude.shape != orientation.shape, (
        "camera_id and fingerprint now produce the same descriptor shape; "
        "check they have not been merged, or this cross-check is circular"
    )
    assert np.isfinite(magnitude).all() and np.isfinite(orientation).all()
