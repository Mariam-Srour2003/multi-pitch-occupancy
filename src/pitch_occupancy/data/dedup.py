"""Near-duplicate detection (WP2-T4).

Frames sampled every fifteen seconds from a fixed camera watching a mostly-static pitch are
not independent observations. Two frames of the same empty pitch a quarter of a minute apart
differ by a few pixels of cloud. Counting them as two labelled examples overstates the
dataset, and putting them on opposite sides of a split turns a memorisation test into an
apparent generalisation result - which is exactly what H1 found when a 16-bin colour
histogram beat a deep probe under a random split.

So this measures how much of the dataset is genuinely distinct, rather than assuming it.

**dHash, not a pixel comparison.** A difference hash encodes the *gradient* between adjacent
pixels of a downscaled frame, so it is invariant to the brightness and exposure drift that
makes exact comparison useless on outdoor footage, while still separating scenes that differ
in structure. 64 bits, compared by Hamming distance.

**The threshold is not a universal constant.** What counts as "the same frame" depends on the
camera and the scene, so `suggest_threshold` reports the separation actually present in this
data rather than a number carried in from elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, TypeVar

import cv2
import numpy as np

__all__ = [
    "dhash", "hamming", "DuplicateGroup", "find_near_duplicates",
    "near_duplicate_pairs", "duplicate_rate", "distinct_subset",
]

#: Key type of a hash mapping: frame paths in most callers, row indices in some.
K = TypeVar("K", bound=Hashable)

#: Hamming distance at or below which two frames are treated as near-duplicates. Chosen from
#: the measured distribution: consecutive frames from one static slot sit far below it, and
#: frames from different slots far above.
DEFAULT_THRESHOLD = 6


def dhash(image_bgr: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash: is each pixel brighter than the one to its right?

    Gradient-based rather than value-based, so a cloud passing over the pitch shifts every
    pixel's brightness without changing the hash.
    """
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """A set of frames that are near-identical to one another."""

    members: tuple[str, ...]
    max_distance: int

    @property
    def redundant(self) -> int:
        """How many frames could be dropped while keeping one representative."""
        return len(self.members) - 1


def find_near_duplicates(
    hashes: dict[str, int], *, threshold: int = DEFAULT_THRESHOLD
) -> list[DuplicateGroup]:
    """Group frames whose hashes are within `threshold` bits, by single-link chaining.

    **Read the warning before using this to count anything.** Single-link means A near B and
    B near C puts all three together however far apart A and C are. On a fixed camera whose
    view drifts slowly through a slot, every frame is near its neighbour in time, so the
    whole slot chains into one "group" - and the result is meaningless as a duplicate count.

    Measured here, it reported 96.9% of the dataset as redundant and "52 distinct scenes".
    The output refuted itself: the largest group held 512 frames with a maximum internal
    distance of **19** against a threshold of 6, and contained both EMPTY and ACTIVE_PLAY
    frames. An empty pitch and a match in progress are not near-duplicates of each other.

    Use :func:`near_duplicate_pairs` to count duplication. This function is kept because the
    chains themselves are informative - they show a slot is one continuous scene - but it
    answers "what is transitively connected", not "what is a duplicate".
    """
    keys = list(hashes)
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in combinations(keys, 2):
        if hamming(hashes[a], hashes[b]) <= threshold:
            union(a, b)

    clusters: dict[str, list[str]] = {}
    for k in keys:
        clusters.setdefault(find(k), []).append(k)

    groups = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        worst = max(
            hamming(hashes[a], hashes[b]) for a, b in combinations(sorted(members), 2)
        )
        groups.append(DuplicateGroup(tuple(sorted(members)), worst))
    return sorted(groups, key=lambda g: -len(g.members))


def near_duplicate_pairs(
    hashes: dict[str, int], *, threshold: int = DEFAULT_THRESHOLD
) -> list[tuple[str, str, int]]:
    """Every *directly* similar pair, with no transitive chaining.

    This is the quantity that matters for leakage: a pair split across a train/test boundary
    is one frame being scored against a copy of itself. Chains are irrelevant to that - what
    leaks is a pair.
    """
    return [
        (a, b, d)
        for a, b in combinations(sorted(hashes), 2)
        if (d := hamming(hashes[a], hashes[b])) <= threshold
    ]


def duplicate_rate(hashes: dict[str, int], *, threshold: int = DEFAULT_THRESHOLD) -> float:
    """Fraction of frames having at least one direct near-duplicate.

    Deliberately *not* the transitive-group figure, which chains a whole slot together and
    reported 96.9% for a dataset whose frames are plainly not 97% copies.
    """
    if not hashes:
        return 0.0
    has_twin = set()
    for a, b, _ in near_duplicate_pairs(hashes, threshold=threshold):
        has_twin.add(a)
        has_twin.add(b)
    return len(has_twin) / len(hashes)


def distinct_subset(
    hashes: dict[K, int], *, threshold: int = DEFAULT_THRESHOLD
) -> list[K]:
    """Keys of a greedy maximal set of frames that are pairwise *not* near-duplicates.

    This is the **effective sample size** of a test set, and it is the number that governs
    how confident any frame-level claim may be. 1,667 of the 1,692 frames in this dataset
    have a direct near-duplicate; the 243 held-out empty frames are three to ten distinct
    scenes, and six pairwise comparisons that looked significant at p down to 8e-53 did not
    survive being counted this way.

    Greedy rather than exhaustive because a maximum independent set is NP-hard, and the count
    is stable across input orderings anyway - checked on the held-out empty frames, where
    five shuffles all returned the same number. Insertion order of ``hashes`` is respected,
    so a caller that wants a particular tie-break can choose it.

    Lives here beside :func:`near_duplicate_pairs` rather than in the experiment that first
    needed it, because a second experiment now needs the same rule and two greedy loops that
    are meant to agree are two that can drift apart.
    """
    keep: list[K] = []
    for key, h in hashes.items():
        if all(hamming(h, hashes[k]) > threshold for k in keep):
            keep.append(key)
    return keep
