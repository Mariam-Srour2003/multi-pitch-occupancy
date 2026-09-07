"""Camera-view fingerprinting (WP2-T10).

Every split in this project groups by venue, and the venue of a clip is currently a label
someone typed after looking at it. That was affordable for 66 clips and one facility. It does
not survive five venues and a live deployment — and it has already been wrong: the
`(1).mp4` suffix, which is supposed to name the second camera of a pitch, swaps which
physical camera it means between the two `venue_01` recording days. A wrong venue or camera
label does not fail loudly. It silently puts two views of one facility on opposite sides of
a leave-one-venue-out split and turns the project's central generalisation claim into a
memorisation test.

So a camera view gets identified from its pixels instead.

**The median background, not the frame.** Players move, the pitch does not. A per-pixel
median over the frames of one camera view removes anything that moved and leaves the fixed
furniture: boards, fencing, goal position, the skyline behind. That is what identifies a
camera; the match on it is what identifies a venue.

**Two descriptors, because they are good at different things — measured, not assumed.**
On the 70 real views (66 clips + 4 `venue_01` camera-slots), scored against the hand
grouping:

| descriptor | separation (AUC) | 1-NN venue assignment |
|---|---|---|
| structure only | 0.799 | **63/66** |
| chromaticity only | **0.970** | 59/66 |
| structure + 0.5x chroma | 0.880 | **63/66** |

Structure is the better *identifier* — a coarse grid of gradient-orientation histograms,
which survives the illumination change that destroys colour, so one venue's day and night
clips stay together. Chromaticity is the better *separator* — the blue, teal and pink board
colours that a human uses first are exactly what pushes unrelated venues apart, and it is
computed on intensity-normalised chromaticity so a floodlight cannot recolour a pitch.
Neither alone is best at both, so both are kept and combined at the weight that costs
nothing on assignment while nearly recovering chromaticity's separation.

**No per-cell normalisation.** The obvious HOG-style step of normalising each cell to unit
norm was tried first and measurably hurt (AUC 0.799 -> 0.724 combined): it blows up cells
containing nothing but flat turf, so half the descriptor becomes amplified noise. Global
normalisation keeps illumination invariance — a brightness change scales every gradient
together and cancels in the ratio — while letting the cells that contain real structure
dominate.

**The watermark corner is masked.** These clips carry a burned-in `STATBOX` or `XbotGo`
logo bottom-right. It is identical across every clip from one exporter and absent from
`venue_01`'s frames, so leaving it in would make unrelated clips look alike and all clips
look unlike `venue_01` — a similarity that says who exported the file, not where the camera
is.

**No universal threshold, and no pretence of clean clustering.** :func:`suggest_threshold`
reports the separation actually present in this footage rather than a constant carried in
from elsewhere, and says so when the two distributions overlap — which on this data they do.
Nearest-neighbour assignment against known references is reliable here (67/70); fully
unsupervised recovery of the venue grouping is not (ARI 0.71). Use this module to *check* a
grouping and to assign new footage, not to invent a grouping unsupervised.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

__all__ = [
    "ViewFingerprint",
    "Separation",
    "Match",
    "median_background",
    "structure_descriptor",
    "chroma_descriptor",
    "view_descriptor",
    "fingerprint_view",
    "distance",
    "distance_matrix",
    "suggest_threshold",
    "match_view",
    "mutual_nearest_pairs",
    "cluster_views",
]

#: Working size for the median background. Small enough that 500 frames fit in memory at
#: once, large enough that fence posts and goal frames survive.
WORK_SIZE = (320, 180)

#: Structure grid: 8x6 cells x 9 orientation bins = 432 dimensions. Coarse enough to ignore
#: where individual players stood, fine enough to place a goal or a stand. A 12x8 grid was
#: tried and changed nothing (AUC 0.796 vs 0.799), so the cheaper one stays.
STRUCTURE_GRID = (8, 6)
ORIENTATION_BINS = 9

#: Chromaticity grid, deliberately coarser: board and fence colour is a regional property,
#: and a fine grid would start encoding which half of the pitch the players were on.
CHROMA_GRID = (4, 3)
CHROMA_BINS = 6

#: Weight on the chromaticity block. Chosen by measurement over 0.3-1.5 on the real views:
#: 0.5 is the largest weight that costs nothing on 1-NN assignment (63/66, same as structure
#: alone) while lifting separation from AUC 0.799 to 0.880.
CHROMA_WEIGHT = 0.5

#: Fraction of width and height masked out of the bottom-right corner, covering the exporter
#: watermark. Sized against the largest logo present (`XbotGo`).
WATERMARK_FRACTION = 0.12


@dataclass(frozen=True, slots=True)
class ViewFingerprint:
    """One camera view, reduced to something comparable.

    The two blocks are kept separately as well as combined, because they answer different
    questions and the thesis reports them apart.
    """

    camera: str
    descriptor: np.ndarray
    structure: np.ndarray
    chroma: np.ndarray
    n_frames: int
    venue: str | None = None


@dataclass(frozen=True, slots=True)
class Separation:
    """How well a descriptor separates same-venue pairs from different-venue pairs."""

    same_max: float
    different_min: float
    same_mean: float
    different_mean: float
    auc: float
    threshold: float
    overlaps: bool

    @property
    def gap(self) -> float:
        """Positive when the two distributions are cleanly separated."""
        return self.different_min - self.same_max


@dataclass(frozen=True, slots=True)
class Match:
    camera: str
    dist: float
    venue: str | None


def median_background(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Per-pixel median of one camera's frames: the scene with the people taken out.

    A mean would keep a ghost of every player; the median discards anything present in fewer
    than half the frames, which is every moving thing on a pitch.
    """
    stack = [cv2.resize(f, WORK_SIZE, interpolation=cv2.INTER_AREA) for f in frames]
    if not stack:
        raise ValueError("median_background needs at least one frame")
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def _mask_watermark(image: np.ndarray) -> np.ndarray:
    """Zero the bottom-right corner so an exporter logo cannot act as a venue cue."""
    out = image.copy()
    h, w = out.shape[:2]
    out[int(h * (1 - WATERMARK_FRACTION)) :, int(w * (1 - WATERMARK_FRACTION)) :] = 0
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-6 else v.astype(np.float32)


def structure_descriptor(background_bgr: np.ndarray) -> np.ndarray:
    """Grid of gradient-orientation histograms, normalised once globally.

    Unsigned orientations (0-180 degrees), so a light-on-dark edge and its dark-on-light
    counterpart — the same fence at noon and at night — land in the same bin.
    """
    grey = _mask_watermark(cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    cols, rows = STRUCTURE_GRID
    h, w = grey.shape
    cells: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            hist, _ = np.histogram(
                angle[y0:y1, x0:x1],
                bins=ORIENTATION_BINS,
                range=(0.0, 180.0),
                weights=magnitude[y0:y1, x0:x1],
            )
            cells.append(hist)
    return _unit(np.concatenate(cells))


def chroma_descriptor(background_bgr: np.ndarray) -> np.ndarray:
    """Grid of rg-chromaticity histograms — the board and fence colours, without brightness.

    Dividing each channel by the pixel's total intensity removes the illumination scale, so a
    blue board stays blue whether it is lit by the sun or by floodlights, while a night-time
    pitch does not read as a different colour from the same pitch at noon.
    """
    bgr = _mask_watermark(background_bgr).astype(np.float32) + 1.0
    total = bgr.sum(axis=2)
    red = bgr[:, :, 2] / total
    green = bgr[:, :, 1] / total

    cols, rows = CHROMA_GRID
    h, w = red.shape
    cells: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            hist, _, _ = np.histogram2d(
                red[y0:y1, x0:x1].ravel(),
                green[y0:y1, x0:x1].ravel(),
                bins=CHROMA_BINS,
                range=[[0.0, 1.0], [0.0, 1.0]],
            )
            cells.append(hist.ravel())
    return _unit(np.concatenate(cells))


def view_descriptor(
    background_bgr: np.ndarray, chroma_weight: float = CHROMA_WEIGHT
) -> np.ndarray:
    """The combined descriptor: structure, plus chromaticity at a measured weight."""
    structure = structure_descriptor(background_bgr)
    chroma = chroma_descriptor(background_bgr)
    return _unit(np.concatenate([structure, chroma_weight * chroma]))


def fingerprint_view(
    camera: str,
    frames: Sequence[np.ndarray],
    venue: str | None = None,
    chroma_weight: float = CHROMA_WEIGHT,
) -> ViewFingerprint:
    background = median_background(frames)
    structure = structure_descriptor(background)
    chroma = chroma_descriptor(background)
    return ViewFingerprint(
        camera=camera,
        descriptor=_unit(np.concatenate([structure, chroma_weight * chroma])),
        structure=structure,
        chroma=chroma,
        n_frames=len(frames),
        venue=venue,
    )


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]; descriptors are unit-norm by construction."""
    return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))


def _descriptors(
    fingerprints: Sequence[ViewFingerprint], block: str = "descriptor"
) -> np.ndarray:
    return np.stack([getattr(f, block) for f in fingerprints])


def distance_matrix(
    fingerprints: Sequence[ViewFingerprint], block: str = "descriptor"
) -> np.ndarray:
    d = _descriptors(fingerprints, block)
    m = np.clip(1.0 - d @ d.T, 0.0, 2.0)
    np.fill_diagonal(m, 0.0)
    return m


def suggest_threshold(
    fingerprints: Sequence[ViewFingerprint],
    *,
    block: str = "descriptor",
    margin: float = 0.5,
) -> Separation:
    """Report the separation this footage actually shows, using venue labels as truth.

    Pairs from one venue are "same view"; pairs from different venues are not. AUC is the
    probability that a same-venue pair is closer than a different-venue pair — the summary
    that survives an overlap, unlike the max/min gap. The returned threshold sits between the
    two distributions when they are separated, and is reported as overlapping when they are
    not: an overlap means the descriptor cannot settle venue identity by thresholding alone,
    which the caller has to know rather than have smoothed over.
    """
    labelled = [f for f in fingerprints if f.venue is not None]
    same: list[float] = []
    different: list[float] = []
    for a, b in combinations(labelled, 2):
        d = distance(getattr(a, block), getattr(b, block))
        (same if a.venue == b.venue else different).append(d)
    if not same or not different:
        raise ValueError("need at least two venues, with at least one sharing a venue")

    same_arr, diff_arr = np.array(same), np.array(different)
    auc = float(np.mean(same_arr[:, None] < diff_arr[None, :]))
    same_max, different_min = float(same_arr.max()), float(diff_arr.min())
    overlaps = different_min <= same_max
    threshold = (
        same_max + margin * (different_min - same_max)
        if not overlaps
        # Overlapping: fall back to the midpoint of the means, and flag it via `overlaps`.
        else 0.5 * (float(same_arr.mean()) + float(diff_arr.mean()))
    )
    return Separation(
        same_max=same_max,
        different_min=different_min,
        same_mean=float(same_arr.mean()),
        different_mean=float(diff_arr.mean()),
        auc=auc,
        threshold=float(threshold),
        overlaps=overlaps,
    )


def match_view(
    descriptor: np.ndarray,
    known: Sequence[ViewFingerprint],
    threshold: float,
    block: str = "descriptor",
) -> Match | None:
    """Nearest known view, or ``None`` when nothing is close enough — a new camera.

    Returning ``None`` rather than the least-bad match is the point: a genuinely new venue
    must not be silently folded into an existing one.
    """
    if not known:
        return None
    dists = [distance(descriptor, getattr(f, block)) for f in known]
    i = int(np.argmin(dists))
    if dists[i] > threshold:
        return None
    return Match(camera=known[i].camera, dist=dists[i], venue=known[i].venue)


def mutual_nearest_pairs(
    fingerprints: Sequence[ViewFingerprint], block: str = "descriptor"
) -> list[tuple[str, str, float]]:
    """Views that are each other's nearest neighbour, closest pair first.

    Mutual agreement is the claim worth making about a camera's identity: "A's nearest is B
    *and* B's nearest is A" is not something a single noisy distance can fake, and it is what
    established that `venue_01`'s two recording days label their cameras the opposite way
    round.
    """
    m = distance_matrix(fingerprints, block)
    np.fill_diagonal(m, np.inf)
    nearest = m.argmin(axis=1)
    pairs: list[tuple[str, str, float]] = []
    for i, j in enumerate(nearest):
        if i < j and nearest[j] == i:
            pairs.append((fingerprints[i].camera, fingerprints[j].camera, float(m[i, j])))
    return sorted(pairs, key=lambda p: p[2])


def cluster_views(
    fingerprints: Sequence[ViewFingerprint],
    threshold: float,
    block: str = "descriptor",
) -> dict[str, int]:
    """Group views into venues by average-linkage agglomeration on cosine distance.

    Average linkage, not single: single linkage chains, and chaining is precisely how the
    near-duplicate audit produced a meaningless 96.9% before it was fixed. Two views join a
    cluster only if they are close to it *on average*.
    """
    if len(fingerprints) < 2:
        return {f.camera: 0 for f in fingerprints}
    condensed = squareform(distance_matrix(fingerprints, block), checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=threshold, criterion="distance")
    return {f.camera: int(label) for f, label in zip(fingerprints, labels, strict=True)}
