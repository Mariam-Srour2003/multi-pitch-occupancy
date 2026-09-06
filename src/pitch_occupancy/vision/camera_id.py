"""Identifying which physical view a recording came from (WP2-T10).

The export convention marks the second camera of a slot with a ``(1)`` suffix, and it is
tempting to read that as "camera B". **It is not stable**: across the two recorded days the
suffix maps to the opposite physical view. Anything that keys a camera by filename suffix
is therefore correct on one day and silently wrong on the other - and "silently" is the
problem, because fusion still produces a plausible verdict from swapped halves.

So a camera is identified by *what it sees*. The descriptor has to survive the thing that
changes most between recordings - a daylight morning and a floodlit night are the same
pitch under wildly different illumination - so it is built from structure rather than
brightness or colour:

1. take the **median** of several frames, which removes moving players and leaves the
   fixed background;
2. convert to grayscale and equalise, discarding absolute illumination;
3. take the **gradient magnitude**, keeping edges - fences, pitch lines, buildings, stands;
4. downscale and L2-normalise, so what remains is the layout of the view.

Two recordings of the same view correlate highly on this descriptor even across day and
night; two halves of the same pitch do not, because they look at opposite ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["ViewDescriptor", "describe_view", "similarity", "match_cameras"]


@dataclass(frozen=True, slots=True)
class ViewDescriptor:
    source: str
    vector: np.ndarray  # L2-normalised gradient-structure descriptor

    def __len__(self) -> int:
        return int(self.vector.size)


def _structure(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    small = cv2.resize(mag, size, interpolation=cv2.INTER_AREA).ravel()
    norm = np.linalg.norm(small)
    return small / norm if norm else small


def describe_view(
    video_path: Path, *, n_frames: int = 7, size: tuple[int, int] = (48, 27)
) -> ViewDescriptor:
    """Build a lighting-invariant descriptor of a recording's fixed background."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        frac = (i + 0.5) / n_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ok, frame = cap.read()
        if ok:
            frames.append(frame.astype(np.float32))
    cap.release()
    if not frames:
        raise OSError(f"no readable frames in {video_path}")

    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return ViewDescriptor(str(video_path), _structure(background, size))


def similarity(a: ViewDescriptor, b: ViewDescriptor) -> float:
    """Cosine similarity of two view descriptors, in [0, 1] for these non-negative vectors."""
    return float(np.dot(a.vector, b.vector))


def match_cameras(
    descriptors: dict[str, ViewDescriptor],
    references: dict[str, ViewDescriptor],
    *,
    min_similarity: float = 0.0,
) -> dict[str, str | None]:
    """Assign each recording the reference camera whose view it matches.

    Greedy best-first over the full similarity matrix, so the strongest pairing is fixed
    first and each reference is used once - a per-recording argmax could hand both halves
    of a pitch to the same camera.

    Returns recording key -> reference camera id, or ``None`` where nothing cleared
    ``min_similarity``. An unmatched recording is left unassigned rather than forced onto
    its nearest reference: a wrong camera id silently swaps the halves of a pitch, and a
    missing one is at least visible.
    """
    pairs = sorted(
        (
            (similarity(desc, ref), key, ref_id)
            for key, desc in descriptors.items()
            for ref_id, ref in references.items()
        ),
        key=lambda t: -t[0],
    )
    assigned: dict[str, str | None] = {k: None for k in descriptors}
    used: set[str] = set()
    for score, key, ref_id in pairs:
        if assigned[key] is None and ref_id not in used and score >= min_similarity:
            assigned[key] = ref_id
            used.add(ref_id)
    return assigned
