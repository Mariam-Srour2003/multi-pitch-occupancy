"""Frame extraction from source footage (WP2-T3).

Two kinds of source produce two kinds of frame, and they carry different metadata:

**Slot recordings** (``raw/venue_01/``) are hour-long, from a known pitch, date, time and
camera. All of that is already in the filename, so extracted frames keep the established
``slot_<date>_<time>_<cam>_t<sec>.jpg`` convention and the manifest parses it directly.

**Clips** (``raw/highlights_2026-09-04/``) are 10-14 s highlights whose filenames encode
only the export session. Venue comes from ``configs/clip_venues.csv`` (assigned by
background fingerprint plus visual confirmation), and lighting has to be measured from the
pixels because there is no timestamp to infer it from. Since a filename cannot carry all
of that, extraction writes a sidecar - ``data/interim/clip_frames.csv`` - that the manifest
joins on, exactly as it already joins ``labels.csv`` for provenance.

Frames are sampled from the middle 80% of each clip: the first and last moments of a
highlight often contain a cut or a replay wipe.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2

__all__ = ["ClipFrame", "load_clip_venues", "extract_clip_frames", "CLIP_SIDECAR"]

CLIP_SIDECAR = Path("data/interim/clip_frames.csv")

#: Mean grayscale intensity below which a frame is treated as floodlit rather than
#: daylight. Calibrated against the known day/night recordings in raw/venue_01.
NIGHT_BRIGHTNESS_BELOW = 80.0


@dataclass(frozen=True, slots=True)
class ClipFrame:
    file: str  # basename, as it will appear inside the class folder
    venue: str
    venue_code: str
    clip_id: str
    t_ms: int
    brightness: float
    lighting: str


def load_clip_venues(path: Path | None = None) -> dict[str, tuple[str, str]]:
    """Map clip filename -> (venue, venue_code)."""
    path = path or Path("configs/clip_venues.csv")
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["file"]: (r["venue"], r["venue_code"]) for r in csv.DictReader(fh)}


def extract_clip_frames(
    clips_dir: Path,
    out_dir: Path,
    *,
    venues: dict[str, tuple[str, str]],
    per_clip: int = 6,
    quality: int = 92,
) -> list[ClipFrame]:
    """Sample ``per_clip`` frames from each clip and write them to ``out_dir``.

    Frames are evenly spaced across the middle 80% of the clip. Returns the sidecar rows;
    the caller decides where to persist them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[ClipFrame] = []

    for clip in sorted(clips_dir.glob("*.mp4")):
        if clip.name not in venues:
            raise KeyError(
                f"{clip.name} has no venue assignment in configs/clip_venues.csv - "
                f"add it before extracting, or its frames cannot be grouped for a "
                f"leave-one-venue-out split"
            )
        venue, code = venues[clip.name]
        clip_id = clip.stem.rsplit("-", 1)[-1]

        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise OSError(f"cannot open {clip}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        for i in range(per_clip):
            frac = 0.10 + (0.80 * i / max(per_clip - 1, 1))
            idx = int(total * frac)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            t_ms = int(idx / fps * 1000)
            name = f"clip_{code}_{clip_id}_t{t_ms:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            written.append(
                ClipFrame(
                    file=name,
                    venue=venue,
                    venue_code=code,
                    clip_id=clip_id,
                    t_ms=t_ms,
                    brightness=round(brightness, 2),
                    lighting="night" if brightness < NIGHT_BRIGHTNESS_BELOW else "day",
                )
            )
        cap.release()

    return written


def write_sidecar(rows: list[ClipFrame], path: Path | None = None) -> Path:
    path = path or CLIP_SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f.name for f in fields(ClipFrame)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def read_sidecar(path: Path | None = None) -> dict[str, ClipFrame]:
    """Map frame basename -> its clip metadata. Empty if no clips have been extracted."""
    path = path or CLIP_SIDECAR
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {
            r["file"]: ClipFrame(
                file=r["file"],
                venue=r["venue"],
                venue_code=r["venue_code"],
                clip_id=r["clip_id"],
                t_ms=int(r["t_ms"]),
                brightness=float(r["brightness"]),
                lighting=r["lighting"],
            )
            for r in csv.DictReader(fh)
        }
