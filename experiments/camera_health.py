"""Frame quality and camera health across the dataset (WP3-T5).

Measures every labelled frame, builds a per-camera baseline, judges each frame against its
own camera, and writes:

* `results/camera_health.csv` - one row per camera: baselines, floors, how many of its own
  frames fall below them.
* `results/frame_quality.csv` - one row per frame, for the manifest's `quality` column.

It also runs the comparison that motivates the whole design: what a **global** threshold
would have flagged, against what the per-camera one flags. That number is the finding.

    uv run python experiments/camera_health.py
"""

from __future__ import annotations

import csv
from collections import defaultdict

import cv2

from pitch_occupancy.config import settings
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.vision.quality import (
    FrameQuality,
    assess,
    build_baseline,
    measure,
)

RESULTS = settings.results_dir
HEALTH_CSV = RESULTS / "camera_health.csv"
FRAMES_CSV = RESULTS / "frame_quality.csv"


def camera_key(row: dict) -> str:
    """Which baseline this frame belongs to: a physical camera under one lighting condition.

    **Lighting has to be part of the key.** Without it, a camera's night frames are judged
    against a median dominated by its day frames, and a night frame is then "degraded" for
    being darker than daylight - which is the venue-identity mistake this module exists to
    avoid, one level down. It fired exactly that way: six perfectly readable floodlit frames
    from `clipvenue_g_netting` were flagged at contrast 25.9 against a 43.4 median built
    mostly from that venue's daylight frames.

    **The physical camera, not the filename.** `slot_20260711_1000_camA` and
    `slot_20260712_2030_camB` are the same lens - the `camA`/`camB` suffix flips between
    recording days - so keying on the filename would build two half-baselines per camera and
    make drift between days invisible, which is the one thing a lens-dirt check needs to see.

    Clip venues have no camera identity at all: their `camera` field is a per-clip export id.
    They are grouped by venue and lighting, which mostly leaves them under the trustworthy
    threshold and therefore `unknown`. That is the honest answer - and see the note in
    `results/EXPERIMENT_LOG.md`, because their lighting labels are themselves derived from
    brightness and are wrong for several venues.
    """
    if row["venue"].startswith("clipvenue_"):
        return f"{row['venue']}::{row['lighting']}"
    camera = PHYSICAL_CAMERA.get(row["camera"], row["camera"])
    return f"{row['venue']}::{camera}::{row['lighting']}"


def main() -> None:
    manifest = settings.dataset_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest}")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))

    measured: list[tuple[dict, FrameQuality]] = []
    unreadable = 0
    for row in rows:
        image = cv2.imread(str(settings.dataset_dir / row["file"]))
        if image is None:
            unreadable += 1
            continue
        measured.append((row, measure(image)))
    print(f"measured {len(measured)} frames" + (f" ({unreadable} unreadable)" if unreadable else ""))

    by_camera: dict[str, list[FrameQuality]] = defaultdict(list)
    for row, quality in measured:
        by_camera[camera_key(row)].append(quality)
    baselines = {k: build_baseline(k, v) for k, v in by_camera.items()}

    RESULTS.mkdir(parents=True, exist_ok=True)
    with FRAMES_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "camera", "lighting", "mean_intensity", "rms_contrast",
                         "blur", "clipped_low", "clipped_high", "quality", "reasons"])
        flagged = 0
        for row, q in measured:
            verdict = assess(q, baselines.get(camera_key(row)))
            flagged += verdict.quality == "bad"
            writer.writerow([
                row["file"], camera_key(row), row["lighting"],
                f"{q.mean_intensity:.2f}", f"{q.rms_contrast:.2f}", f"{q.blur:.1f}",
                f"{q.clipped_low:.4f}", f"{q.clipped_high:.4f}",
                verdict.quality, "; ".join(verdict.reasons),
            ])

    with HEALTH_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["camera", "n_frames", "trustworthy", "contrast_median",
                         "contrast_floor", "blur_median", "blur_floor", "n_bad"])
        for key, baseline in sorted(baselines.items()):
            bad = sum(
                1 for row, q in measured
                if camera_key(row) == key and assess(q, baseline).quality == "bad"
            )
            writer.writerow([
                key, baseline.n_frames, baseline.trustworthy,
                f"{baseline.contrast_median:.2f}", f"{baseline.contrast_floor:.2f}",
                f"{baseline.blur_median:.1f}", f"{baseline.blur_floor:.1f}", bad,
            ])

    print(f"per-camera thresholds flagged {flagged} of {len(measured)} frames "
          f"({flagged / len(measured):.1%})")
    print(f"wrote {HEALTH_CSV.name} and {FRAMES_CSV.name}")
    _report_what_a_global_threshold_would_do(measured)


def _report_what_a_global_threshold_would_do(
    measured: list[tuple[dict, FrameQuality]]
) -> None:
    """The comparison that justifies the per-camera design.

    A global blur cutoff does not separate usable frames from unusable ones. It separates
    `venue_01` - soft optics, and the only venue with full-length recordings - from the
    clip venues, then calls the first group broken. Printed with the venue breakdown so the
    claim is checkable rather than asserted.
    """
    blurs = sorted(q.blur for _, q in measured)
    cutoff = blurs[len(blurs) // 10]  # a plausible "bottom 10% are blurry" rule
    below = [row for row, q in measured if q.blur < cutoff]
    if not below:
        return
    venues: dict[str, int] = defaultdict(int)
    for row in below:
        venues[row["venue"]] += 1
    total = defaultdict(int)
    for row, _ in measured:
        total[row["venue"]] += 1
    top = max(venues.items(), key=lambda kv: kv[1])
    print(
        f"\nfor comparison, a global blur cutoff at {cutoff:.0f} would flag "
        f"{len(below)} frames, and {top[1] / len(below):.0%} of them come from "
        f"{top[0]} alone ({top[1]} of that venue's {total[top[0]]} frames) - "
        f"it would be measuring which camera took the frame, not whether it is usable"
    )


if __name__ == "__main__":
    main()
