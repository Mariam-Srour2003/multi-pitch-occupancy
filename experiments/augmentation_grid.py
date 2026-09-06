"""What each augmentation preset actually does to a frame (WP3-T6).

The acceptance criterion for WP3-T6 asks for a visual grid, and the reason is not
decoration. Augmentation code is unusually easy to get silently wrong - a preset that
does nothing, a fog veil that washes the pitch to flat grey, a hue shift that turns turf
a colour no pitch has ever been - and every one of those failures still passes a shape
and dtype check. The only reliable test is a person looking at the output, so this
produces the sheet for them to look at.

Two frames, chosen to stress different things:

* a **night ACTIVE_PLAY** frame, where the model's real work happens and where added
  noise and low gamma are most likely to destroy the people it needs to see;
* a **day EMPTY** frame, where colour jitter has the most turf to act on - and turf hue
  is the venue cue the input ablation implicated.

`p` is forced to 1 for every preset so the sheet shows the effect rather than a coin
flip; at training time each effect fires with probability `p`, so a real training batch
is a mixture of these and the untouched original.

    uv run python experiments/augmentation_grid.py
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import cv2
import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.vision.augment import AUGMENTATIONS, augment

OUT = settings.results_dir / "figs" / "augmentation_grid.jpg"
TILE_W, TILE_H = 320, 180
LABEL_H = 26
SEEDS = (11, 12, 13)  # three draws per preset; one draw hides the variation


def _pick_frames() -> list[tuple[str, Path]]:
    """One night play frame and one day empty frame, by manifest order for stability."""
    rows = read_manifest(settings.dataset_dir / "manifest.csv")
    wanted = [
        ("night · active play", lambda r: r.class3 == "C2_ACTIVE_PLAY" and r.lighting == "night"),
        ("day · empty", lambda r: r.class3 == "C1_EMPTY" and r.lighting == "day"),
    ]
    picked: list[tuple[str, Path]] = []
    for caption, match in wanted:
        for row in rows:
            if match(row):
                picked.append((caption, settings.dataset_dir / row.file))
                break
    return picked


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    """A caption strip below the tile. Burned in, because a grid whose labels live in a
    separate legend stops being readable the moment it is cropped into a slide."""
    strip = np.full((LABEL_H, tile.shape[1], 3), 22, np.uint8)
    cv2.putText(strip, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1,
                cv2.LINE_AA)
    return np.vstack([tile, strip])


def build_sheet() -> np.ndarray:
    frames = _pick_frames()
    if not frames:
        raise SystemExit("no frames matched; is data/processed/manifest.csv present?")

    blocks: list[np.ndarray] = []
    for caption, path in frames:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not read {path}")
        base = cv2.resize(image, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

        # the original sits alone in the first column so it lines up with the first draw
        # of every preset below it - the pairing a reader compares against
        row_w = TILE_W * len(SEEDS)
        first = np.hstack([base, np.full((TILE_H, row_w - TILE_W, 3), 22, np.uint8)])
        rows = [_label(first, f"original — {caption}")]
        for name, cfg in AUGMENTATIONS.items():
            if name == "none":
                continue  # identical to the original by construction, and tested as such
            forced = replace(cfg, p=1.0)
            draws = [
                augment(base, forced, rng=np.random.default_rng(seed)) for seed in SEEDS
            ]
            rows.append(_label(np.hstack(draws), f"{name} — three draws (p forced to 1)"))
        blocks.append(np.vstack(rows))

    width = max(b.shape[1] for b in blocks)
    padded = [
        np.hstack([b, np.full((b.shape[0], width - b.shape[1], 3), 22, np.uint8)])
        if b.shape[1] < width else b
        for b in blocks
    ]
    gap = np.full((14, width, 3), 22, np.uint8)
    return np.vstack([x for pair in zip(padded, [gap] * len(padded)) for x in pair][:-1])


def main() -> None:
    sheet = build_sheet()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {OUT}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print(
        "Look for: people still visible in every night draw (if not, the noise or gamma "
        "range is too wide), and turf that still reads as turf in every colour draw."
    )


if __name__ == "__main__":
    main()
