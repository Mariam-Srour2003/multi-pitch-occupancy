"""What each augmentation and preprocessing switch actually does to a frame (WP3-T6, WP3).

The acceptance criterion for WP3-T6 asks for a visual grid, and the reason is not
decoration. Augmentation code is unusually easy to get silently wrong - a preset that
does nothing, a fog veil that washes the pitch to flat grey, a hue shift that turns turf
a colour no pitch has ever been - and every one of those failures still passes a shape
and dtype check. The only reliable test is a person looking at the output, so this
produces the sheets for them to look at.

Three sheets, because they answer three different questions:

* `augmentation_grid.jpg` - **every preset**, which is what a training batch actually
  draws from;
* `augmentation_effects.jpg` - **one effect at a time**, which is the only way to see
  which knob produced which part of a preset. A preset row shows four effects compounded;
  when one of them is wrong, the compound row shows that something is wrong and not what;
* `preprocess_effects.jpg` - **every preprocessing switch**, at the model's own input
  size, so the removal-versus-variation argument can be looked at rather than read. This
  one is deterministic: preprocessing has no draw.

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

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.vision.augment import AUGMENTATIONS, AugmentConfig, augment
from pitch_occupancy.vision.explain import redact_people
from pitch_occupancy.vision.preprocess import SWITCHES, PreprocessConfig, preprocess

FIGS = settings.results_dir / "figs"
OUT = FIGS / "augmentation_grid.jpg"
OUT_EFFECTS = FIGS / "augmentation_effects.jpg"
OUT_PREPROCESS = FIGS / "preprocess_effects.jpg"
TILE_W, TILE_H = 320, 180
LABEL_H = 26
SEEDS = (11, 12, 13)  # three draws per preset; one draw hides the variation

#: Preprocessing tiles are square because that is the shape the model receives. Drawn at the
#: real 224 rather than scaled up, so a reader judging whether a switch destroyed the players
#: is looking at the pixels the backbone gets and not at an interpolation of them.
PRE_TILE = PreprocessConfig().size
PRE_COLUMNS = 6


def _pick_frames() -> list[tuple[str, Path]]:
    """One night play frame and one day empty frame, by manifest order for stability."""
    rows = read_manifest(settings.dataset_dir / "manifest.csv")
    wanted = [
        ("night - active play", lambda r: r.class3 == "C2_ACTIVE_PLAY" and r.lighting == "night"),
        ("day - empty", lambda r: r.class3 == "C1_EMPTY" and r.lighting == "day"),
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


def _redacted_frames() -> list[tuple[str, np.ndarray, int]]:
    """The source frames, faces pixelated, before anything is drawn.

    Redacted here rather than inside each sheet builder, so a sheet added later cannot be
    the one that forgets. These are committed to git and served on the thesis site, and
    `thesis/ethics.md` commits to blurring faces in any published figure. `explain.py` has
    said the redaction "is not optional and not a flag" since it was written - and this
    script, which publishes real night-match frames, was not calling it. Every sheet below
    augments or preprocesses the *redacted* frame, so what a reader sees is exactly what
    they can check.
    """
    frames = _pick_frames()
    if not frames:
        raise SystemExit("no frames matched; is data/processed/manifest.csv present?")

    out: list[tuple[str, np.ndarray, int]] = []
    for caption, path in frames:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not read {path}")
        image, n_people = redact_people(image)
        if n_people < 0:
            raise SystemExit(
                "the person detector could not be loaded, so this frame cannot be shown to "
                "have been redacted. Refusing to write the sheet rather than publishing an "
                "unredacted night match - see redact_people's -1 contract."
            )
        out.append((caption, image, n_people))
    return out


def _stack_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    """Pad blocks to one width and stack them with a gap between."""
    width = max(b.shape[1] for b in blocks)
    padded = [
        np.hstack([b, np.full((b.shape[0], width - b.shape[1], 3), 22, np.uint8)])
        if b.shape[1] < width else b
        for b in blocks
    ]
    gap = np.full((14, width, 3), 22, np.uint8)
    return np.vstack(
        [x for pair in zip(padded, [gap] * len(padded), strict=True) for x in pair][:-1]
    )


def build_sheet(frames: list[tuple[str, np.ndarray, int]]) -> np.ndarray:
    """Every preset, three draws each - what a training batch draws from."""
    blocks: list[np.ndarray] = []
    for caption, image, _ in frames:
        base = cv2.resize(image, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

        # the original sits alone in the first column so it lines up with the first draw
        # of every preset below it - the pairing a reader compares against
        row_w = TILE_W * len(SEEDS)
        first = np.hstack([base, np.full((TILE_H, row_w - TILE_W, 3), 22, np.uint8)])
        rows = [_label(first, f"original - {caption}")]
        for name, cfg in AUGMENTATIONS.items():
            if name == "none":
                continue  # identical to the original by construction, and tested as such
            forced = replace(cfg, p=1.0)
            draws = [
                augment(base, forced, rng=np.random.default_rng(seed)) for seed in SEEDS
            ]
            rows.append(_label(np.hstack(draws), f"{name} - three draws (p forced to 1)"))
        blocks.append(np.vstack(rows))
    return _stack_blocks(blocks)


def single_effects() -> dict[str, AugmentConfig]:
    """One effect at a time, at the magnitude the ``full`` preset uses for it.

    Derived from the preset rather than retyped, so the sheet cannot drift from the config
    it claims to illustrate: change `full` and this changes with it.

    ``flip`` is the exception and is forced to 1.0. Its field is a probability rather than a
    magnitude, so at the preset's 0.5 roughly half the draws would show an unflipped frame -
    a row that looks like an augmentation which sometimes does nothing. That is true of the
    *preset* and useless as an illustration of the *effect*.
    """
    full = AUGMENTATIONS["full"]
    return {
        name: AugmentConfig(**{name: 1.0 if name == "flip" else getattr(full, name)}, p=1.0)
        for name in full.enabled()
    }


def build_effect_sheet(frames: list[tuple[str, np.ndarray, int]]) -> np.ndarray:
    """One row per individual effect - which knob did which part of a preset.

    The preset sheet compounds up to nine effects into one tile. That is the right picture
    of what training sees and the wrong picture for finding a broken effect: a `full` row
    that looks wrong says something in it is wrong and not which. The rain-geometry bug
    found on the first run was visible only because rain happened to dominate its row.
    """
    blocks: list[np.ndarray] = []
    for caption, image, _ in frames:
        base = cv2.resize(image, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        row_w = TILE_W * len(SEEDS)
        first = np.hstack([base, np.full((TILE_H, row_w - TILE_W, 3), 22, np.uint8)])
        rows = [_label(first, f"original - {caption}")]

        for name, cfg in single_effects().items():
            if name == "flip":
                # A mirror is deterministic: three draws would be three identical tiles,
                # which reads as a bug in the sheet rather than a property of the effect.
                flipped = augment(base, cfg, rng=np.random.default_rng(SEEDS[0]))
                tile = np.hstack(
                    [flipped, np.full((TILE_H, row_w - TILE_W, 3), 22, np.uint8)]
                )
                rows.append(_label(tile, "flip - deterministic, one mirror"))
                continue
            draws = [augment(base, cfg, rng=np.random.default_rng(s)) for s in SEEDS]
            rows.append(
                _label(np.hstack(draws), f"{name}={getattr(cfg, name):g} - three draws")
            )
        blocks.append(np.vstack(rows))
    return _stack_blocks(blocks)


def preprocess_variants() -> list[tuple[str, PreprocessConfig]]:
    """Baseline, then every switch value the preprocessing search actually tries.

    Built from `preprocess.SWITCHES` so the sheet shows the searched space rather than a
    hand-picked subset of it. Two deliberate departures, both stated on the page:

    * ``roi`` is absent from `SWITCHES` because no pitch polygon has been drawn yet, and
      `roi_mask` with no polygon returns the frame untouched. Drawing it here would put a
      tile captioned "ROI" beside an unmodified frame, which is the most misleading thing
      this sheet could do.
    * ``blur_sigma`` is absent from `SWITCHES` because it is a diagnostic rather than a
      setting - but it is the variant that made the input ablation's point (blur at
      sigma=4 calls 100% of held-out empty pitches a match), so it is shown and labelled
      as one.
    """
    variants: list[tuple[str, PreprocessConfig]] = [("baseline", PreprocessConfig())]
    for switch, values in SWITCHES.items():
        for value in values:
            variants.append((f"{switch}={value}", PreprocessConfig(**{switch: value})))
    variants.append(("blur_sigma=4.0 (diagnostic)", PreprocessConfig(blur_sigma=4.0)))
    return variants


def build_preprocess_sheet(frames: list[tuple[str, np.ndarray, int]]) -> np.ndarray:
    """Every preprocessing switch, at the size the model receives.

    Preprocessing is deterministic, so there is nothing to draw three times. What this
    sheet is for is the claim the augmentation page makes in words: removal has a floor. A
    reader who can see that `blur_sigma=4.0` has removed the people, and that
    `centre_crop=0.5` has removed the goalmouth an empty-pitch judgement depends on, does
    not have to take the false-play numbers on trust.
    """
    variants = preprocess_variants()
    blocks: list[np.ndarray] = []
    for caption, image, _ in frames:
        tiles = [_label(preprocess(image, cfg), name) for name, cfg in variants]
        # A caption strip, not a blank tile. Built at tile height first, it left a 224px
        # void above each block that read as a missing row.
        rows = [
            _label(
                np.full((6, PRE_TILE * PRE_COLUMNS, 3), 22, np.uint8),
                f"preprocessing - {caption}  ({PRE_TILE}x{PRE_TILE}, what the model receives)",
            )
        ]
        for i in range(0, len(tiles), PRE_COLUMNS):
            row = tiles[i : i + PRE_COLUMNS]
            if len(row) < PRE_COLUMNS:
                row = [
                    *row,
                    np.full(
                        (row[0].shape[0], PRE_TILE * (PRE_COLUMNS - len(row)), 3),
                        22, np.uint8,
                    ),
                ]
            rows.append(np.hstack(row))
        blocks.append(np.vstack(rows))
    return _stack_blocks(blocks)


def main() -> None:
    frames = _redacted_frames()
    FIGS.mkdir(parents=True, exist_ok=True)

    for path, sheet in (
        (OUT, build_sheet(frames)),
        (OUT_EFFECTS, build_effect_sheet(frames)),
        (OUT_PREPROCESS, build_preprocess_sheet(frames)),
    ):
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")

    # The count, not just the fact. A detector that found nobody has not established that
    # there was nobody, and a sheet that says "redacted" without saying how many were found
    # is the kind of reassurance this project keeps discovering to be empty.
    print("\nredaction (yolov8n, people pixelated before drawing):")
    for caption, _, n in frames:
        note = "  <- none found; that is not the same as none present" if n == 0 else ""
        print(f"  {caption:<28} {n} person box(es){note}")
    print(
        "\nLook for: people still visible in every night draw (if not, the noise or gamma "
        "range is too wide), turf that still reads as turf in every colour draw, and - on "
        "the preprocessing sheet - which switches have removed the thing the label depends "
        "on rather than merely changed how it looks."
    )


if __name__ == "__main__":
    main()
