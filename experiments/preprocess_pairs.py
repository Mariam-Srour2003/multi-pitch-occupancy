"""Before and after, one preprocessing switch at a time, with the change measured (WP8-T5).

`augmentation_grid.py` already draws every switch on one contact sheet, and that sheet is the
right artefact for judging the whole searched space at once. It is the wrong artefact for
answering *what did this switch do?* - the baseline sits in the top-left corner, six tiles
away from the switch you are looking at, so the comparison happens in the reader's memory
rather than on the page.

So this writes one image per switch: the untouched frame and the switched frame side by side,
same scale, a divider between them, captioned with which is which. A reader can see the
difference without holding anything in mind.

**And it measures the difference, which the contact sheet cannot.** Two switches in this
search are near no-ops on this footage and look identical to a baseline tile at a glance;
one of them (`denoise`) had weaker values dropped from the search precisely because they
barely altered the frame. A pair of pictures invites "looks the same to me", which is an
impression. So each pair carries three numbers:

* **mean absolute change** per pixel channel, on 0-255 - how far the frame moved;
* **share of pixels changed** by more than 2/255 - whether it moved everywhere or somewhere;
* **frame area retained** - the one number the geometry switches turn on, and the one a
  side-by-side of two same-sized tiles actively hides, because a crop is re-letterboxed
  back to 224 and therefore looks like a zoom rather than a loss.

Those three separate "changed how it looks" from "removed what the label depends on", which
is the distinction the input ablation is about.

Frames are redacted before anything is drawn - `redact_people` first, exactly as
`augmentation_grid.py` does, because these files are committed and published.

    uv run python experiments/preprocess_pairs.py

Writes `results/figs/preproc/*.jpg` and `results/preprocess_pairs.csv`. The models and
preprocessing pages read both.
"""

from __future__ import annotations

import csv

import cv2
import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.vision.explain import redact_people
from pitch_occupancy.vision.preprocess import SWITCHES, PreprocessConfig, preprocess

FIGS = settings.results_dir / "figs" / "preproc"
OUT_CSV = settings.results_dir / "preprocess_pairs.csv"

TILE = PreprocessConfig().size  # 224 - drawn at the real input size, never interpolated
GAP = 10  # divider between before and after
LABEL_H = 24
CHANGE_EPS = 2.0 / 255.0  # a pixel counts as changed above this, in 0-1 units

#: What each switch is a hypothesis *about*. Taken from the field docstrings in
#: `vision/preprocess.py` and kept short enough to sit under a picture. A switch whose
#: caption here disagrees with its docstring there is a bug in one of the two.
HYPOTHESES = {
    "undistort": "The footage is fisheye and the lens differs per venue, so geometry is "
                 "itself a venue cue. Straightening it should make a pitch look more like "
                 "a pitch and less like *that* pitch.",
    "centre_crop": "The border carries stands, sky and adjacent pitches, none of which "
                   "transfer. Keep only the middle.",
    "top_crop": "More targeted than a symmetric crop: sky, stands and adjacent pitches sit "
                "above the horizon, while the pitch does not.",
    "letterbox": "Off means a squashing resize instead of aspect-preserving padding. Worth "
                 "about 0.03 recall on this fisheye footage - so this row is what that "
                 "0.03 looks like.",
    "per_image_standardise": "Z-score the frame, removing the global brightness offset that "
                             "separates a daylight morning from a floodlit night - i.e. it "
                             "attacks the day/night confound head-on.",
    "clahe": "Local contrast equalisation. `auto` gates on RMS contrast below 40, which on "
             "this footage fires for 78% of frames - a weak switch, not a vacuous one.",
    "gamma": "Below 1 brightens, above 1 darkens. Night frames are dark; a lift may expose "
             "players the backbone would otherwise miss.",
    "saturation": "Scale colour. 0.0 is grayscale, which helped alone (+0.022 recall) and "
                  "broke when combined with cropping - so this is a dial, not a switch.",
    "denoise": "Bilateral filter. Night footage is noisy; smoothing noise while keeping "
               "edges may help where a plain blur hurts.",
    "sharpen": "Unsharp mask, countering the fisheye softness at the frame edge.",
    "blur_sigma": "A diagnostic, not a setting. It removes the detail that people are made "
                  "of, so a score that survives it was never about people.",
}

#: Which stage of the fixed pipeline each switch belongs to. The order is
#: geometry -> photometric -> resize -> post-resize, and it is fixed rather than searched.
STAGE = {
    "undistort": "geometry", "centre_crop": "geometry", "top_crop": "geometry",
    "letterbox": "resize",
    "per_image_standardise": "photometric", "clahe": "photometric", "gamma": "photometric",
    "saturation": "photometric", "denoise": "photometric",
    "sharpen": "post-resize", "blur_sigma": "post-resize",
}


def variants() -> list[tuple[str, str, object, PreprocessConfig]]:
    """``(switch, label, value, config)`` for every value the search actually tries.

    Built from `preprocess.SWITCHES` so this shows the searched space rather than a
    hand-picked subset, with the same two departures `augmentation_grid.py` documents:
    `roi` is absent because no pitch polygon exists yet and drawing a tile captioned "ROI"
    beside an unmodified frame would be the most misleading thing here; `blur_sigma` is
    absent from the search but is the variant that made the input ablation's point, so it
    is shown and labelled as a diagnostic.
    """
    out: list[tuple[str, str, object, PreprocessConfig]] = []
    for switch, values in SWITCHES.items():
        for value in values:
            out.append((switch, f"{switch}={value}", value, PreprocessConfig(**{switch: value})))
    out.append(("blur_sigma", "blur_sigma=4.0", 4.0, PreprocessConfig(blur_sigma=4.0)))
    return out


def area_retained(switch: str, value: object) -> float:
    """Fraction of the original frame still present after the geometry switches.

    Computed from the switch rather than from the pixels, because both crops are
    re-letterboxed back to 224x224: the output is the same size as the baseline, so no
    measurement of the *output* can see that anything was discarded. This is the number the
    side-by-side hides.
    """
    if switch == "centre_crop":
        return float(value) ** 2
    if switch == "top_crop":
        return 1.0 - float(value)
    return 1.0


def _strip(width: int, text: str, *, height: int = LABEL_H, scale: float = 0.44) -> np.ndarray:
    strip = np.full((height, width, 3), 22, np.uint8)
    cv2.putText(strip, text, (7, height - 8), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (235, 235, 235), 1, cv2.LINE_AA)
    return strip


def pair_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Before on the left, after on the right, captioned and divided."""
    gap = np.full((TILE, GAP, 3), 22, np.uint8)
    body = np.hstack([before, gap, after])
    caption = np.hstack([
        _strip(TILE, "BEFORE  (baseline)"),
        np.full((LABEL_H, GAP, 3), 22, np.uint8),
        _strip(TILE, "AFTER"),
    ])
    return np.vstack([body, caption])


def main() -> None:
    rows = read_manifest(settings.dataset_dir / "manifest.csv")
    # One night ACTIVE_PLAY frame: the switches that matter are the ones that can destroy
    # the people the model needs, and a day empty frame has no people to destroy. The day
    # EMPTY frame stays on the contact sheet, which is the artefact for comparing scenes.
    source = next(
        (r for r in rows if r.class3 == "C2_ACTIVE_PLAY" and r.lighting == "night"), None
    )
    if source is None:
        raise SystemExit("no night active-play frame in the manifest")

    raw = cv2.imread(str(settings.dataset_dir / source.file))
    if raw is None:
        raise SystemExit(f"could not read {source.file}")
    raw, n_people = redact_people(raw)
    if n_people < 0:
        raise SystemExit("redaction unavailable - refusing to write publishable frames")

    baseline = preprocess(raw, PreprocessConfig())
    base_f = baseline.astype(np.float32) / 255.0

    FIGS.mkdir(parents=True, exist_ok=True)
    records = []
    for switch, label, value, cfg in variants():
        after = preprocess(raw, cfg)
        after_f = after.astype(np.float32) / 255.0
        diff = np.abs(after_f - base_f)

        name = label.replace("=", "_").replace(".", "p")
        path = FIGS / f"{name}.jpg"
        cv2.imwrite(str(path), pair_image(baseline, after), [cv2.IMWRITE_JPEG_QUALITY, 94])

        records.append({
            "switch": switch,
            "label": label,
            "value": value,
            "stage": STAGE[switch],
            "searched": switch in SWITCHES,
            "file": f"figs/preproc/{path.name}",
            "mean_abs_change_255": round(float(diff.mean()) * 255, 2),
            "share_pixels_changed": round(float((diff.max(axis=2) > CHANGE_EPS).mean()), 4),
            "area_retained": round(area_retained(switch, value), 4),
            "hypothesis": HYPOTHESES[switch],
        })

    records.sort(key=lambda r: -r["mean_abs_change_255"])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    print(f"source frame: {source.file}  ({n_people} person box(es) redacted)")
    print(f"\n{'switch':<26} {'mean abs':>9} {'pixels':>8} {'area':>7}")
    for r in records:
        print(f"{r['label']:<26} {r['mean_abs_change_255']:>9.2f} "
              f"{r['share_pixels_changed']:>8.1%} {r['area_retained']:>7.0%}")
    print(f"\nwrote {len(records)} pairs to {FIGS} and {OUT_CSV.name}")
    print("\nA switch near the bottom of that table changed the frame very little, and a "
          "searched margin on top of it is not a margin - see search_resolution.csv.")


if __name__ == "__main__":
    main()
