"""What a pitch boundary actually keeps out of the model, measured three ways (WP3-T1).

An operator drew a boundary, ran the image walkthrough, and saw the evidence map light up
*outside the outline*. The reasonable conclusion - "the boundary is not reaching the model" -
was wrong, and the reason it was wrong is worth a number rather than an explanation.

`roi.apply` replaced the outside of the outline before the frame reached the backbone, so the
neighbouring pitch genuinely was gone. But pooling averaged over **every** patch position,
filled ones included, so the *fill* reached the probe: a large uniform region, of a kind no
pretraining set contains, averaged into the vector the classifier scores and decomposed onto
the map the operator was reading. The boundary had removed one distraction and introduced
another. This script separates the two and measures each.

**Three questions, because they have different answers.**

1. *Can the neighbouring pitch reach the model?* Two frames identical inside the outline and
   completely different outside it. Cosine 1.0 means it cannot. This was already true for the
   `black` and `mean` fills before any of this - and is **not** true for `blur`, which blurs
   the outside rather than replacing it, so a busy neighbouring pitch arrives as a blurred
   busy pitch. That is a real finding about `roi.FILLS` and an argument against `blur` for the
   one job the boundary exists to do.

2. *Does the fill itself reach the model?* The same frame under different fills. Without ROI
   pooling the embeddings differ substantially, which is the fill being scored. With it, the
   filled positions leave the average.

3. *How much of the difference survives ROI pooling?* Not zero, and the honest number. A
   transformer's patch tokens attend to each other and a convnet's receptive field crosses the
   outline, so an inside position has already seen the fill by the time it is pooled. Removing
   the outside positions removes their **direct** contribution - the one the evidence map
   draws, which is why that map is now exactly zero outside the outline - and cannot remove
   the indirect one. No architecture-preserving change can.

    uv run python experiments/roi_pooling_leak.py

Writes `results/roi_pooling_leak.csv`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pitch_occupancy.vision import roi
from pitch_occupancy.vision.backbones import BACKBONES, embed_batch, load_backbone

OUT = Path("results/roi_pooling_leak.csv")

#: Two frames from different slots at the same venue. They need only differ; what is measured
#: is whether a difference *confined to the outside of the outline* moves the embedding.
FRAMES = (
    Path("data/evidence/venue_01_2026-07-11_1000/minute_005_file1.jpg"),
    Path("data/evidence/venue_01_2026-07-12_2030/minute_007_file1.jpg"),
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _inside_mask(polygon, shape) -> np.ndarray:
    height, width = shape[:2]
    points = np.array([[int(round(x * width)), int(round(y * height))] for x, y in polygon],
                      dtype=np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask.astype(bool)[:, :, None]


def main() -> None:
    polygon = next(iter(roi.load_all().values()), None)
    if polygon is None:  # pragma: no cover - depends on the committed configs/roi.json
        raise SystemExit("no boundary in configs/roi.json; draw one at /roi first")

    a = cv2.imread(str(FRAMES[0]))
    b = cv2.imread(str(FRAMES[1]))
    if a is None or b is None:  # pragma: no cover - depends on the evidence corpus
        raise SystemExit(f"could not read {FRAMES[0]} / {FRAMES[1]}")
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    inside = _inside_mask(polygon, a.shape)
    # Same pitch inside the outline, a different scene outside it.
    swapped = np.where(inside, a, b)

    rows = []
    for key in BACKBONES:
        model, processor, spec = load_backbone(key)

        def embed(image, pooled_inside=False, _m=model, _p=processor, _s=spec):
            pil = [Image.fromarray(image[:, :, ::-1])]
            return embed_batch(_m, _p, _s, pil,
                               roi_polygon=polygon if pooled_inside else None)[0]

        for roi_pooled in (False, True):
            for fill in (None, *roi.FILLS):
                left = a if fill is None else roi.apply(a, polygon, fill=fill)
                right = (swapped if fill is None
                         else roi.apply(swapped, polygon, fill=fill))
                neighbour = _cosine(embed(left, roi_pooled), embed(right, roi_pooled))
                # Against the same frame with no boundary at all: how far the boundary moved
                # the vector the probe is asked about.
                shift = _cosine(embed(left, roi_pooled), embed(a))
                rows.append({
                    "backbone": key,
                    "kind": spec.kind,
                    "fill": fill or "none",
                    "roi_pooled": roi_pooled,
                    # 1.0 = a scene change outside the outline cannot reach the model.
                    "cosine_neighbour_blocked": round(neighbour, 5),
                    # Distance from the unbounded embedding the feature cache was built with.
                    "cosine_to_unbounded": round(shift, 5),
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} rows)")
    leaks = [r for r in rows
             if r["fill"] != "none" and r["cosine_neighbour_blocked"] < 0.9999]
    for row in leaks:
        print(f"  LEAKS: {row['backbone']} fill={row['fill']} "
              f"roi_pooled={row['roi_pooled']} -> {row['cosine_neighbour_blocked']}")
    print(json.dumps({"n_fills_leaking_outside_content": len(leaks)}))


if __name__ == "__main__":
    main()
