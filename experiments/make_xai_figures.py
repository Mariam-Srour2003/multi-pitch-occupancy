"""XAI overlays for the thesis (WP4-T5) -> `results/figs/xai/`.

Answers the question an examiner asks about any frozen-backbone classifier: *is it reading
the pitch, or the sky?* This project has a specific reason to worry - the clock rule scored
98.4% without looking at pixels, and the day/night confound has now been found five times - so
"where is it looking" is not decoration here, it is the confound check in visual form.

**Two views per frame, and one of them is exact.** Every probe is a logistic regression on
mean-pooled frozen features, which is linear end to end, so the class score is *exactly* the
mean of a per-position contribution. That map is not an attribution heuristic with parameters
to tune; it is the summands of the score. Every figure is generated with the reconstruction
check printed beside it - the score rebuilt from the map against the score the probe returns -
and the run aborts if they ever disagree, because a decomposition that does not reconstruct is
explaining a different model. Attention rollout is the second, classifier-independent view,
available for the two transformers and not for ConvNeXtV2, which has no attention.

**Every frame is redacted before it is written.** Two layers, because one is not enough: YOLO
pixelates each detected person, and then a floor of blur is applied to the *whole* frame so a
missed detection is still not an identifiable face. The heatmap is computed on the original
frame and drawn over the redacted copy, which has identical geometry - so the explanation
stays faithful while the picture stays publishable. Explaining the redacted frame instead
would produce an honest picture of a model looking at pixelation.

That redaction is not a flag and not a default that can be turned off from the command line.
These images go into git history, where they are permanent; WP1-T5 (the data-release decision)
is what governs publishing frames from this dataset, and the cheapest way to stay inside any
decision it eventually reaches is to write nothing identifiable now. The detection count is
printed per frame, and `people detected: 0` means the detector found none - never that the
frame was verified empty.

    uv run python experiments/make_xai_figures.py
    uv run python experiments/make_xai_figures.py --per-class 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision.backbones import load_backbone
from pitch_occupancy.vision.explain import (
    SUPPORTS_ATTENTION,
    ExplainedFrame,
    attention_rollout,
    class_evidence_map,
    detect_people,
    evidence_on_people,
    load_for_attention,
    overlay_heatmap,
    pixelate_boxes,
    probe_weights,
    spatial_features,
)
from pitch_occupancy.vision.heads import LinearProbe

SEED = 42
OUT = settings.results_dir / "figs" / "xai"
BACKBONES = ("dinov2", "convnextv2", "vit")

#: Tolerance on the exactness claim. This is float64 arithmetic over a few hundred positions,
#: so anything above this is a bug in the decomposition rather than accumulated error.
RECONSTRUCTION_TOLERANCE = 1e-6

#: The blur applied to the whole frame regardless of what the detector found. Chosen so a face
#: at this dataset's scale is unrecoverable while the pitch, the lines and the crowd/turf
#: boundary - the things a heatmap is read against - stay legible.
FLOOR_BLUR_KERNEL = 21


def redact(image_bgr: np.ndarray, boxes) -> np.ndarray:
    """Pixelate the detected people, then blur everything. Both, not either.

    Takes the boxes rather than detecting again, so the redaction and the
    evidence-on-people measurement below are guaranteed to be about the same detections.
    """
    import cv2

    out = pixelate_boxes(image_bgr, boxes or [])
    return cv2.GaussianBlur(out, (FLOOR_BLUR_KERNEL, FLOOR_BLUR_KERNEL), 0)


def choose_frames(rows, per_class: int) -> list:
    """A few frames per class, spread across venue and lighting rather than taken in order.

    Consecutive manifest rows are near-duplicates - 98.5% of this dataset has one - so the
    first three frames of a class would be three pictures of the same minute.
    """
    rng = np.random.default_rng(SEED)
    chosen = []
    for cls in sorted({r.class3 for r in rows}):
        pool = [r for r in rows if r.class3 == cls]
        buckets: dict[tuple[str, str], list] = {}
        for r in pool:
            buckets.setdefault((r.venue, r.lighting), []).append(r)
        keys = sorted(buckets)
        for i in range(min(per_class, len(pool))):
            bucket = buckets[keys[i % len(keys)]]
            chosen.append(bucket[int(rng.integers(len(bucket)))])
    return chosen


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=2, help="frames explained per class")
    args = ap.parse_args()

    rows_all = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    OUT.mkdir(parents=True, exist_ok=True)

    caches = {}
    for key in BACKBONES:
        d = np.load(settings.feature_cache_dir / f"{key}.npz", allow_pickle=True)
        caches[key] = ({str(f): i for i, f in enumerate(d["files"])}, d["features"])

    rows = [r for r in rows_all if all(r.file in idx for idx, _ in caches.values())]
    frames = choose_frames(rows, args.per_class)
    print(f"{len(rows)} frames available; explaining {len(frames)}\n")

    explained: list[ExplainedFrame] = []
    written: list[Path] = []
    focus: list[dict] = []

    for key in BACKBONES:
        index, features = caches[key]
        X = features[[index[r.file] for r in rows]]
        probe = LinearProbe(key, seed=SEED).fit(X, rows)
        classes = [str(c) for c in probe.classes_]
        model, processor, spec = load_backbone(key)
        attn_model = attn_processor = None
        if key in SUPPORTS_ATTENTION:
            attn_model, attn_processor, _ = load_for_attention(key)

        print(f"=== {key} ===")
        for row in frames:
            image = cv2.imread(str(settings.dataset_dir / row.file))
            if image is None:
                print(f"  skipped unreadable {row.file}")
                continue

            vector = features[index[row.file]][None, :]
            predicted = str(probe.predict(vector, [row])[0])
            weights, constant = probe_weights(probe, row.class3)
            feats, grid = spatial_features(model, processor, spec, image)
            evidence, score = class_evidence_map(
                feats, weights, constant, grid, drop_first=spec.kind != "convnet"
            )
            direct = float(
                probe._model.decision_function(vector)[0][classes.index(row.class3)]
            )
            if abs(score - direct) > RECONSTRUCTION_TOLERANCE:
                raise AssertionError(
                    f"{key}/{row.file}: the evidence map reconstructs {score:.8f} but the "
                    f"probe scores {direct:.8f}. The decomposition is explaining a different "
                    f"function from the one the probe computes; do not publish these figures."
                )

            attention = None
            if attn_model is not None:
                attention = attention_rollout(attn_model, attn_processor, image)

            boxes = detect_people(image)
            people = -1 if boxes is None else len(boxes)
            on_people, area = evidence_on_people(evidence, boxes or [], image.shape[:2])
            concentration = (on_people / area) if area > 1e-9 else float("nan")
            safe = redact(image, boxes)
            panels = [safe, overlay_heatmap(safe, evidence)]
            if attention is not None:
                panels.append(overlay_heatmap(safe, attention))
            sheet = np.hstack([cv2.resize(p, (320, 320)) for p in panels])

            name = f"{key}__{row.class3.split('_', 1)[1].lower()}__{Path(row.file).stem}.png"
            path = OUT / name
            cv2.imwrite(str(path), sheet)
            written.append(path)
            explained.append(ExplainedFrame(
                file=row.file, backbone=key, predicted=predicted,
                explained_class=row.class3, evidence=evidence,
                score_from_map=score, score_direct=direct, attention=attention,
            ))
            focus.append({"backbone": key, "class": row.class3, "file": row.file,
                          "n_people": people, "evidence_on_people": on_people,
                          "person_area": area, "concentration": concentration})
            mark = "ok " if predicted == row.class3 else "MIS"
            print(f"  {mark} {row.class3.split('_', 1)[1].lower():<24} "
                  f"score {score:+8.3f} (rebuilt {abs(score - direct):.0e})  "
                  f"people {people:<3} evidence-on-people {on_people:5.2f} "
                  f"vs area {area:5.2f} = {concentration:5.2f}x")

    print("\n=== exactness ===")
    worst = max(e.reconstruction_error for e in explained) if explained else 0.0
    print(f"  worst reconstruction error over {len(explained)} explanations: {worst:.2e}")
    print("  the map is the summands of the score, not an approximation of it")

    with_attention = [e for e in explained if e.attention is not None]
    print("\n=== attention ===")
    print(f"  {len(with_attention)} of {len(explained)} explanations carry an attention "
          f"rollout; ConvNeXtV2 has no attention to roll out")

    # The measurement the pictures exist to support. A model reading *players* puts far more
    # of its positive evidence on them than their share of the frame; a ratio near 1 means the
    # evidence is spread as though the people were not there, which is the visual form of the
    # confound this project keeps finding.
    print("\n=== is the evidence on the players? (ACTIVE_PLAY frames with detections) ===")
    print(f"  {'backbone':<12}{'frames':>7}{'evidence':>10}{'area':>8}{'ratio':>9}")
    play_rows = [f for f in focus if f["class"] == "C2_ACTIVE_PLAY" and f["n_people"] > 0]
    for key in BACKBONES:
        mine = [f for f in play_rows if f["backbone"] == key]
        if not mine:
            continue
        ev = float(np.mean([f["evidence_on_people"] for f in mine]))
        ar = float(np.mean([f["person_area"] for f in mine]))
        ratio = float(np.mean([f["concentration"] for f in mine]))
        print(f"  {key:<12}{len(mine):>7}{ev:>10.3f}{ar:>8.3f}{ratio:>8.2f}x")
    print("  ratio 1.00 = evidence spread as though the players were not there")
    print("  caveats: YOLO boxes are a proxy for 'where the players are' and are themselves")
    print("  imperfect; a 16x16 patch grid upsampled to frame size cannot resolve a distant")
    print(f"  player; and this is {len(play_rows)} frames, so read it as a direction, not a rate")

    print("\n=== redaction ===")
    print("  every frame: YOLO person pixelation, then a whole-frame blur floor")
    print("  'people detected: 0' means the detector found none, not that the frame is empty")

    # The per-frame numbers behind the table above. Written out because the table is a mean
    # over nine frames, and a mean over nine is the kind of number a reader should be able to
    # open up rather than take on trust.
    import csv as csv_module

    summary = settings.results_dir / "xai_evidence_focus.csv"
    with summary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv_module.DictWriter(fh, fieldnames=[
            "backbone", "class", "file", "n_people", "evidence_on_people", "person_area",
            "concentration",
        ])
        writer.writeheader()
        for entry in focus:
            writer.writerow(entry)

    print(f"\nwrote {len(written)} sheet(s) to "
          f"{OUT.relative_to(settings.results_dir.parent)} and {summary.name}")
    record(
        "WP4-T5 XAI overlays",
        "`python experiments/make_xai_figures.py`",
        f"`results/figs/xai/` ({len(written)} sheets)",
        f"exact linear decomposition, worst reconstruction error {worst:.1e}; "
        f"every frame person-pixelated and blurred before writing",
    )


if __name__ == "__main__":
    main()
