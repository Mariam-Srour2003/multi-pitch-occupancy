"""Search the prompt space for the best zero-shot classifier (RQ1, cold start).

The question this answers is the one a trained probe cannot: **how well does the system
work at a new venue before anyone has labelled anything there?** That is the actual
onboarding cost of a client site, and the pilot reported 75.8% for OpenCLIP from a single
hand-written prompt per class - a figure that measures the prompt as much as the model.

Unlike the preprocessing search, this one can be **exhaustive**. Images are embedded once;
a prompt set is a handful of short strings, so the whole product of descriptors is scored
in seconds. 5 x 5 x 5 = 125 descriptor combinations, each under several template
ensembles.

Scored the same way as everything else here - cross-venue play recall with the false-play
control beside it - so a prompt set that wins by calling everything "playing" is visible
rather than crowned. **This matters more for prompts than for probes**: a prompt like
"a football pitch" with no mention of people would push every frame toward ACTIVE_PLAY and
score a perfect recall on a test set that is 100% ACTIVE_PLAY.

    uv run python experiments/prompt_search.py --limit 600
    uv run python experiments/prompt_search.py --templates 1 3 5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows, leave_one_group_out
from pitch_occupancy.data.taxonomy import CLASS3_ORDER, Class3
from pitch_occupancy.evaluation.experiment_log import record
from pitch_occupancy.vision.zeroshot import (
    DESCRIPTORS,
    TEMPLATES,
    PromptSet,
    classify,
    encode_prompts,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
CLIP_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SEED = 42
PLAY = Class3.ACTIVE_PLAY.value
EMPTY = Class3.EMPTY.value


def load_clip():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_ID)
    model.eval()
    return model, CLIPProcessor.from_pretrained(CLIP_ID)


def _projected(output):
    """Pull the projected embedding out of a transformers 5.x CLIP feature call.

    `get_image_features` / `get_text_features` return a `BaseModelOutputWithPooling`
    rather than a tensor, and the embedding is its `pooler_output`.

    **This is not the trap that cost the pilot a false 38%.** There, `pooler_output` on a
    plain ViT was a randomly initialised head that had never been trained. Here it is
    CLIP's learned projection: its width is `projection_dim` (512) rather than the vision
    hidden size (768), and it was checked functionally before use - a real ACTIVE_PLAY
    frame scores 0.296 against "people playing football", 0.062 against "a plate of
    spaghetti" and -0.011 against "a cat on a sofa". Random features would not order that
    way.
    """
    return output.pooler_output


@torch.inference_mode()
def image_features(rows) -> np.ndarray:
    """Embed every frame once with CLIP, cached. This is the only expensive step."""
    path = CACHE / "clip_image_features.npz"
    files = [r.file for r in rows]
    if path.exists():
        z = np.load(path, allow_pickle=True)
        idx = {str(f): i for i, f in enumerate(z["files"])}
        if all(f in idx for f in files):
            return z["features"][[idx[f] for f in files]]

    model, processor = load_clip()
    chunks = []
    for start in range(0, len(files), 32):
        batch = [
            Image.open(DATASET / f).convert("RGB") for f in files[start : start + 32]
        ]
        inputs = processor(images=batch, return_tensors="pt")
        feats = _projected(model.get_image_features(**inputs)).cpu().numpy()
        chunks.append(feats)
        print(f"  embedding {min(start + 32, len(files))}/{len(files)}", end="\r", flush=True)
    print()
    feats = np.concatenate(chunks).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, files=np.array(files, dtype=object), features=feats)
    return feats


def make_text_encoder():
    model, processor = load_clip()

    @torch.inference_mode()
    def encode(phrases: list[str]) -> np.ndarray:
        inputs = processor(text=phrases, return_tensors="pt", padding=True, truncation=True)
        return _projected(model.get_text_features(**inputs)).cpu().numpy()

    return encode


def score(rows, X: np.ndarray, directions: np.ndarray, classes, folds) -> dict:
    """Cross-venue play recall plus the false-play control.

    Zero-shot has no training step, so a fold is only a partition of the evaluation - each
    held-out venue is scored with the same prompt set.
    """
    pos = {r.file: i for i, r in enumerate(rows)}
    recalls = []
    for fold in folds:
        te = [pos[r.file] for r in fold.test]
        pred = classify(X[te], directions, classes)
        truth = [r.class3 for r in fold.test]
        play = [(p, t) for p, t in zip(pred, truth, strict=True) if t == PLAY]
        recalls.append(sum(p == PLAY for p, _ in play) / len(play) if play else float("nan"))

    empties = [r for r in rows if r.class3 == EMPTY]
    false_play = 0.0
    if empties:
        pe = classify(X[[pos[r.file] for r in empties]], directions, classes)
        false_play = float(np.mean([p == PLAY for p in pe]))

    arr = np.array(recalls, dtype=float)
    return {
        "play_recall": float(arr.mean()),
        "worst_fold": float(arr.min()),
        "false_play": false_play,
        # the honest single number: recall is free if you always say PLAY
        "balanced": float(arr.mean()) - false_play,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Exhaustive zero-shot prompt search.")
    ap.add_argument("--limit", type=int, default=None, help="subsample frames")
    ap.add_argument("--templates", type=int, nargs="*", default=[1, 3, 5],
                    help="template-ensemble sizes to try")
    ap.add_argument("--top", type=int, default=15, help="how many rows to print")
    args = ap.parse_args()

    rows = development_rows(read_manifest(DATASET / "manifest.csv"))
    if args.limit and args.limit < len(rows):
        rng = np.random.default_rng(SEED)
        keep = set(rng.choice(len(rows), args.limit, replace=False).tolist())
        rows = [r for i, r in enumerate(rows) if i in keep]
    folds = [
        f for f in leave_one_group_out(rows, group_key="venue")
        if not f.name.endswith("venue_01")
    ]
    classes = list(CLASS3_ORDER)

    combos = list(itertools.product(*(DESCRIPTORS[c] for c in classes)))
    total = len(combos) * len(args.templates)
    print(f"{len(rows)} frames | {len(folds)} venue folds")
    print(f"{len(combos)} descriptor combinations x {len(args.templates)} ensemble sizes "
          f"= {total} prompt sets\n")

    X = image_features(rows)
    encode = make_text_encoder()
    text_cache: dict[str, np.ndarray] = {}

    def encode_cached(phrases: list[str]) -> np.ndarray:
        missing = [p for p in phrases if p not in text_cache]
        if missing:
            for phrase, vec in zip(missing, encode(missing), strict=True):
                text_cache[phrase] = vec
        return np.stack([text_cache[p] for p in phrases])

    records = []
    for n_templates in args.templates:
        templates = tuple(TEMPLATES[:n_templates])
        for combo in combos:
            ps = PromptSet(dict(zip(classes, combo, strict=True)), templates)
            directions = encode_prompts(ps, classes, encode_cached)
            metrics = score(rows, X, directions, classes, folds)
            records.append({
                "n_templates": n_templates,
                **{f"desc_{c.name}": ps.descriptors[c] for c in classes},
                **{k: round(v, 4) for k, v in metrics.items()},
            })
        print(f"  {n_templates} template(s): {len(combos)} sets scored")

    records.sort(key=lambda r: -r["balanced"])
    print(f"\n{'':<3}{'recall':>8}{'falsePlay':>11}{'balanced':>10}{'tmpl':>6}  descriptors")
    for i, r in enumerate(records[: args.top], 1):
        print(f"{i:<3}{r['play_recall']:>8.4f}{r['false_play']:>11.4f}"
              f"{r['balanced']:>10.4f}{r['n_templates']:>6}  "
              f"{r['desc_ACTIVE_PLAY'][:44]}")

    best = records[0]
    naive = max(records, key=lambda r: r["play_recall"])
    print(f"\nbest by balanced score : recall {best['play_recall']:.4f}  "
          f"falsePlay {best['false_play']:.4f}")
    print(f"best by recall alone   : recall {naive['play_recall']:.4f}  "
          f"falsePlay {naive['false_play']:.4f}")
    if naive["false_play"] > best["false_play"] + 0.01:
        print("  -> ranking on recall alone would have picked a prompt that simply says "
              "PLAY more often")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "prompt_search.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    (RESULTS / "prompt_search_best.json").write_text(
        json.dumps({"best": best, "best_by_recall": naive, "n_evaluated": len(records)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")

    record(
        "zero-shot prompt search",
        "`python experiments/prompt_search.py`",
        f"`{out.name}`",
        f"{len(records)} prompt sets",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
