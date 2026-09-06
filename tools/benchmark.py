"""Model bake-off harness.

Evaluates candidate models on the labeled dataset (data/dataset/{1_empty,2_playing,
3_people_not_playing,4_maintenance}) and appends one row per model to
results/leaderboard.csv: accuracy, macro-F1, per-class precision/recall, CPU ms/frame.

Two model families:
  zero-shot   : CLIP / SigLIP / OpenCLIP prompted with class descriptions (no training)
  embed+head  : frozen backbone embeddings + logistic-regression head (trains in seconds)

Split strategies:
  --split random : stratified 80/20 (fast sanity check; frames are correlated!)
  --split slot   : train on one recorded slot day, test on the other (honest,
                   cross-lighting: day vs night)

Usage:
    python tools/benchmark.py --family zeroshot                  # all zero-shot models
    python tools/benchmark.py --family embed                     # all embedding models
    python tools/benchmark.py --models zs:openai/clip-vit-base-patch32
    python tools/benchmark.py --limit 200                        # quick pass on a subset
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset"
CONFIG = ROOT / "config" / "cameras.json"
RESULTS = ROOT / "results"

CLASSES = ["1_empty", "2_playing", "3_people_not_playing", "4_maintenance"]

# Multiple prompt templates per class; scores are averaged over templates.
PROMPTS = {
    "1_empty": [
        "an empty artificial turf football pitch with nobody on it",
        "a deserted five-a-side soccer field at night under floodlights, no people",
        "an empty green football field, no players",
    ],
    "2_playing": [
        "people playing football on a five-a-side pitch",
        "a soccer match in progress with players and a ball",
        "football training session with players doing drills on the pitch",
    ],
    "3_people_not_playing": [
        "a few people standing or walking on a football pitch, not playing",
        "people strolling casually across a soccer field without playing football",
        "people standing idle on an artificial turf field, no game happening",
    ],
    "4_maintenance": [
        "a maintenance worker in a high-visibility vest working on a football pitch",
        "groundskeeper with a broom or machine maintaining an artificial turf field",
        "workers doing turf maintenance on a soccer pitch with equipment",
    ],
}

ZEROSHOT_MODELS = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "google/siglip2-base-patch16-224",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
]

EMBED_MODELS = [
    "facebook/dinov2-base",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "google/vit-base-patch16-224",
    "facebook/convnextv2-tiny-22k-224",
]


# ---------------------------------------------------------------- data loading
def load_rois() -> dict:
    if CONFIG.exists():
        return {k: v.get("roi") for k, v in json.loads(CONFIG.read_text()).items()}
    return {}


def apply_roi(img, roi_norm):
    """Black out everything outside the ROI polygon. img is a PIL Image."""
    import cv2
    arr = np.array(img)
    h, w = arr.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in roi_norm], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    arr = cv2.bitwise_and(arr, arr, mask=mask)
    from PIL import Image
    return Image.fromarray(arr)


def collect_samples(limit: int | None):
    samples = []  # (path, label, cam_tag, slot_key)
    for cls in CLASSES:
        for p in sorted((DATASET / cls).rglob("*.jpg")):
            # camera tag is encoded in the filename: slot_<date>_<time>_cam<X>_t<secs>.jpg
            cam = p.stem.rsplit("_t", 1)[0] if "_cam" in p.stem else ""
            slot = cam.split("_cam")[0] if cam else "unknown"
            samples.append((p, cls, cam, slot))
    if limit:
        rng = np.random.default_rng(42)
        by_cls = defaultdict(list)
        for s in samples:
            by_cls[s[1]].append(s)
        per = max(1, limit // len(CLASSES))
        samples = []
        for cls, items in by_cls.items():
            idx = rng.permutation(len(items))[:per]
            samples += [items[i] for i in idx]
    return samples


def split_samples(samples, mode: str):
    labels = [s[1] for s in samples]
    if mode == "slot":
        slots = sorted({s[3] for s in samples})
        if len(slots) < 2:
            print("!! only one slot present, falling back to random split")
            mode = "random"
        else:
            test_slot = slots[-1]
            train = [s for s in samples if s[3] != test_slot]
            test = [s for s in samples if s[3] == test_slot]
            print(f"split=slot  train slots={slots[:-1]}  test slot={test_slot}")
            return train, test
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(samples, test_size=0.2, random_state=42, stratify=labels)
    print("split=random (80/20 stratified)")
    return train, test


def load_images(samples, rois):
    from PIL import Image
    imgs = []
    for p, cls, cam, _ in samples:
        img = Image.open(p).convert("RGB")
        if cam in rois and rois[cam]:
            img = apply_roi(img, rois[cam])
        img.thumbnail((640, 640))  # models resize to ~224 anyway; keeps RAM sane
        imgs.append(img)
    return imgs


# ---------------------------------------------------------------- evaluation
def metrics_row(y_true, y_pred, model_name, family, ms_per_frame, n_train, n_test, split):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0)
    row = {
        "model": model_name, "family": family, "split": split,
        "n_train": n_train, "n_test": n_test,
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "ms_per_frame": round(ms_per_frame, 1),
    }
    for i, cls in enumerate(CLASSES):
        row[f"P_{cls}"] = round(prec[i], 3)
        row[f"R_{cls}"] = round(rec[i], 3)
    return row


def append_leaderboard(row: dict):
    import pandas as pd
    RESULTS.mkdir(exist_ok=True)
    lb = RESULTS / "leaderboard.csv"
    df = pd.DataFrame([row])
    if lb.exists():
        df = pd.concat([pd.read_csv(lb), df], ignore_index=True)
    try:
        df.sort_values("macro_f1", ascending=False).to_csv(lb, index=False)
    except PermissionError:
        alt = RESULTS / "leaderboard_new.csv"
        df.sort_values("macro_f1", ascending=False).to_csv(alt, index=False)
        print(f"!! {lb.name} is locked (open in another program?) — wrote {alt.name} instead")
    print(f"\n=== leaderboard ({lb}) ===")
    cols = ["model", "family", "split", "accuracy", "macro_f1", "ms_per_frame"]
    print(df.sort_values("macro_f1", ascending=False)[cols].to_string(index=False))


def print_confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    short = [c.split("_", 1)[1][:12] for c in CLASSES]
    print("confusion (rows=true, cols=pred):")
    print(" " * 14 + "".join(f"{s:>14}" for s in short))
    for i, r in enumerate(cm):
        print(f"{short[i]:>14}" + "".join(f"{v:>14}" for v in r))


# ---------------------------------------------------------------- zero-shot
def run_zeroshot(model_name, test_samples, test_imgs, split):
    import torch
    from transformers import AutoModel, AutoProcessor

    print(f"\n--- zero-shot: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    texts = [t for cls in CLASSES for t in PROMPTS[cls]]
    n_templates = {cls: len(PROMPTS[cls]) for cls in CLASSES}
    is_siglip = "siglip" in model_name.lower()

    y_pred, times = [], []
    with torch.no_grad():
        pad = "max_length" if is_siglip else True  # SigLIP tokenizer requires max_length padding
        text_inputs = processor(text=texts, return_tensors="pt", padding=pad, truncation=True)
        for img in test_imgs:
            t0 = time.perf_counter()
            inputs = processor(images=img, return_tensors="pt")
            out = model(**inputs, **text_inputs)
            logits = out.logits_per_image[0]
            probs = torch.sigmoid(logits) if is_siglip else logits.softmax(-1)
            # average template scores per class
            scores, i = {}, 0
            for cls in CLASSES:
                k = n_templates[cls]
                scores[cls] = float(probs[i:i + k].mean())
                i += k
            y_pred.append(max(scores, key=scores.get))
            times.append((time.perf_counter() - t0) * 1000)

    y_true = [s[1] for s in test_samples]
    row = metrics_row(y_true, y_pred, model_name, "zeroshot",
                      float(np.mean(times)), 0, len(y_true), split)
    print_confusion(y_true, y_pred)
    append_leaderboard(row)


# ---------------------------------------------------------------- embed + head
def embed_images(model_name, imgs, batch_size=8):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    feats, times = [], []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i + batch_size]
            t0 = time.perf_counter()
            inputs = processor(images=batch, return_tensors="pt")
            out = model(**inputs)
            # NOTE: never use pooler_output for plain ViT — HF ships it randomly
            # initialized ("newly initialized" warning) and it destroys the features.
            if out.last_hidden_state.dim() == 3:
                emb = out.last_hidden_state.mean(dim=1)   # transformers: mean over tokens
            else:
                emb = out.last_hidden_state.mean(dim=(2, 3))  # convnets: global avg pool
            dt = (time.perf_counter() - t0) * 1000 / len(batch)
            times += [dt] * len(batch)
            feats.append(emb.cpu().numpy())
    return np.concatenate(feats), float(np.mean(times))


def run_embed(model_name, train_samples, train_imgs, test_samples, test_imgs, split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    print(f"\n--- embed+head: {model_name}")
    X_train, _ = embed_images(model_name, train_imgs)
    X_test, ms = embed_images(model_name, test_imgs)
    y_train = [s[1] for s in train_samples]
    y_true = [s[1] for s in test_samples]

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_train, y_train)
    y_pred = list(clf.predict(X_test))

    # persist the trained head first — it must survive a locked leaderboard file
    import pickle
    RESULTS.mkdir(exist_ok=True)
    safe = model_name.replace("/", "__")
    with (RESULTS / f"head_{safe}.pkl").open("wb") as f:
        pickle.dump({"model": model_name, "classes": CLASSES, "clf": clf,
                     "pooling": "mean"}, f)

    row = metrics_row(y_true, y_pred, model_name, "embed+head", ms,
                      len(y_train), len(y_true), split)
    print_confusion(y_true, y_pred)
    append_leaderboard(row)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["zeroshot", "embed", "all"], default="all")
    ap.add_argument("--models", nargs="*", default=None,
                    help="explicit list, prefixed zs: or emb: (e.g. zs:openai/clip-vit-base-patch32)")
    ap.add_argument("--split", choices=["random", "slot"], default="random")
    ap.add_argument("--limit", type=int, default=None, help="cap total samples for a quick pass")
    ap.add_argument("--no-roi", action="store_true", help="skip ROI masking")
    args = ap.parse_args()

    samples = collect_samples(args.limit)
    counts = defaultdict(int)
    for s in samples:
        counts[s[1]] += 1
    print(f"dataset: {len(samples)} labeled frames  {dict(counts)}")
    if not samples:
        raise SystemExit("No labeled frames. Run tools/label_tool.py first.")

    train, test = split_samples(samples, args.split)
    rois = {} if args.no_roi else load_rois()
    if rois:
        print(f"ROI masks loaded for: {list(rois)}")
    print("loading images...")
    train_imgs = load_images(train, rois)
    test_imgs = load_images(test, rois)

    if args.models:
        zs = [m[3:] for m in args.models if m.startswith("zs:")]
        emb = [m[4:] for m in args.models if m.startswith("emb:")]
    else:
        zs = ZEROSHOT_MODELS if args.family in ("zeroshot", "all") else []
        emb = EMBED_MODELS if args.family in ("embed", "all") else []

    for m in zs:
        try:
            run_zeroshot(m, test, test_imgs, args.split)
        except Exception as e:
            print(f"!! {m} failed: {e}")
    for m in emb:
        try:
            run_embed(m, train, train_imgs, test, test_imgs, args.split)
        except Exception as e:
            print(f"!! {m} failed: {e}")


if __name__ == "__main__":
    main()
