"""Tier-1 frame classifier with a pluggable model registry.

Any model from config/system_config.json MODELS can be loaded by key:
  - embed_head: frozen HF backbone -> features -> trained logistic-regression head
                (heads produced by tools/benchmark.py, stored in results/)
  - zeroshot:   CLIP-style model prompted with the class descriptions

Usage:
    clf = Classifier.from_config("convnextv2")
    label, conf, probs = clf.predict(pil_image)
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "system_config.json").read_text())

CLASSES = ["1_empty", "2_playing", "3_people_not_playing", "4_maintenance"]

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


class Classifier:
    def __init__(self, key: str, spec: dict):
        self.key = key
        self.type = spec["type"]
        self.backbone_name = spec["backbone"]

        if self.type == "embed_head":
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(self.backbone_name)
            self.model = AutoModel.from_pretrained(self.backbone_name)
            self.model.eval()
            with (ROOT / spec["head"]).open("rb") as f:
                bundle = pickle.load(f)
            assert bundle["model"] == self.backbone_name, "head/backbone mismatch"
            # heads must be trained on the same feature extraction this engine uses
            assert bundle.get("pooling") == "mean", (
                f"head {spec['head']} was trained with pooling="
                f"{bundle.get('pooling', 'pooler_output(legacy)')} but the engine "
                "extracts mean-pooled features — retrain via tools/benchmark.py")
            self.head = bundle["clf"]
            self.head_classes = list(bundle["classes"])
        elif self.type == "zeroshot":
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.backbone_name)
            self.model = AutoModel.from_pretrained(self.backbone_name)
            self.model.eval()
            texts = [t for cls in CLASSES for t in PROMPTS[cls]]
            self._text_inputs = self.processor(
                text=texts, return_tensors="pt", padding=True, truncation=True)
            self._n_templates = [len(PROMPTS[c]) for c in CLASSES]
        else:
            raise ValueError(f"unknown model type {self.type}")

    @classmethod
    def from_config(cls, key: str | None = None) -> "Classifier":
        key = key or CONFIG["DEFAULT_MODEL"]
        return cls(key, CONFIG["MODELS"][key])

    def _embed(self, imgs: list[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(images=imgs, return_tensors="pt")
            out = self.model(**inputs)
            # mean-pool: tokens for transformers, spatial map for convnets.
            # Never pooler_output (randomly initialized on plain ViT).
            if out.last_hidden_state.dim() == 3:
                emb = out.last_hidden_state.mean(dim=1)
            else:
                emb = out.last_hidden_state.mean(dim=(2, 3))
        return emb.cpu().numpy()

    def predict_batch(self, imgs: list[Image.Image]) -> list[tuple[str, float, dict]]:
        """Returns [(label, confidence, {class: prob})] per image."""
        imgs = [im.convert("RGB") for im in imgs]
        for im in imgs:
            im.thumbnail((640, 640))

        if self.type == "embed_head":
            X = self._embed(imgs)
            proba = self.head.predict_proba(X)
            results = []
            for row in proba:
                probs = dict.fromkeys(CLASSES, 0.0)
                probs.update(zip(self.head_classes, (float(v) for v in row)))
                label = max(probs, key=probs.get)
                results.append((label, probs[label], probs))
            return results

        results = []
        with torch.no_grad():
            for im in imgs:
                inputs = self.processor(images=im, return_tensors="pt")
                out = self.model(**inputs, **self._text_inputs)
                p = out.logits_per_image[0].softmax(-1)
                probs, i = {}, 0
                for cls, k in zip(CLASSES, self._n_templates):
                    probs[cls] = float(p[i:i + k].mean())
                    i += k
                s = sum(probs.values())
                probs = {c: v / s for c, v in probs.items()}
                label = max(probs, key=probs.get)
                results.append((label, probs[label], probs))
        return results

    def predict(self, img: Image.Image) -> tuple[str, float, dict]:
        return self.predict_batch([img])[0]
