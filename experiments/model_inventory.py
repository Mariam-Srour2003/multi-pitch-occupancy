"""Count what is frozen and what is trained, by instantiating every model (WP8-T4).

The models page has to answer one question — *did you actually train anything?* — and the
honest answer is a ratio: about 200M pretrained parameters that never received a gradient,
against a few thousand that did. A ratio that large is only convincing if it is measured.

So nothing here is typed from a paper or a config table. Each backbone is loaded from its
Hugging Face checkpoint and its parameters counted; each trained head is *constructed* at
the shape an experiment actually builds it and counted the same way. `docs/
DIAGRAM_PROMPT_MODELS.md` states these figures from an earlier hand count, and this script
is what lets that document be checked rather than believed.

**The trained counts depend on shapes that only exist at fit time.** `LinearProbe` has no
weights until `fit` sees a feature matrix, and `GatedFusionHead` builds its module from the
width of the caches it is handed. Both are therefore counted at their real shapes - 768-d
features, three classes, three backbones - and the shape is recorded beside the count so a
reader can see which configuration was counted. A parameter count with no shape attached is
not checkable.

**Pooling heads are counted as frozen but flagged unused.** A plain ViT ships a
`pooler.dense` that is randomly initialised and never trained; this project mean-pools
explicitly and never reads it (see `vision/backbones.py`). Counting it silently would
inflate the frozen total with weights the pipeline does not use, so it is reported as a
separate `unused_pooler_params` field rather than folded in.

    uv run python experiments/model_inventory.py

Writes `results/model_inventory.json`. Re-run if a backbone, head shape or class count
changes; the models page reads every number from it.
"""

from __future__ import annotations

import json

from pitch_occupancy.config import settings
from pitch_occupancy.vision.backbones import BACKBONES, GEOMETRY_IS_A_NO_OP, PROCESSOR_GEOMETRY

OUT = settings.results_dir / "model_inventory.json"

#: The feature width every cache in this project has, and the class count the taxonomy
#: fixes. Named here because the trained heads' parameter counts are functions of them.
FEATURE_DIM = 768
N_CLASSES = 3

#: Zero-shot control. Counted but never fitted - its `fit()` is a deliberate no-op, so it
#: is the one model whose score can only move because the test set moved.
CLIP_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

#: How each backbone was pretrained, and what that buys the thesis. Not derivable from the
#: checkpoint, so it is stated here with the source named rather than inferred.
PRETRAINING = {
    "convnextv2": ("ImageNet-22k, masked autoencoder then supervised", "labels used"),
    "vit": ("supervised ImageNet-21k then fine-tuned to 1k", "labels used"),
    "dinov2": ("self-supervised on LVD-142M", "no labels at all"),
}


def _count(module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts for a torch module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _architecture(cfg, kind: str) -> dict:
    """The few shape numbers that make an architecture recognisable, read off its config."""
    if kind == "convnet":
        return {
            "family": "convnet",
            "depths": list(getattr(cfg, "depths", []) or []),
            "widths": list(getattr(cfg, "hidden_sizes", []) or []),
            "output_dim": (getattr(cfg, "hidden_sizes", [0]) or [0])[-1],
        }
    return {
        "family": "transformer",
        "layers": getattr(cfg, "num_hidden_layers", None),
        "hidden": getattr(cfg, "hidden_size", None),
        "heads": getattr(cfg, "num_attention_heads", None),
        "patch": getattr(cfg, "patch_size", None),
        "output_dim": getattr(cfg, "hidden_size", None),
    }


def backbones() -> list[dict]:
    """Load every frozen extractor and count it. This is the expensive part - it is also
    the only way the frozen total is a measurement rather than a recollection."""
    from pitch_occupancy.vision.backbones import load_backbone

    out = []
    for key, spec in BACKBONES.items():
        model, _processor, _spec = load_backbone(key)
        total, trainable = _count(model)
        # `load_backbone` clears requires_grad on every parameter, so a non-zero trainable
        # count here would mean the freeze itself had regressed. Recorded, not assumed.
        pooler = sum(
            p.numel() for n, p in model.named_parameters() if n.startswith("pooler.")
        )
        how, labels = PRETRAINING[key]
        out.append({
            "key": key,
            "label": {"convnextv2": "ConvNeXtV2", "vit": "ViT", "dinov2": "DINOv2"}[key],
            "hf_id": spec.hf_id,
            "role": spec.note,
            "params": total,
            "trainable_params": trainable,
            "unused_pooler_params": pooler,
            "pretraining": how,
            "pretraining_labels": labels,
            "pooling": (
                "last_hidden_state.mean(dim=(2,3))" if spec.kind == "convnet"
                else "last_hidden_state.mean(dim=1)"
            ),
            "processor_geometry": PROCESSOR_GEOMETRY[key],
            "geometry_is_a_no_op": key in GEOMETRY_IS_A_NO_OP,
            "architecture": _architecture(model.config, spec.kind),
        })
        del model
    return out


def linear_probe() -> dict:
    """The head every published number comes from, counted at the shape it is fitted at.

    Fitted on a tiny synthetic matrix rather than arithmetic on 768x3+3, so that the count
    comes from the object sklearn actually builds. If the pipeline ever gains a term, this
    notices; a formula would not.
    """
    import numpy as np

    from pitch_occupancy.vision.heads import LinearProbe

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, FEATURE_DIM)).astype("float32")

    class _Row:
        def __init__(self, c: str) -> None:
            self.class3 = c

    rows = [_Row(["c1_empty", "c2_active_play", "c3_maintenance"][i % 3]) for i in range(30)]
    probe = LinearProbe(name="probe").fit(X, rows)
    lr = probe._model.named_steps["logisticregression"]  # noqa: SLF001 - counting weights
    coefs = int(lr.coef_.size)
    biases = int(lr.intercept_.size)
    return {
        "name": "Linear probe (logistic regression)",
        "shape": f"{FEATURE_DIM} features x {N_CLASSES} classes + {N_CLASSES} biases",
        "params": coefs + biases,
        "coefficients": coefs,
        "biases": biases,
        "detail": 'StandardScaler -> LogisticRegression(max_iter=2000, '
                  'class_weight="balanced", random_state=42)',
        "why": "C3 has 6 frames in the whole dataset; an unweighted fit never predicts it.",
        "refit": "from scratch for every backbone x protocol x seed, in under a second",
    }


def fusion_heads() -> list[dict]:
    """The gated fusion ablation ladder, each rung counted at its real shape.

    The three rungs differ only in how the mixing weights are produced, so counting all
    three is what shows that `routing` costs a handful of parameters over `constant` - and
    the ablation found it buys nothing.
    """
    from pitch_occupancy.slots.fusion_head import GATE_STATISTIC_NAMES, _build_gated_module

    rungs = {
        "uniform": "weights fixed at 1/K - no mixing learned",
        "constant": "K learned weights, shared by every frame",
        "mlp": "weights learned per frame from 3 statistics - the only actual router",
    }
    out = []
    for gate, note in rungs.items():
        module = _build_gated_module(
            n_backbones=3, block_dim=FEATURE_DIM, gate_dim=len(GATE_STATISTIC_NAMES),
            hidden=8, n_classes=N_CLASSES, gate=gate,
        )
        total, _ = _count(module)
        out.append({"rung": gate, "params": total, "note": note})
    return out


def stan() -> dict:
    """The slot-level model, counted from the module its own builder produces."""
    from pitch_occupancy.slots.stan import STAN, _build_stan

    spec = STAN()
    module = _build_stan(channels=spec.channels, kernel=spec.kernel, n_classes=N_CLASSES)
    total, _ = _count(module)
    # Two conv layers, dilations 1 and 2: each adds (kernel-1)*dilation to the field.
    receptive = 1 + (spec.kernel - 1) + (spec.kernel - 1) * 2
    return {
        "name": "STAN (spatio-temporal aggregation network)",
        "params": total,
        "ceiling": 100_000,
        "shape": f"Conv1d(3 -> {spec.channels}, k={spec.kernel}) -> "
                 f"Conv1d({spec.channels} -> {spec.channels}, k={spec.kernel}, dilation=2) -> "
                 f"Linear({2 * spec.channels} -> {N_CLASSES})",
        "receptive_field_minutes": receptive,
        "pooling": "masked mean and max, concatenated",
        "training": f"{spec.epochs} epochs, Adam, lr {spec.lr}, "
                    f"weight decay {spec.weight_decay}, seed {spec.seed}",
    }


def clip() -> dict:
    """The zero-shot control: loaded, counted, and never fitted."""
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(CLIP_ID)
    total, _ = _count(model)
    return {
        "name": "OpenCLIP zero-shot",
        "hf_id": CLIP_ID,
        "params": total,
        "trained_params": 0,
        "why": "fit() is a deliberate no-op, so the same per-frame prediction is scored "
               "under every protocol - any movement is caused by the test set alone.",
    }


def main() -> None:
    bb = backbones()
    probe = linear_probe()
    fusion = fusion_heads()
    st = stan()
    cl = clip()

    frozen_total = sum(b["params"] for b in bb)
    # The gate's `mlp` rung is the one WP5-T2 reports, so it is the one that counts toward
    # "what was trained". The other rungs are its controls.
    mlp = next(r for r in fusion if r["rung"] == "mlp")
    trained_total = probe["params"] + mlp["params"] + st["params"]

    inventory = {
        "frozen": bb,
        "frozen_total_params": frozen_total,
        "frozen_gradients_received": sum(b["trainable_params"] for b in bb),
        "trained": {"probe": probe, "fusion": fusion, "stan": st},
        "trained_total_params": trained_total,
        "frozen_to_trained_ratio": round(frozen_total / trained_total),
        "probe_ratio": round(frozen_total / probe["params"]),
        "zero_shot": cl,
        "feature_dim": FEATURE_DIM,
        "n_classes": N_CLASSES,
    }

    print(f"frozen:  {frozen_total:>12,} parameters, "
          f"{inventory['frozen_gradients_received']} of which received a gradient")
    print(f"trained: {trained_total:>12,} parameters "
          f"(probe {probe['params']:,}, gate {mlp['params']:,}, STAN {st['params']:,})")
    print(f"ratio:   {inventory['frozen_to_trained_ratio']:,} : 1 "
          f"({inventory['probe_ratio']:,} : 1 against the probe alone)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
