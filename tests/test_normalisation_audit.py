"""Photometric and geometric normalisation audit (WP3-T3).

The question the audit was meant to answer - are each model's normalisation statistics
applied? - turned out to be the easy half. They are: the processor is called directly, so
each model gets its own mean and standard deviation.

The half nobody had checked is what the processor does to the *geometry* of a frame that
`preprocess.py` has already resized. For ViT, nothing. For DINOv2 and
ConvNeXtV2 - the default model and the one the 740-minute preprocessing search ran on - it
upscales to 256 and centre-crops back to 224, discarding 23.4% of the frame including the
letterbox padding WP3-T2 exists to add.

ConvNeXtV2 is the one that hid: its `size` reads `shortest_edge: 224` and looks like a
no-op, and only `crop_pct: 0.875` reveals that it resizes to 256 first. Reading `size` alone
gives the wrong answer, which is exactly how an audit misses something.

These tests pin that finding so it cannot quietly regress or be forgotten.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pitch_occupancy.vision.backbones import (
    BACKBONES, GEOMETRY_IS_A_NO_OP, PROCESSOR_GEOMETRY,
)

pytestmark = pytest.mark.slow  # every test here downloads and runs a real backbone


@pytest.fixture(scope="module")
def frames() -> list[Image.Image]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(2):
        a = np.zeros((224, 224, 3), np.uint8)
        a[:, :, 1] = 130
        a[:24] = a[-24:] = 114  # the grey letterbox bars
        a[100:130, 100:130] = (200, 40, 40)
        out.append(Image.fromarray(
            np.clip(a.astype(int) + rng.normal(0, 3, a.shape), 0, 255).astype(np.uint8)
        ))
    return out


def processor_for(key: str):
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(BACKBONES[key].hf_id)


# --- the photometric half: this part was already right ----------------------


@pytest.mark.parametrize("key", sorted(BACKBONES))
def test_each_model_normalises_with_its_own_statistics(key) -> None:
    """ViT is trained on 0.5/0.5 and the other two on ImageNet statistics. Sharing one set
    across models would quietly mis-centre two of the three."""
    d = processor_for(key).to_dict()
    assert d["do_normalize"] and d["do_rescale"]
    assert len(d["image_mean"]) == 3 and len(d["image_std"]) == 3


def test_vit_and_the_imagenet_models_do_not_share_statistics() -> None:
    vit = processor_for("vit").to_dict()["image_mean"]
    dino = processor_for("dinov2").to_dict()["image_mean"]
    assert tuple(vit) != tuple(dino), "if these ever match, one model is being mis-normalised"


# --- the geometric half: this part was not -----------------------------------


def test_every_backbone_has_its_geometry_documented() -> None:
    assert set(PROCESSOR_GEOMETRY) == set(BACKBONES)


def test_vit_is_the_only_processor_that_leaves_a_224_frame_alone(frames) -> None:
    px = processor_for("vit")(images=frames, return_tensors="pt")["pixel_values"]
    assert px.shape[-2:] == (224, 224)
    assert not processor_for("vit").to_dict().get("do_center_crop", False)
    assert GEOMETRY_IS_A_NO_OP == ("vit",)


def test_dinov2_crops_a_quarter_of_the_frame_away() -> None:
    """It is the default model, so this applies to the primary configuration."""
    d = processor_for("dinov2").to_dict()
    assert d["do_center_crop"] is True
    assert d["size"]["shortest_edge"] == 256
    assert d["crop_size"]["height"] == 224
    assert 0.76 < (224 / 256) ** 2 < 0.77, "the documented 23.4% loss no longer matches"


def test_convnextv2_crops_the_same_amount_through_crop_pct() -> None:
    """The one that hid. `size` says shortest_edge 224 and reads as a no-op; `crop_pct`
    is what makes it resize to 256 and crop back. An audit that read `size` alone - as the
    first pass of this one did - would have cleared it."""
    d = processor_for("convnextv2").to_dict()
    assert d["size"]["shortest_edge"] == 224, "looks like a no-op..."
    assert d["crop_pct"] == 0.875, "...but this is what makes it one"
    assert round(224 / d["crop_pct"]) == 256


def test_turning_the_geometry_off_changes_the_embedding_for_dinov2(frames) -> None:
    """If these ever agree, the double crop stopped happening and the note is stale."""
    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    model, processor, spec = load_backbone("dinov2")
    a = embed_batch(model, processor, spec, frames)
    b = embed_batch(model, processor, spec, frames, processor_geometry=False)
    cos = float((a[0] @ b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0])))
    assert cos < 0.999, "the processor's resize and crop are no longer doing anything"


def test_turning_the_geometry_off_changes_convnextv2_even_more(frames) -> None:
    """0.886 against DINOv2's 0.961: the model the 740-minute search ran on is the one most
    affected. That also explains why `centre_crop=0.5` scored so badly in that search - it
    was a crop applied on top of a crop."""
    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    model, processor, spec = load_backbone("convnextv2")
    a = embed_batch(model, processor, spec, frames)
    b = embed_batch(model, processor, spec, frames, processor_geometry=False)
    cos = float((a[0] @ b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0])))
    assert cos < 0.99


def test_turning_the_geometry_off_is_a_true_no_op_for_vit(frames) -> None:
    """The control. ViT sees exactly what preprocessing produced, either way."""
    from pitch_occupancy.vision.backbones import embed_batch, load_backbone

    model, processor, spec = load_backbone("vit")
    a = embed_batch(model, processor, spec, frames)
    b = embed_batch(model, processor, spec, frames, processor_geometry=False)
    assert np.allclose(a, b, atol=1e-5)
