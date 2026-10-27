"""Release golden (MOD-REL): the shipped data recipes produce the same training pairs.

TC ids follow docs/modules/20261015-MOD-ReleaseGolden.md section 4. Each recipe runs on one synthetic
image with random and np.random seeded per sample, both as shipped and with every enabled op forced on
(prob 1, clean_prob 0) so that rarely drawn ops are covered too. JPEG and resize bits depend on the
OpenCV and NumPy builds, so the case is skipped on other versions.
"""

import copy
import hashlib
import os
import random

import cv2
import numpy as np
import pytest
import yaml

from src.data.datasets import DenoiseDataset, SRDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDED_WITH = {"cv2": "4.12.0", "numpy": "2.2.6"}
SAMPLES = 24

# (data config under configs/data/bicubicpp, variant, scale): sha256[:16] of 24 seeded lr/hr pairs
DIGESTS = {
    ("sr_train_phase-1", "shipped", 2): "6be70acb5615d837",
    ("sr_train_phase-1", "shipped", 3): "ff9093633a864253",
    ("sr_train_phase-1", "shipped", 4): "78bee96321434420",
    ("sr_train_phase-1", "forced", 2): "6be70acb5615d837",
    ("sr_train_phase-1", "forced", 3): "ff9093633a864253",
    ("sr_train_phase-1", "forced", 4): "78bee96321434420",
    ("sr_finetune_eo_phase-2", "shipped", 2): "40c1125c5a21bdc6",
    ("sr_finetune_eo_phase-2", "shipped", 3): "765cbbbf25440c12",
    ("sr_finetune_eo_phase-2", "shipped", 4): "40ba0ac0cd2c6cef",
    ("sr_finetune_eo_phase-2", "forced", 2): "4c94f4149c74bb3b",
    ("sr_finetune_eo_phase-2", "forced", 3): "e4ee1d939a264aa7",
    ("sr_finetune_eo_phase-2", "forced", 4): "593995d13d1764d5",
    ("sr_finetune_ir_phase-2", "shipped", 2): "ca977a7ece7dd913",
    ("sr_finetune_ir_phase-2", "shipped", 3): "9207ba04c0b45b17",
    ("sr_finetune_ir_phase-2", "shipped", 4): "bd52c8350237fcce",
    ("sr_finetune_ir_phase-2", "forced", 2): "a76d8abd97df1bef",
    ("sr_finetune_ir_phase-2", "forced", 3): "61e0cf696e23fc10",
    ("sr_finetune_ir_phase-2", "forced", 4): "d15c2c46e4cf8b4e",
    ("denoise", "shipped", 1): "2adad2af7ddfdb2d",
    ("denoise", "forced", 1): "8a005c79d4a3252e",
}


def scene(h, w, seed=0):
    """Gradients, two flat patches and mild noise: every op has edges and flats to act on."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([40 + 150 * xx / w, 60 + 120 * yy / h, 90 + 60 * (xx + yy) / (h + w)], axis=-1)
    img[h // 4 : h // 2, w // 4 : w // 2] = (200, 190, 180)
    img[h // 2 : 3 * h // 4, w // 2 : 3 * w // 4] = (30, 40, 50)
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def forced(cfg):
    """Every enabled op (and sub-op) fires on every sample; no clean pairs."""
    cfg = copy.deepcopy(cfg)
    cfg["clean_prob"] = 0.0
    for stage in (cfg.get("degradation") or {}).values():
        for op in stage.values():
            if isinstance(op, dict) and op.get("enabled", False):
                if "prob" in op:
                    op["prob"] = 1.0
                for sub in op.values():
                    if isinstance(sub, dict) and sub.get("enabled", False):
                        sub["prob"] = 1.0
    return cfg


@pytest.fixture(scope="module")
def image_root(tmp_path_factory):
    if {"cv2": cv2.__version__, "numpy": np.__version__} != RECORDED_WITH:
        pytest.skip(f"digests were recorded with {RECORDED_WITH}")
    root = tmp_path_factory.mktemp("recipe")
    cv2.imwrite(str(root / "img.png"), scene(400, 400))
    return str(root)


# ---------- MREQ-REL-04: data recipes ----------
@pytest.mark.parametrize("name,variant,scale", list(DIGESTS))
def test_TC_REL_04_1_DataRecipeUnchanged(name, variant, scale, image_root):
    with open(f"{REPO}/configs/data/bicubicpp/{name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if variant == "forced":
        cfg = forced(cfg)
    if name == "denoise":
        ds = DenoiseDataset(image_root, patch_size=128, config=cfg)
    else:
        ds = SRDataset(image_root, scale_factor=scale, patch_size=96, config=cfg)
    h = hashlib.sha256()
    for i in range(SAMPLES):
        random.seed(i)
        np.random.seed(i)
        s = ds[0]
        h.update(s["lr"].numpy().tobytes())
        h.update(s["hr"].numpy().tobytes())
    assert h.hexdigest()[:16] == DIGESTS[(name, variant, scale)]
