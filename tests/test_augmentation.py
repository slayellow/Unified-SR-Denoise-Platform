"""Geometric augmentation, and the guard that stops the target being upscaled.

Two defects this pins, both found by auditing the retired ADDITIONAL_PLAN.md
(deleted in 528d7cd; `git show fd86960:ADDITIONAL_PLAN.md`) against the code
rather than by the plan itself:

  * `use_flip` / `use_rot` sat in every data config from the start and nothing
    read them, so all six deployed models trained without the 8x expansion the
    dihedral group hands a 22.7k-parameter network for free.
  * an image smaller than the HR crop was bicubic-upscaled to fit, which makes
    the TARGET a bicubic upsample. At the old scale-blind patch_size of 288 that
    hit 93% of LSDIR samples at x4 -- the run would have taught the model to
    reproduce the very softness the project is trying to remove.
"""
import numpy as np
import pytest

from src.data.datasets import augment_pair


def _pair(seed=0):
    """LR built by block average, which is exactly equivariant to flips and
    rotations. Subsampling with hr[::2, ::2] is not: it keeps each 2x2 block's
    top-left pixel, and a flip turns that into the bottom-right, so the pair
    would appear misregistered by one sample even when the code is correct."""
    import cv2
    g = np.random.default_rng(seed)
    hr = g.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    lr = cv2.resize(hr, (32, 32), interpolation=cv2.INTER_AREA)
    return hr, lr


def test_disabled_is_identity():
    hr, lr = _pair()
    a, b = augment_pair(hr, lr, use_flip=False, use_rot=False)
    assert np.array_equal(a, hr) and np.array_equal(b, lr)


def test_pair_stays_registered():
    """The same draw must drive both images: if the HR flips and the LR does
    not, the model is handed contradictory targets and learns to blur."""
    import random
    for seed in range(40):
        random.seed(seed)
        hr, lr = _pair(seed)
        a, b = augment_pair(hr, lr, use_flip=True, use_rot=True)
        import cv2
        assert np.array_equal(cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA), b), \
            f"seed {seed}: 쌍이 어긋남 -- HR 과 LR 에 다른 변환이 걸렸다"


def test_every_dihedral_element_is_reachable():
    """Flip and rot90 together generate all 8 orientations."""
    import random
    hr, lr = _pair(1)
    seen = set()
    for seed in range(400):
        random.seed(seed)
        a, _ = augment_pair(hr, lr, use_flip=True, use_rot=True)
        seen.add(a.tobytes())
    assert len(seen) == 8, f"{len(seen)}개만 나옴 (8개여야)"


def test_output_is_contiguous():
    """Negative-stride views from ::-1 break torch.from_numpy."""
    import random
    import torch
    random.seed(3)
    hr, lr = _pair()
    a, b = augment_pair(hr, lr, use_flip=True, use_rot=True)
    torch.from_numpy(a)      # raises on a negative stride
    torch.from_numpy(b)


@pytest.mark.parametrize("scale,patch,hr_crop", [(2, 288, 576), (3, 192, 576), (4, 144, 576)])
def test_phase1_configs_crop_the_same_hr_area(scale, patch, hr_crop):
    """patch_size is the LR size, so a scale-blind value makes the HR crop grow
    with the scale -- 1152px at x4, larger than most LSDIR images."""
    import yaml
    cfg = yaml.safe_load(open(
        f"configs/train/BicubicPlusPlus/bicubicplusplus_{scale}x.yaml", encoding="utf-8"))
    assert cfg["train"]["patch_size"] == patch
    assert cfg["train"]["patch_size"] * scale == hr_crop
