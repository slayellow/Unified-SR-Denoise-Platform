"""build_model registry contract.

Every name the if/elif chain in src/models/__init__.py accepts must build from
a minimal config, return an nn.Module, and map (N, 3, H, W) -> (N, 3, H*s, W*s)
on CPU with finite values.

(QuickSRNet, LRCSR, SVSRNet, DDRNet, MambaIR/v2, NAFNet and Restormer were
deleted in refactor step 7, the Guided-SR family and rrdbnet in 3fa11ee and
7ba7874; if one is ever restored, add its case back here.)
"""
import copy

import pytest
import torch
import torch.nn as nn

from src.models import build_model

H, W = 16, 16  # even: the legacy case's stride-2 stem must land exactly on H*s

# (config, expected output scale)
CASES = [
    pytest.param({"name": "svfocussrnet", "scale": 2, "dim": 16,
                  "n_resblocks": 2}, 2, id="svfocussrnet-x2"),
    pytest.param({"name": "svfocussrnet", "scale": 1, "dim": 16,
                  "n_resblocks": 2, "use_advanced_rep": True}, 1,
                 id="svfocussrnet-x1-denoise"),
    # all three upsample_mode paths and the x2/x3/x4 scales in training now
    pytest.param({"name": "bicubicplusplus", "scale": 2, "dim": 16,
                  "upsample_mode": "legacy"}, 2, id="bicubicpp-x2-legacy"),
    pytest.param({"name": "bicubicplusplus", "scale": 3, "dim": 16,
                  "upsample_mode": "auto"}, 3, id="bicubicpp-x3-auto"),
    pytest.param({"name": "bicubicplusplus", "scale": 4, "dim": 16,
                  "upsample_mode": "auto", "mid_feats": 8}, 4,
                 id="bicubicpp-x4-auto-split"),
    pytest.param({"name": "bicubicplusplus", "scale": 3, "dim": 16,
                  "upsample_mode": "deconv"}, 3, id="bicubicpp-x3-deconv"),
]


@pytest.mark.parametrize("cfg,scale", CASES)
def test_registry_name_builds_and_upscales(cfg, scale):
    model = build_model(copy.deepcopy(cfg))
    assert isinstance(model, nn.Module)
    model.eval()
    x = torch.rand(1, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor), f"forward returned {type(y)}"
    assert y.shape == (1, 3, H * scale, W * scale), f"got {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "forward produced non-finite values"


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model({"name": "definitely_not_a_model"})


@pytest.mark.parametrize("act,neg_out", [(None, -0.1), ("leaky_relu", -0.1), ("relu", 0.0)])
def test_bicubicpp_act_selects_the_activation(act, neg_out):
    # a missing key must keep LeakyReLU(0.1): every shipped checkpoint was trained with it
    cfg = {"name": "bicubicplusplus", "scale": 4, "dim": 16, "upsample_mode": "auto", "mid_feats": 8}
    if act is not None:
        cfg["act"] = act
    model = build_model(cfg)
    assert model.act(torch.tensor([-1.0])).item() == pytest.approx(neg_out)


def test_bicubicpp_unknown_act_raises():
    with pytest.raises(ValueError):
        build_model({"name": "bicubicplusplus", "scale": 2, "dim": 16, "act": "swish"})
