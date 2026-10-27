"""Re-parameterization equivalence: switch_to_deploy() must not change outputs.

Random weights N(0, 0.2) are injected before the check on purpose: a fresh
init can make a wrong fusion look exact (a 180-degree kernel-rotation bug in
the since-deleted LRCSR stayed invisible that way, fixed in 0a3bad5).

fp64 separates structural error from float accumulation: a correct fusion
lands near 1e-13 in fp64 even when 8 blocks accumulate ~2e-4 in fp32.
"""
import copy

import pytest
import torch

from src.models import build_model

# Every case exposes switch_to_deploy().  For the noop-suffixed ones it is an
# intended no-op (plain CNNs, nothing to fuse) -- equality must still hold so a
# future accidental body is caught.
CASES = [
    pytest.param({"name": "svfocussrnet", "scale": 2, "dim": 32,
                  "n_resblocks": 8, "use_advanced_rep": False},
                 id="svfocussrnet_2x_blk8"),
    pytest.param({"name": "svfocussrnet", "scale": 1, "dim": 32,
                  "n_resblocks": 3, "use_advanced_rep": True},
                 id="svfocusdenoise_adv_blk3"),
    pytest.param({"name": "bicubicplusplus", "scale": 2, "dim": 32,
                  "bias": False, "negative_slope": 0.1,
                  "clamp_output": True, "upsample_mode": "auto"},
                 id="bicubicplusplus_2x_noop"),
    # use_rep=True is the case with something to fuse: five branches per 3x3,
    # folded back to that 3x3 before QAT ever sees the graph. x4 also exercises
    # the split upsampler, whose two conv_out stages are ECB as well.
    pytest.param({"name": "bicubicplusplus", "scale": 2, "dim": 32,
                  "bias": False, "negative_slope": 0.1,
                  "clamp_output": True, "upsample_mode": "auto",
                  "use_rep": True},
                 id="bicubicplusplus_2x_rep"),
    pytest.param({"name": "bicubicplusplus", "scale": 4, "dim": 32,
                  "bias": False, "negative_slope": 0.1,
                  "clamp_output": True, "upsample_mode": "auto",
                  "use_rep": True},
                 id="bicubicplusplus_4x_rep"),
]

TOL = {torch.float32: 1e-3, torch.float64: 1e-10}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64],
                         ids=["fp32", "fp64"])
@pytest.mark.parametrize("cfg", CASES)
def test_switch_to_deploy_is_exact(cfg, dtype):
    torch.manual_seed(0)
    model = build_model(copy.deepcopy(cfg)).to(dtype).eval()
    assert hasattr(model, "switch_to_deploy")
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 4 and p.requires_grad:  # keep frozen AnchorOp Dirac
                p.normal_(0, 0.2)

    scale = cfg.get("scale", 1)
    x = torch.rand(1, 3, 48, 64, dtype=dtype)
    with torch.no_grad():
        before = model(x)
        model.switch_to_deploy()
        after = model(x)

    assert before.shape == (1, 3, 48 * scale, 64 * scale)
    assert after.shape == before.shape
    diff = (before - after).abs().max().item()
    assert diff < TOL[dtype], f"train/deploy diverged by {diff:.3e}"
