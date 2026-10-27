"""BicubicPPDenoise (MOD-BPPD): Bicubic++ at scale 1 as an EO denoiser.

TC ids follow docs/modules/20261012-MOD-BicubicPPDenoise.md section 4.
"""

import io

import onnx
import pytest
import torch

from src.models import build_model
from src.models.bicubicplusplus import BicubicPlusPlus


def denoiser(mode="legacy", n_body=2, act="leaky_relu", use_rep=True, seed=0):
    torch.manual_seed(seed)
    return build_model(
        {
            "name": "bicubicplusplus",
            "scale": 1,
            "dim": 32,
            "upsample_mode": mode,
            "n_body": n_body,
            "act": act,
            "use_rep": use_rep,
            "clamp_output": True,
        }
    ).eval()


def onnx_ops(model, h=48, w=64):
    buf = io.BytesIO()
    torch.onnx.export(model, torch.rand(1, 3, h, w), buf, opset_version=17, dynamo=False)
    g = onnx.load_from_string(buf.getvalue()).graph
    return [n.op_type for n in g.node], g


# ---------- MREQ-BPPD-01: scale-1 graph ----------
@pytest.mark.parametrize("mode", ["legacy", "auto"])
@pytest.mark.parametrize("n_body", [2, 4])
def test_TC_BPPD_01_1_ShapeAndNearIdentityAtInit(mode, n_body):
    m = denoiser(mode, n_body)
    x = torch.rand(2, 3, 48, 64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape
    assert float((y - x).abs().mean()) < 0.02  # residual starts near zero: output ~ input


def test_TC_BPPD_01_2_BodyDepthAndSRKeysUnchanged():
    keys2 = set(denoiser("legacy", 2, use_rep=False).state_dict())
    keys4 = set(denoiser("legacy", 4, use_rep=False).state_dict())
    assert keys2 == {"conv0.weight", "conv1.weight", "conv2.weight", "conv_out.weight"}
    assert keys4 - keys2 == {"body_extra.0.weight", "body_extra.1.weight"}
    sr = BicubicPlusPlus(scaling_factor=2, upsample_mode="auto", use_rep=False)
    assert set(sr.state_dict()) == {"conv0.weight", "conv1.weight", "conv2.weight", "conv_out.weight"}


# ---------- MREQ-BPPD-02: deploy fold and export ----------
@pytest.mark.parametrize(
    "mode,n_body,act", [("legacy", 4, "leaky_relu"), ("legacy", 2, "leaky_relu"), ("auto", 2, "relu")]
)
def test_TC_BPPD_02_1_DeployFoldIsExact(mode, n_body, act):
    m = denoiser(mode, n_body, act, use_rep=True, seed=3)
    # Train-time branches must carry weight, or the fold check is vacuous.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        before = m(x)
        m.switch_to_deploy()
        after = m(x)
    assert not any("sobel" in k or "laplacian" in k for k in m.state_dict())
    assert torch.allclose(before, after, atol=1e-5)


@pytest.mark.parametrize("mode,shuffles", [("legacy", 1), ("auto", 0)])
def test_TC_BPPD_02_2_OnnxGraph(mode, shuffles):
    m = denoiser(mode, 4)
    m.switch_to_deploy()
    ops, g = onnx_ops(m)
    assert ops.count("DepthToSpace") == shuffles  # auto at scale 1 has no shuffle, not a DepthToSpace(1)
    last = [n for n in g.node if g.output[0].name in n.output][0]
    assert last.op_type == "Clip"
    assert ops.count("Add") == 2  # body skip + input skip


IR_HEADS = [
    pytest.param({"scale": 2, "upsample_mode": "auto"}, id="x2-auto"),
    pytest.param({"scale": 3, "upsample_mode": "deconv"}, id="x3-deconv"),
    pytest.param({"scale": 4, "upsample_mode": "auto"}, id="x4-split"),
]


@pytest.mark.parametrize("use_rep", [True, False], ids=["rep", "plain"])
@pytest.mark.parametrize("head", IR_HEADS)
def test_TC_BPPD_02_3_AchromaticFoldInSwitchToDeploy(head, use_rep):
    """switch_to_deploy folds the IR channel mean into the last layer, so every deploy path (tools, AIMET 1-7)
    exports the shipped graph. An output-equivalence check alone passes with the mean still in forward."""
    torch.manual_seed(0)
    cfg = {"name": "bicubicplusplus", "dim": 16, "bias": False, "negative_slope": 0.1, "clamp_output": True,
           "use_rep": use_rep, "achromatic_output": True, **head}
    model = build_model(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    x = torch.rand(1, 3, 24, 32)
    with torch.no_grad():
        before = model(x)
        model.switch_to_deploy()
        after = model(x)
        model.switch_to_deploy()  # idempotent
        again = model(x)
    assert model.achromatic_output is False
    if head["upsample_mode"] == "deconv":
        w = model.deconv.weight
        assert all(torch.equal(w[:, 0], w[:, c]) for c in (1, 2))
    else:
        w = (model.conv_out2 if model.upsample_split else model.conv_out).weight
        g = w.view(3, w.shape[0] // 3, *w.shape[1:])
        assert torch.equal(g[0], g[1]) and torch.equal(g[0], g[2])
    assert (after - before).abs().max() < 1e-5 and torch.equal(after, again)
    buf = io.BytesIO()
    torch.onnx.export(model, x, buf, opset_version=17, dynamo=False)
    ops = {n.op_type for n in onnx.load_from_string(buf.getvalue()).graph.node}
    assert "ReduceMean" not in ops and "Expand" not in ops


def test_TC_BPPD_02_4_ClampIsTheDefault():
    """Every shipped model ends on Hardtanh 'final_act', the layer the HTP conversion and gst ask for;
    a config that forgot clamp_output used to export a graph without it."""
    assert isinstance(build_model({"name": "bicubicplusplus", "scale": 2}).final_act, torch.nn.Hardtanh)
    assert not hasattr(build_model({"name": "bicubicplusplus", "scale": 2, "clamp_output": False}), "final_act")


# ---------- MREQ-BPPD-03: rejected configurations ----------
@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(scaling_factor=1, upsample_mode="deconv"), "deconv"),
        (dict(scaling_factor=0), "scaling_factor"),
        (dict(scaling_factor=1, achromatic_output=True), "achromatic"),
        (dict(scaling_factor=2, n_body=3), "n_body"),
        (dict(scaling_factor=1, n_body=1), "n_body"),
        (dict(scaling_factor=2, upsample_mode="legacy", use_rep=True), "use_rep"),  # used to drop ECB silently
        (dict(scaling_factor=2, upsample_mode="auto", use_rep=True, bias=True), "bias"),  # ECB bias folds wrong at borders
        (dict(scaling_factor=1.5), "integer"),  # int(1.5) used to build a denoiser
    ],
)
def test_TC_BPPD_03_1_InvalidConfigRejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BicubicPlusPlus(**kwargs)


def test_TC_BPPD_03_3_UnknownModelKeyRejected():
    # A typo such as use_reps used to be ignored and trained the default model (legacy, no ECB).
    with pytest.raises(TypeError, match="use_reps"):
        build_model({"name": "bicubicplusplus", "scale": 2, "use_reps": True})


def test_TC_BPPD_03_2_HalfResFailsLoudlyOnOddSize():
    m = denoiser("legacy", 2)
    with pytest.raises(RuntimeError):
        m(torch.rand(1, 3, 47, 64))
