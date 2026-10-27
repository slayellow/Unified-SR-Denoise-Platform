"""Release golden (MOD-REL): the shipped Bicubic++ SR v3 and BicubicPPDenoise[v1] stay reproducible.

TC ids follow docs/modules/20261015-MOD-ReleaseGolden.md section 4. The oracles are the shipped files
under results/ and checkpoints/, which git does not track: a case whose file is missing is skipped,
so on the training server every case must run (check with pytest -rs).
"""

import hashlib
import importlib.util
import json
import os

import pytest
import torch
import yaml

from src.engine.checkpoint import normalize_state_dict
from src.models import build_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = "results/release_v3"
DN = "results/denoise_release_v1"
P1 = "checkpoints/bicubicpp_deploy/train_bicubicplusplus_sr_x{}_dim32_knee_ep120_bs16_lr5e-4_mse_v3"
SOFTEN = (
    "checkpoints/csuav_deploy/"
    "train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309"
)

# name: (checkpoint, config whose model block the checkpoint was trained with, checkpoint md5, deploy output sha)
CHECKPOINTS = {
    "EO_x2": (
        f"{V3}/checkpoints/EO_x2.pth",
        "configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml",
        "30b8a4caab33f39227393f2587839e93",
        "b91aec1ed6cf2c951b31735c896b86c8",
    ),
    "EO_x3": (
        f"{V3}/checkpoints/EO_x3.pth",
        "configs/finetune/BicubicPlusPlus/bicubicplusplus_3x_eo.yaml",
        "735161389a71f0f77e7b23c2a1819ae3",
        "6200304645739866586e268e6a6eb7e9",
    ),
    "EO_x4": (
        f"{V3}/checkpoints/EO_x4.pth",
        "configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_eo.yaml",
        "7f84de9bdca8f6432acb209657b85167",
        "6f684729631193a9ea7fee1d29fa1f0d",
    ),
    "IR_x2": (
        f"{V3}/checkpoints/IR_x2.pth",
        "configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_ir.yaml",
        "3b16b81bcbf6b69c4de4a8d99c88a5fa",
        "e6be124381dc47c70d5e9e0ffc17588f",
    ),
    "IR_x3": (
        f"{V3}/checkpoints/IR_x3.pth",
        "configs/finetune/BicubicPlusPlus/bicubicplusplus_3x_ir.yaml",
        "4d4803fc658750f98d6b934580c226da",
        "b56def19f6eddaf7c09aae3b5f1f177a",
    ),
    "IR_x4": (
        f"{V3}/checkpoints/IR_x4.pth",
        "configs/finetune/BicubicPlusPlus/bicubicplusplus_4x_ir.yaml",
        "64849a367e0a9e25dd8d7631313c0836",
        "90165dd96518a8489254ad84df430a61",
    ),
    "DN_v1": (
        f"{DN}/checkpoint/best.pth",
        "configs/train/Denoise/bicubicppdenoise.yaml",
        "198930772692f5d17ab693847799d9eb",
        "88037953d30a722a98c33e1a6cd36f40",
    ),
    "P1_x2": (
        f"{P1.format(2)}/best.pth",
        "configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml",
        "30b8a4caab33f39227393f2587839e93",
        "b91aec1ed6cf2c951b31735c896b86c8",
    ),
    "P1_x3": (
        f"{P1.format(3)}/best.pth",
        "configs/train/BicubicPlusPlus/bicubicplusplus_3x.yaml",
        "dde24507313b91bf9ca3abccfe69d30e",
        "c7c3b70cb3234aaf3c194ee6700c86ee",
    ),
    "P1_x4": (
        f"{P1.format(4)}/best.pth",
        "configs/train/BicubicPlusPlus/bicubicplusplus_4x.yaml",
        "19b66cb0f544087355d35b97f3c604ba",
        "27163e7376838a2fe26ec05b38f19aae",
    ),
    "SOFTEN": (
        f"{SOFTEN}/best.pth",
        f"{SOFTEN}/train_config.yaml",
        "0df4cc5174f9c8463ee28b0f036b42fb",
        "9283350c54f4baf48affbe1d2ac08628",
    ),
}

# profile: (checkpoint name, width, height, shipped qat dir, ONNX md5 when the shipped file is not the oracle)
# EO was exported on a GPU and comes out one W16 ulp apart on the CPU, so EO is held to the CPU export of
# the commit this test was written at; IR and DN reproduce the shipped bytes exactly on the CPU.
PROFILES = {
    "x2_eo": ("EO_x2", 640, 360, f"{V3}/x2_eo/qat", "7e2fe59ffbeb1660b7f83091e63a87e5"),
    "x3_eo": ("EO_x3", 428, 240, f"{V3}/x3_eo/qat", "411a63efe1e55cdca648ec9da0959448"),
    "x4_eo": ("EO_x4", 320, 180, f"{V3}/x4_eo/qat", "eb299ebbbcbf5b2888d24ff7fc4d4981"),
    "x2_ir": ("IR_x2", 640, 360, f"{V3}/x2_ir/qat", None),
    "x2_ir_1024": ("IR_x2", 640, 512, f"{V3}/x2_ir_1024/qat", None),
    "x3_ir": ("IR_x3", 428, 240, f"{V3}/x3_ir/qat", None),
    "x3_ir_1024": ("IR_x3", 428, 342, f"{V3}/x3_ir_1024/qat", None),
    "x4_ir": ("IR_x4", 320, 180, f"{V3}/x4_ir/qat", None),
    "x4_ir_1024": ("IR_x4", 320, 256, f"{V3}/x4_ir_1024/qat", None),
    "dn_v1": ("DN_v1", 1280, 720, f"{DN}/dn_bpp_half4_w16a16/qat", None),
}

# architecture: (config, state_dict sha after torch.manual_seed(0) and build_model)
SEEDED_INIT = {
    "sr_x2_auto": ("configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml", "03e0af2d0a2e472c"),
    "sr_x3_deconv": ("configs/train/BicubicPlusPlus/bicubicplusplus_3x.yaml", "4bfca691a0cb2739"),
    "sr_x4_split": ("configs/train/BicubicPlusPlus/bicubicplusplus_4x.yaml", "afe272479773171d"),
    "dn_half4": ("configs/train/Denoise/bicubicppdenoise.yaml", "75b4d0de7dc830d9"),
}


def repo_path(p):
    return os.path.join(REPO, p)


def need(p):
    if not os.path.isfile(repo_path(p)):
        pytest.skip(f"{p} not on this machine")
    return repo_path(p)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def sha(tensors):
    h = hashlib.sha256()
    for k, v in tensors:
        h.update(k.encode())
        h.update(v.detach().float().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def model_block(cfg_path):
    with open(repo_path(cfg_path), encoding="utf-8") as f:
        return yaml.safe_load(f)["model"]


def apply_qat6():
    spec = importlib.util.spec_from_file_location("apply_qat6", repo_path("examples/aimet_quantization/6_apply_qat.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def q6():
    pytest.importorskip("aimet_torch")
    return apply_qat6()


def deploy_model(name, q6):
    """The model as the shipped flows build it: strict load, ECB fold, IR achromatic fold.

    The softening net is built the way Trainer._load_denoise_net builds it.
    """
    ckpt = torch.load(need(CHECKPOINTS[name][0]), map_location="cpu", weights_only=False)
    cfg = {"model": dict(ckpt["config"]["model"])}
    model = build_model(dict(cfg["model"]))
    state = ckpt["model_state_dict"]
    model.load_state_dict(normalize_state_dict(state) if name == "SOFTEN" else state)
    model.eval()
    model.switch_to_deploy()
    if name != "SOFTEN":
        q6._check_achromatic_fold(model, cfg)
    return model


# ---------- MREQ-REL-01: released checkpoints ----------
@pytest.mark.parametrize("name", list(CHECKPOINTS))
def test_TC_REL_01_1_CheckpointMatchesConfigAndLoadsStrict(name):
    path, cfg_path, ckpt_md5, _ = CHECKPOINTS[name]
    ckpt = torch.load(need(path), map_location="cpu", weights_only=False)
    assert md5(repo_path(path)) == ckpt_md5
    assert ckpt["config"]["model"] == model_block(cfg_path)
    build_model(dict(ckpt["config"]["model"])).load_state_dict(ckpt["model_state_dict"])


@pytest.mark.parametrize("name", list(CHECKPOINTS))
def test_TC_REL_01_2_DeployOutputUnchanged(name, q6):
    model = deploy_model(name, q6)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x = torch.rand(1, 3, 48, 64, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            y = model(x)
    finally:
        torch.set_num_threads(threads)
    assert sha(model.state_dict().items()) + sha([("y", y)]) == CHECKPOINTS[name][3]


# ---------- MREQ-REL-02: W16A16 export graph ----------
@pytest.mark.parametrize("profile", list(PROFILES))
def test_TC_REL_02_1_AimetExportMatchesShipped(profile, q6, tmp_path, monkeypatch):
    from aimet_common.defs import QuantScheme

    from src.aimet.quant_sim import DEFAULT_QUANTSIM_CONFIG, build_quantsim, prepare_deploy_model
    from src.aimet.utils import apply_mmp_from_json

    name, w, h, ship_dir, onnx_md5 = PROFILES[profile]
    ship_onnx = need(f"{ship_dir}/qat_model.onnx")
    monkeypatch.chdir(tmp_path)  # AIMET writes some artefacts relative to the working directory
    torch.manual_seed(0)
    model = prepare_deploy_model(deploy_model(name, q6))
    dummy = torch.randn(1, 3, h, w)
    sim = build_quantsim(
        model,
        dummy,
        quant_scheme=QuantScheme.training_range_learning_with_tf_init,
        default_output_bw=16,
        default_param_bw=16,
        config_file=DEFAULT_QUANTSIM_CONFIG,
        prepare=False,
    )
    apply_mmp_from_json(sim, repo_path("configs/aimet/mmp_noprune_w16a16.json"))
    sim.compute_encodings(
        forward_pass_callback=q6.pass_calibration_data,
        forward_pass_callback_args=[torch.rand(1, 3, h, w) for _ in range(2)],
    )
    sim.export(path=str(tmp_path), filename_prefix="qat_model", dummy_input=dummy)

    assert md5(tmp_path / "qat_model.onnx") == (onnx_md5 or md5(ship_onnx))
    ours = json.load(open(tmp_path / "qat_model.encodings"))
    ship = json.load(open(f"{os.path.dirname(ship_onnx)}/qat_model.encodings"))
    # Calibration is unseeded in the shipped flow, so activation ranges are compared by name only.
    assert [e["name"] for e in ours["activation_encodings"]] == [e["name"] for e in ship["activation_encodings"]]
    if onnx_md5 is None:
        assert ours["param_encodings"] == ship["param_encodings"]
    else:
        assert [e["name"] for e in ours["param_encodings"]] == [e["name"] for e in ship["param_encodings"]]


# ---------- MREQ-REL-03: seeded init ----------
@pytest.mark.parametrize("arch", list(SEEDED_INIT))
def test_TC_REL_03_1_SeededInitUnchanged(arch):
    cfg_path, expected = SEEDED_INIT[arch]
    torch.manual_seed(0)
    model = build_model(dict(model_block(cfg_path)))
    assert sha(model.state_dict().items()) == expected
