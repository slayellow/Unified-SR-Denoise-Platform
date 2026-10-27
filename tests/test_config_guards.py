"""The four silent failures that could each burn a 120-epoch pretrain.

Every one of these used to proceed quietly with the wrong thing, which is worse
than crashing: the run completes, the numbers look plausible, and the cause is
invisible in the logs. Each test pins the loud behaviour.
"""
import os
import random
import re
import subprocess
import sys
import tempfile

import pytest
import yaml

from src.losses.losses import UnifiedLoss

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train_cfg(**over):
    cfg = yaml.safe_load(open(
        f"{REPO}/configs/train/BicubicPlusPlus/bicubicplusplus_2x.yaml", encoding="utf-8"))
    for k, v in over.items():
        cur, *rest = k.split(".")
        d = cfg
        while rest:
            d = d[cur]
            cur, *rest = rest
        d[cur] = v
    return cfg


def _run(cfg):
    # --work_dir keeps the run directory out of checkpoints/, where it would take a shipped run's name.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh)
        path = fh.name
    try:
        with tempfile.TemporaryDirectory() as work_dir:
            return subprocess.run([sys.executable, "tools/train.py", "--config", path, "--work_dir", work_dir],
                                  capture_output=True, text=True, timeout=300, cwd=REPO)
    finally:
        os.unlink(path)


def test_missing_data_config_raises():
    """An empty degradation config is not 'no degradation' -- most effects
    default to enabled: true, so falling through to {} trains the full
    Real-ESRGAN stack where a resize-only recipe was asked for."""
    r = _run(_train_cfg(**{"data_config_path": "configs/data/no_such_recipe.yaml"}))
    assert "FileNotFoundError" in r.stderr, r.stderr[-400:]


@pytest.mark.parametrize("sched_type", ["CosineAnnealing", "ReduceLROnPlateau"])
def test_unknown_scheduler_raises(sched_type):
    """A typo used to fall through to StepLR(50, 0.5), which would visibly wreck
    the LR curve and read as a modelling result. ReduceLROnPlateau was removed in
    refactor step 7: only the deleted ddrnet/lrcsr configs used it, and its
    verbose=True already raised TypeError on torch 2.9."""
    r = _run(_train_cfg(**{"train.scheduler": {"type": sched_type, "T_max": 120}}))
    assert "scheduler" in r.stderr and "ValueError" in r.stderr, r.stderr[-400:]


def test_unknown_optimizer_raises():
    r = _run(_train_cfg(**{"train.optimizer": {"type": "AdamWW"}}))
    assert "optimizer" in r.stderr and "ValueError" in r.stderr, r.stderr[-400:]


def test_empty_loss_raises_at_construction_not_in_backward():
    with pytest.raises(ValueError, match="손실"):
        UnifiedLoss({})(None, None)


def test_edge_signed_reaches_the_module():
    """The constructor accepted `signed` but the call site never passed it, so a
    config asking for the signed form silently got the magnitude form."""
    u = UnifiedLoss({"edge": {"enabled": True, "weight": 0.02, "mode": "mean", "signed": True}})
    assert u.loss_funcs["edge"].signed is True
    u2 = UnifiedLoss({"edge": {"enabled": True, "weight": 0.02, "mode": "mean"}})
    assert u2.loss_funcs["edge"].signed is False


def test_uint8_casts_round():
    """A bare astype floors, biasing every synthesised pixel by -0.5 DN."""
    import re
    src = open(f"{REPO}/src/data/datasets.py", encoding="utf-8").read()
    bad = [ln for ln in src.split("\n")
           if re.search(r"\.astype\((np\.uint8|'uint8'|\"uint8\")\)", ln) and ".round()" not in ln and "np.round" not in ln]
    assert not bad, "반올림 없는 uint8 캐스팅:\n" + "\n".join(bad)


def test_missing_pretrained_path_raises():
    """Used to warn and train from random init. A finetune config naming a
    checkpoint always means to start from it, and the warning scrolls past in a
    log that then runs for days -- a bulk path rewrite had pointed all three
    Phase-2 configs at a directory that did not exist, and every one would have
    silently thrown away a two-day pretrain."""
    cfg = yaml.safe_load(open(
        f"{REPO}/configs/finetune/BicubicPlusPlus/bicubicplusplus_2x_eo.yaml", encoding="utf-8"))
    cfg["train"]["pretrained_path"] = "checkpoints/does_not_exist/best.pth"
    r = _run(cfg)
    assert "FileNotFoundError" in r.stderr, r.stderr[-400:]


def test_eo_phase2_starts_from_the_phase1_output():
    """The shipped EO Phase-2 config must start from the shipped Phase-1 run of the same scale."""
    import os
    for s in (2, 3, 4):
        cfg = yaml.safe_load(open(
            f"{REPO}/configs/finetune/BicubicPlusPlus/bicubicplusplus_{s}x_eo.yaml",
            encoding="utf-8"))
        phase1 = yaml.safe_load(open(
            f"{REPO}/configs/train/BicubicPlusPlus/bicubicplusplus_{s}x.yaml", encoding="utf-8"))
        p = os.path.join(REPO, cfg["train"]["pretrained_path"])
        assert os.path.isfile(p), f"x{s}: {p} 없음"
        assert os.path.basename(os.path.dirname(p)) == phase1["train"]["save_name"], f"x{s}: Phase-1 출력이 아님 -- {p}"


# ---------- silent fallbacks closed in refactor step 3 ----------
IR_MODEL = {"name": "bicubicplusplus", "scale": 2, "dim": 8, "upsample_mode": "auto", "use_rep": True,
            "clamp_output": True, "achromatic_output": True}
EO_MODEL = {k: v for k, v in IR_MODEL.items() if k != "achromatic_output"}


@pytest.fixture
def ir_checkpoint(tmp_path):
    """A tiny IR checkpoint saved the way the Trainer saves one: weights plus the config it trained with."""
    import torch

    from src.models import build_model

    path = tmp_path / "ir.pth"
    torch.save({"config": {"task": "sr", "model": IR_MODEL},
                "model_state_dict": build_model(dict(IR_MODEL)).state_dict()}, path)
    return path


def _yaml(tmp_path, name, model, **extra):
    path = tmp_path / name
    yaml.safe_dump({"task": "sr", "model": model, "data": {}, **extra}, open(path, "w"))
    return path


@pytest.mark.parametrize("dataset", ["SRDataset", "DenoiseDataset"])
@pytest.mark.parametrize("config", [None, {}])
def test_empty_data_config_raises(dataset, config, tmp_path):
    """An empty degradation config is not 'no degradation': every op then runs on its default.
    The AIMET export used to calibrate on that stack when its --data_config path was wrong."""
    import src.data.datasets as d

    with pytest.raises(ValueError, match="data config"):
        getattr(d, dataset)(str(tmp_path), config=config)


def test_checkpoint_model_config_mismatch_raises(ir_checkpoint):
    import torch

    from src.engine.checkpoint import check_model_config

    ckpt = torch.load(ir_checkpoint, weights_only=False)
    check_model_config(ckpt, dict(IR_MODEL), "ir.pth")
    check_model_config({"model_state_dict": {}}, dict(EO_MODEL), "old.pth")  # older runs saved no config
    with pytest.raises(ValueError, match="achromatic_output"):
        check_model_config(ckpt, dict(EO_MODEL), "ir.pth")


@pytest.mark.parametrize("tool", ["export", "inference", "evaluate"])
def test_tools_reject_checkpoint_trained_with_another_model_config(tool, ir_checkpoint, tmp_path):
    """An IR checkpoint opened with the EO config loads strictly (achromatic_output has no weights)
    and then puts colour on grey input; the tools must stop instead."""
    cfg = _yaml(tmp_path, "eo.yaml", EO_MODEL)
    args = {"export": ["--output", str(tmp_path / "onnx")],
            "inference": ["--input", str(tmp_path), "--output", str(tmp_path / "out"), "--device", "cpu"],
            "evaluate": ["--hr_dir", str(tmp_path), "--lr_dir", str(tmp_path), "--save_dir", str(tmp_path / "ev"),
                         "--device", "cpu"]}[tool]
    r = subprocess.run([sys.executable, f"tools/{tool}.py", "--config", str(cfg), "--checkpoint", str(ir_checkpoint),
                        *args], capture_output=True, text=True, timeout=300, cwd=REPO)
    assert r.returncode != 0 and "achromatic_output" in r.stderr, r.stderr[-600:]


def _apply_qat(*args):
    return subprocess.run([sys.executable, "examples/aimet_quantization/6_apply_qat.py", *args],
                          capture_output=True, text=True, timeout=600, cwd=REPO)


def test_apply_qat_requires_config_and_data_config(ir_checkpoint):
    r = _apply_qat("--checkpoint", str(ir_checkpoint))
    assert r.returncode == 2 and "--config" in r.stderr and "--data_config" in r.stderr, r.stderr[-600:]


def test_apply_qat_missing_data_config_raises(ir_checkpoint, tmp_path):
    pytest.importorskip("aimet_torch")
    cfg = _yaml(tmp_path, "ir.yaml", IR_MODEL)
    r = _apply_qat("--config", str(cfg), "--data_config", str(tmp_path / "missing.yaml"),
                   "--checkpoint", str(ir_checkpoint), "--output_dir", str(tmp_path / "qat"), "--device", "cpu")
    assert "FileNotFoundError" in r.stderr and "missing.yaml" in r.stderr, r.stderr[-600:]


def test_apply_qat_rejects_checkpoint_trained_with_another_model_config(ir_checkpoint, tmp_path):
    pytest.importorskip("aimet_torch")
    cfg = _yaml(tmp_path, "eo.yaml", EO_MODEL)
    data = tmp_path / "data.yaml"
    yaml.safe_dump({"clean_prob": 1.0}, open(data, "w"))
    r = _apply_qat("--config", str(cfg), "--data_config", str(data), "--checkpoint", str(ir_checkpoint),
                   "--output_dir", str(tmp_path / "qat"), "--device", "cpu")
    assert "ValueError" in r.stderr and "achromatic_output" in r.stderr, r.stderr[-600:]


@pytest.mark.parametrize("stale", [True, False], ids=["reused_dir_with_stale_best", "fresh_dir"])
def test_apply_qat_zero_epochs_skips_trainer_and_stale_checkpoint(stale, ir_checkpoint, tmp_path):
    """QAT 0 epochs used to build a Trainer anyway (loading the softening net for EO configs) and then
    restore any qat_checkpoints/best.pth left in a reused output folder over the model it exported.
    The fresh case pins that the output folder is still created once the Trainer no longer makes it."""
    pytest.importorskip("aimet_torch")
    import cv2
    import numpy as np
    import torch

    os.makedirs(tmp_path / "train")
    cv2.imwrite(str(tmp_path / "train" / "a.png"), np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8))
    cfg = _yaml(tmp_path, "ir.yaml", IR_MODEL, data={"train_root": str(tmp_path / "train")},
                train={"patch_size": 16, "batch_size": 1, "num_workers": 0})
    data = tmp_path / "data.yaml"
    yaml.safe_dump({"clean_prob": 1.0}, open(data, "w"))
    out = tmp_path / "new" / "qat"
    if stale:
        os.makedirs(out / "qat_checkpoints")
        torch.save({"model_state_dict": {}}, out / "qat_checkpoints" / "best.pth")
    r = _apply_qat("--config", str(cfg), "--data_config", str(data), "--checkpoint", str(ir_checkpoint),
                   "--output_dir", str(out), "--param_bw", "16", "--output_bw", "16", "--calib_batches", "1",
                   "--train_data_ratio", "1", "--width", "32", "--height", "32", "--device", "cpu")
    assert r.returncode == 0, r.stderr[-800:]
    assert (out / "qat_model.onnx").is_file()
    assert sorted(os.listdir(out / "qat_checkpoints")) == ["best.pth"] if stale else not (out / "qat_checkpoints").exists()


def test_mmp_config_missing_file_raises():
    """A typo in --mmp_config used to print one line and export with no overrides at all."""
    pytest.importorskip("aimet_torch")
    from src.aimet.utils import apply_mmp_from_json

    with pytest.raises(FileNotFoundError):
        apply_mmp_from_json(None, f"{REPO}/configs/aimet/no_such_mmp.json")


def test_mixed_precision_step_requires_mmp_config(ir_checkpoint):
    """Step 5 exists to apply an MMP file; without one it used to apply a hardcoded SVSRNet layer list."""
    r = subprocess.run([sys.executable, "examples/aimet_quantization/5_apply_manual_mixed_precision.py",
                        "--config", "x.yaml", "--data_config", "x.yaml", "--checkpoint", str(ir_checkpoint)],
                       capture_output=True, text=True, timeout=600, cwd=REPO)
    assert r.returncode == 2 and "--mmp_config" in r.stderr, r.stderr[-600:]


def test_gaussian_noise_is_unbiased():
    """A truncating cast after adding zero-mean noise darkens the image by about 0.5 DN;
    add_gaussian_noise serves EO/IR Phase-2 and denoise training on every degraded sample."""
    import numpy as np

    from src.data.datasets import add_gaussian_noise

    random.seed(0)
    np.random.seed(0)
    flat = np.full((64, 64, 3), 120, np.uint8)
    bias = np.mean([add_gaussian_noise(flat, sigma_range=(2, 15)).mean() - 120.0 for _ in range(200)])
    assert abs(bias) < 0.05, bias


def test_resume_with_missing_checkpoint_stops_before_training(tmp_path):
    """A mistyped --resume used to start a fresh run from --config inside the typo's directory."""
    cfg = _yaml(tmp_path, "cfg.yaml", EO_MODEL)
    r = subprocess.run([sys.executable, "tools/train.py", "--config", str(cfg),
                        "--resume", str(tmp_path / "nope" / "last.pth")],
                       capture_output=True, text=True, timeout=300, cwd=REPO)
    assert r.returncode == 2 and "--resume" in r.stderr, r.stderr[-600:]
    assert not (tmp_path / "nope").exists()


@pytest.mark.parametrize("flag", ["--config", "--work_dir", "--data_config", "--lr"])
def test_resume_rejects_flags_it_would_ignore(flag, tmp_path):
    """On resume the run directory's own train_config.yaml/data_config.yaml win, so these flags were
    silently ignored; refusing them keeps a resume from looking like a new recipe."""
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "last.pth").write_bytes(b"")
    value = "1e-5" if flag == "--lr" else str(tmp_path / "x")
    r = subprocess.run([sys.executable, "tools/train.py", "--resume", str(tmp_path / "run" / "last.pth"),
                        flag, value], capture_output=True, text=True, timeout=300, cwd=REPO)
    assert r.returncode == 2 and f"run directory and writes there; drop {flag}" in r.stderr, r.stderr[-600:]
