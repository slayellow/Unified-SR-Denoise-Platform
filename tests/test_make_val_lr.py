"""make_val_lr.py rebuilds the gate val sets; pin its output on two real val images per shipped recipe.

The val images and the frozen softening net are not in git: the case is skipped where they are missing.
Recorded at the commit after the 2026-10-16 recipe fixes (docs/CONFIG_RATIONALE.md 4.4).
"""

import hashlib
import importlib.util
import os
import shutil

import cv2
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HR = "/mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val/HR"
SOFTEN = (
    f"{REPO}/checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_"
    "charbonnier_edge_ssim_hotpixel_260309/best.pth"
)

# recipe: (data config, scale, softened, md5 of the decoded pixels of 0000.png and 0001.png)
CASES = {
    "eo_phase2_x4": (
        "configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml",
        4,
        True,
        ("a58bbe19b1b495f9d47dbdbd4a67ff69", "0be747e1fef173551508aa2f59d9ef97"),
    ),
    "ir_phase2_x3": (
        "configs/data/bicubicpp/sr_finetune_ir_phase-2.yaml",
        3,
        False,
        ("901f61e6b16f090c3c1d4b5de9a496dc", "a08022886b3dd75ed307dabc61eecc27"),
    ),
    "phase1_x2": (
        "configs/data/bicubicpp/sr_train_phase-1.yaml",
        2,
        True,
        ("1c5b8ceb0d142654cadfec269ebfbfe9", "ca2383973f143e626221b97c9911ca37"),
    ),
}


def _make_val_lr():
    spec = importlib.util.spec_from_file_location("make_val_lr", f"{REPO}/examples/dataset_preparation/make_val_lr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(case, tmp_path):
    cfg, scale, softened, _ = CASES[case]
    names = ["0000.png", "0001.png"]
    if not all(os.path.isfile(os.path.join(HR, n)) for n in names) or (softened and not os.path.isfile(SOFTEN)):
        pytest.skip("val HR images or the softening checkpoint are not on this machine")
    hr = tmp_path / "hr"
    hr.mkdir()
    for n in names:
        shutil.copy(os.path.join(HR, n), hr / n)
    args = [
        "--data_config",
        f"{REPO}/{cfg}",
        "--scale",
        str(scale),
        "--hr_root",
        str(hr),
        "--out_dir",
        str(tmp_path / "lr"),
    ]
    if softened:
        args += ["--denoise_checkpoint", SOFTEN]
    _make_val_lr().main(args)
    return tuple(
        hashlib.md5(cv2.imread(str(tmp_path / "lr" / n), cv2.IMREAD_UNCHANGED).tobytes()).hexdigest() for n in names
    )


@pytest.mark.parametrize("case", list(CASES))
def test_make_val_lr_output_unchanged(case, tmp_path):
    assert run(case, tmp_path) == CASES[case][3]
