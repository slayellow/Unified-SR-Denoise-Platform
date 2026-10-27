"""tools/evaluate.py on a tiny checkpoint: size mismatch, saved images, required output folder."""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.models import build_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X3 = {"name": "bicubicplusplus", "scale": 3, "dim": 8, "upsample_mode": "deconv", "clamp_output": True}


@pytest.fixture
def x3_run(tmp_path):
    torch.manual_seed(0)
    torch.save({"config": {"model": X3}, "model_state_dict": build_model(dict(X3)).state_dict()}, tmp_path / "x3.pth")
    yaml.safe_dump({"task": "sr", "model": X3, "data": {}}, open(tmp_path / "x3.yaml", "w"))
    return tmp_path


def _pair(root, hr_w):
    rng = np.random.default_rng(0)
    for kind, (h, w) in (("HR", (192, hr_w)), ("LR", (64, 64))):  # LPIPS pools need >= ~100 px
        os.makedirs(root / kind, exist_ok=True)
        cv2.imwrite(str(root / kind / "a.png"), rng.integers(0, 255, (h, w, 3), np.uint8))


def _evaluate(run, *extra):
    # cwd is the tmp folder, so an evaluate that falls back to a relative default writes there, not into the repo
    return subprocess.run([sys.executable, f"{REPO}/tools/evaluate.py", "--config", str(run / "x3.yaml"),
                           "--checkpoint", str(run / "x3.pth"), "--hr_dir", str(run / "HR"), "--lr_dir", str(run / "LR"),
                           "--device", "cpu", *extra], capture_output=True, text=True, timeout=600, cwd=run)


def test_x3_output_narrower_than_hr_is_cropped(x3_run):
    """x3 of a 1280-wide frame is 1278 wide; the metrics used to assert on the shape and stop."""
    _pair(x3_run, hr_w=194)
    r = _evaluate(x3_run, "--save_dir", str(x3_run / "ev"))
    assert r.returncode == 0, r.stderr[-800:]
    assert len(open(x3_run / "ev" / "x3" / "metrics.csv").read().splitlines()) == 2


def test_saved_image_matches_inference(x3_run):
    """evaluate truncated to uint8 where inference rounds, so the same model saved two different images."""
    _pair(x3_run, hr_w=192)
    assert _evaluate(x3_run, "--save_dir", str(x3_run / "ev"), "--save_images").returncode == 0
    r = subprocess.run([sys.executable, "tools/inference.py", "--config", str(x3_run / "x3.yaml"),
                        "--checkpoint", str(x3_run / "x3.pth"), "--input", str(x3_run / "LR"),
                        "--output", str(x3_run / "inf"), "--device", "cpu"],
                       capture_output=True, text=True, timeout=600, cwd=REPO)
    assert r.returncode == 0, r.stderr[-800:]
    assert np.array_equal(cv2.imread(str(x3_run / "ev" / "x3" / "a.png")), cv2.imread(str(x3_run / "inf" / "a.png")))


def test_save_dir_is_required(x3_run):
    """The default wrote every evaluation into the repository's results/."""
    _pair(x3_run, hr_w=192)
    r = _evaluate(x3_run)
    assert r.returncode == 2 and "--save_dir" in r.stderr


def test_export_failure_exits_nonzero(x3_run):
    """A failed torch.onnx.export used to print the error and exit 0, leaving no ONNX behind."""
    r = subprocess.run([sys.executable, f"{REPO}/tools/export.py", "--config", str(x3_run / "x3.yaml"),
                        "--checkpoint", str(x3_run / "x3.pth"), "--output", str(x3_run / "onnx"), "--opset", "99"],
                       capture_output=True, text=True, timeout=600, cwd=x3_run)
    assert r.returncode != 0
