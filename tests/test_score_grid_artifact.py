"""The grid score must see an upsampler-period pattern and nothing else.

A PixelShuffle/ConvTranspose checkerboard is a pattern that repeats every
`period` output pixels. The score is the spectral peak at k/period against the
surrounding bins, so a smooth image must stay near 1 and a sub-GV pattern on
top of it must stand out. See docs/modules/20260928-MOD-GridScore.md.
"""

import hashlib
import json
import os

import cv2
import numpy as np
import pytest

from examples.evaluation.score_grid_artifact import (
    DEFAULT_FRAMES,
    DEFAULT_MANIFEST,
    grid_score,
    main,
    power_spectrum,
    sharpness_ratio,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_IR_X4 = "checkpoints/bicubicpp_deploy/finetune_bicubicplusplus_ir_sr_x4_dim32_v3"


def _scene(seed=0, h=720, w=1280):
    # sigma 1 keeps real content at 1/4 cycles/px, as a natural frame has
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.normal(128, 40, (h, w)).astype(np.float32), (0, 0), 1)


def _spec(overlay=0, n=8):
    # the protocol sums spectra over frames; a single frame's max-of-9 over median is ~4 by chance alone
    return sum(power_spectrum(_scene(i) + overlay) for i in range(n))


def _grid(h, w, period, amp, axis):
    # cos phases put known energy at 1/period along x, y and the diagonal
    y, x = np.mgrid[:h, :w]
    cx, cy = np.cos(2 * np.pi * (x % period) / period), np.cos(2 * np.pi * (y % period) / period)
    tile = cx if axis == "h" else cx + cy + cx * cy
    return (amp * tile).astype(np.float32)


def test_TC_GRID_01_1_SmoothSceneScoresNearOne():
    s = grid_score(_spec(), 4)
    assert max(s["1/4h"], s["1/4v"], s["1/4d"]) < 3


def test_TC_GRID_01_2_SubGVGridStandsOut():
    s = grid_score(_spec(_grid(720, 1280, 4, 0.5, "hv")), 4)
    assert s["1/4h"] > 20 and s["1/4d"] > 20


def test_TC_GRID_01_3_HorizontalOnlyGridLeavesVerticalFlat():
    s = grid_score(_spec(_grid(720, 1280, 4, 1.0, "h")), 4)
    assert s["1/4h"] > 20
    assert s["1/4v"] < 3


def test_TC_GRID_01_4_OddPeriodBinIsFound():
    # 1280/3 is not an integer bin; the score must still find the peak
    s = grid_score(_spec(_grid(720, 1280, 3, 1.0, "hv")), 3)
    assert s["1/3h"] > 20


def test_TC_GRID_01_5_PeriodBelowTwoRejected():
    with pytest.raises(ValueError):
        grid_score(power_spectrum(_scene()), 1)


def test_TC_GRID_02_1_SharpnessRatioOfSelfIsOne():
    img = _scene()
    assert sharpness_ratio(img, img) == pytest.approx(1.0)


def test_TC_GRID_02_2_BlurLowersSharpness():
    img = _scene()
    assert sharpness_ratio(cv2.GaussianBlur(img, (0, 0), 2), img) < 0.9


def _frames(tmp_path, n):
    for i in range(n):
        (tmp_path / f"{i:06d}.png").write_bytes(b"")
    return str(tmp_path)


def test_TC_GRID_03_1_EmptyFrameSelectionRejected(tmp_path):
    # the frame check runs before any checkpoint is opened, so a bogus --ckpt never gets that far
    with pytest.raises(ValueError):
        main(["--ckpt", str(tmp_path / "missing.pth"), "--frames", _frames(tmp_path, 1), "--index", "5:9:1"])


def test_TC_GRID_03_2_TooFewFramesRejected(tmp_path):
    # 11 of the protocol's 50 is what an old index rule gives on a reorganised folder (2026-09-29)
    with pytest.raises(ValueError):
        main(["--ckpt", str(tmp_path / "missing.pth"), "--frames", _frames(tmp_path, 11), "--index", "0:11:1"])


def test_TC_GRID_03_3_MinFramesIsInclusive(tmp_path):
    # exactly --min_frames passes the frame check and reaches the (missing) checkpoint
    with pytest.raises(FileNotFoundError):
        main(
            [
                "--ckpt",
                str(tmp_path / "missing.pth"),
                "--frames",
                _frames(tmp_path, 11),
                "--index",
                "0:11:1",
                "--min_frames",
                "11",
                "--manifest",
                "",
            ]
        )


@pytest.mark.parametrize("min_frames", ["0", "-1"])
def test_TC_GRID_03_4_MinFramesBelowOneStillRejectsEmpty(tmp_path, min_frames):
    with pytest.raises(ValueError):
        main(
            [
                "--ckpt",
                str(tmp_path / "missing.pth"),
                "--frames",
                _frames(tmp_path, 1),
                "--index",
                "5:9:1",
                "--min_frames",
                min_frames,
            ]
        )


def _manifest_case(tmp_path, listed):
    # four frames whose bytes are their index, and a sha256sum-format list of what the protocol expects
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(4):
        (frames / f"{i:06d}.png").write_bytes(str(i).encode())
    manifest = tmp_path / "frames.sha256"
    with open(manifest, "w") as f:
        for name, data in listed:
            f.write(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
    return [
        "--ckpt",
        str(tmp_path / "missing.pth"),
        "--frames",
        str(frames),
        "--index",
        "0:3:1",
        "--min_frames",
        "3",
        "--manifest",
        str(manifest),
    ]


@pytest.mark.parametrize(
    "listed",
    [
        [("000000.png", b"0"), ("000001.png", b"1"), ("000002.png", b"changed")],  # a frame's bytes changed
        [("000000.png", b"0"), ("000002.png", b"2"), ("000003.png", b"3")],  # the selection shifted
    ],
)
def test_TC_GRID_03_5_FramesNotMatchingManifestRejected(tmp_path, listed):
    with pytest.raises(ValueError):
        main(_manifest_case(tmp_path, listed))


def test_TC_GRID_03_6_FramesMatchingManifestAccepted(tmp_path):
    listed = [("000000.png", b"0"), ("000001.png", b"1"), ("000002.png", b"2")]
    with pytest.raises(FileNotFoundError):
        main(_manifest_case(tmp_path, listed))


def test_TC_GRID_04_1_ShippedIRx4ReproducesRecordedP2(tmp_path, monkeypatch):
    """The shipped IR x4 (ep89), rebuilt from its saved config, scores exactly what was recorded for it."""
    need = [
        f"{SHIPPED_IR_X4}/best.pth",
        f"{SHIPPED_IR_X4}/train_config.yaml",
        f"{SHIPPED_IR_X4}/grid_best.json",
        DEFAULT_FRAMES,
        DEFAULT_MANIFEST,
    ]
    missing = [p for p in need if not os.path.exists(os.path.join(REPO, p))]
    if missing:
        pytest.skip(f"not on this machine: {missing}")
    monkeypatch.chdir(REPO)  # the frame and manifest defaults are repo-relative
    main(["--ckpt", f"{SHIPPED_IR_X4}/best.pth", "--json", str(tmp_path / "p2.json")])
    ours = json.load(open(tmp_path / "p2.json"))
    ref = json.load(open(f"{SHIPPED_IR_X4}/grid_best.json"))
    numeric = sorted(k for k, v in ref.items() if isinstance(v, (int, float)))
    assert len(numeric) == 9
    assert {k: ours[k] for k in numeric} == {k: ref[k] for k in numeric}


TINY_X4 = {"name": "bicubicplusplus", "scale": 4, "dim": 8, "upsample_mode": "auto", "clamp_output": True}


def test_TC_GRID_04_2_LoadsFromSavedConfigWithoutYamlBeside(tmp_path):
    """A checkpoint copied into a release folder has no train_config.yaml beside it."""
    import torch

    from examples.evaluation.score_grid_artifact import load_model
    from src.models import build_model

    sd = build_model(dict(TINY_X4)).state_dict()
    torch.save({"config": {"model": TINY_X4}, "model_state_dict": sd, "epoch": 7}, tmp_path / "best.pth")
    model, scale, epoch = load_model(str(tmp_path / "best.pth"), "cpu")
    assert (scale, epoch) == (4, 7) and not model.training
    assert all(torch.equal(v, sd[k]) for k, v in model.state_dict().items())


def test_TC_GRID_04_3_NoSavedConfigRejected(tmp_path):
    """A yaml beside the file may belong to another run, so it is not a fallback."""
    import torch
    import yaml

    from examples.evaluation.score_grid_artifact import load_model
    from src.models import build_model

    torch.save({"model_state_dict": build_model(dict(TINY_X4)).state_dict()}, tmp_path / "best.pth")
    yaml.safe_dump({"model": TINY_X4}, open(tmp_path / "train_config.yaml", "w"))
    with pytest.raises(ValueError, match="saved model config"):
        load_model(str(tmp_path / "best.pth"), "cpu")
