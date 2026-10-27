"""Characterization of the data paths the shipped recipes use, pinned before dead code around them is removed."""

import random

import cv2
import numpy as np
import pytest

from src.data.datasets import PairedDataset, SRDataset, add_vertical_line_noise


def _png(path, h=16, w=16, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), value, np.uint8))


def test_paired_dataset_pairs_by_stem(tmp_path):
    for name in ("b", "a"):
        _png(tmp_path / "hr" / f"{name}.png", 32, 32, 10 if name == "a" else 20)
        _png(tmp_path / "lr" / f"{name}.png", 16, 16, 10 if name == "a" else 20)
    ds = PairedDataset(str(tmp_path / "hr"), str(tmp_path / "lr"))
    assert len(ds) == 2
    for i in range(2):
        s = ds[i]
        assert s["hr"].shape == (3, 32, 32) and s["lr"].shape == (3, 16, 16)
        assert float(s["hr"][0, 0, 0]) == float(s["lr"][0, 0, 0])  # same stem, same value


@pytest.mark.parametrize("hr,lr", [(["a", "b"], ["a"]), (["a"], ["a", "b"])])
def test_paired_dataset_rejects_unmatched_stems(hr, lr, tmp_path):
    for name in hr:
        _png(tmp_path / "hr" / f"{name}.png")
    for name in lr:
        _png(tmp_path / "lr" / f"{name}.png")
    with pytest.raises(ValueError, match="stem mismatch"):
        PairedDataset(str(tmp_path / "hr"), str(tmp_path / "lr"))


def test_paired_dataset_rejects_duplicate_stems(tmp_path):
    _png(tmp_path / "hr" / "a.png")
    _png(tmp_path / "hr2" / "a.png")
    _png(tmp_path / "lr" / "a.png")
    with pytest.raises(ValueError, match="duplicate stem"):
        PairedDataset([str(tmp_path / "hr"), str(tmp_path / "hr2")], str(tmp_path / "lr"))


def test_vertical_line_noise_is_one_offset_per_column():
    random.seed(0)
    np.random.seed(0)
    out = add_vertical_line_noise(np.full((32, 4096, 3), 128, np.uint8), prob=1.0, strong_frac=0.0)
    assert (out == out[:1]).all()  # every row of a column carries the same offset
    assert 0.6 < out[0, :, 0].astype(np.float32).std() < 1.0  # measured sensor shape, sigma 0.75 DN


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_sr_dataset_returns_lr_and_hr_crops(scale, tmp_path):
    _png(tmp_path / "img.png", 200, 200)
    s = SRDataset(str(tmp_path), scale_factor=scale, patch_size=32, config={"clean_prob": 1.0})[0]
    assert s["lr"].shape == (3, 32, 32) and s["hr"].shape == (3, 32 * scale, 32 * scale)
