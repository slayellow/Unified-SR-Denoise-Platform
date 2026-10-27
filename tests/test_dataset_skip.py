"""SRDataset's and DenoiseDataset's handling of images smaller than the crop.

Undersized images are skipped rather than upscaled (an upscaled target is a bicubic upsample). The
replacement must be drawn at random: LSDIR stores files by source, so undersized images come in runs,
and stepping to the next file sent a whole run to one image (1,374 draws per Phase-1 epoch).
"""

from collections import Counter

import cv2
import numpy as np
import pytest

import src.data.datasets as d


def _dataset(root, small, large, monkeypatch):
    for i in range(small):
        cv2.imwrite(str(root / f"a{i}.png"), np.full((64, 64, 3), 100, np.uint8))
    for i in range(large):
        cv2.imwrite(str(root / f"b{i}.png"), np.full((128, 128, 3), 100, np.uint8))
    reads = []
    imread = cv2.imread
    monkeypatch.setattr(d.cv2, "imread", lambda p, *a: reads.append(p) or imread(p, *a))
    ds = d.SRDataset(str(root), scale_factor=2, patch_size=48, config={"clean_prob": 1.0})  # HR crop 96
    return ds, reads


def test_undersized_image_is_replaced_at_random(tmp_path, monkeypatch):
    ds, reads = _dataset(tmp_path, small=6, large=4, monkeypatch=monkeypatch)
    small = [i for i, p in enumerate(ds.image_paths) if "/a" in p]
    used = Counter()
    for seed in range(30):
        d.random.seed(seed)
        for i in small:
            reads.clear()
            ds[i]
            used[reads[-1].rsplit("/", 1)[1]] += 1
    assert set(used) == {"b0.png", "b1.png", "b2.png", "b3.png"}, used
    assert max(used.values()) < 0.4 * sum(used.values()), used


def test_single_large_image_is_always_found(tmp_path, monkeypatch):
    ds, reads = _dataset(tmp_path, small=9, large=1, monkeypatch=monkeypatch)
    for seed in range(50):
        d.random.seed(seed)
        reads.clear()
        ds[0]
        assert reads[-1].endswith("b0.png")


def test_no_image_large_enough_raises(tmp_path, monkeypatch):
    ds, _ = _dataset(tmp_path, small=3, large=0, monkeypatch=monkeypatch)
    with pytest.raises(RuntimeError):
        ds[0]


def test_denoise_dataset_skips_undersized_images(tmp_path, monkeypatch):
    """DenoiseDataset used to bicubic-upscale an undersized image, making the clean target an upsample."""
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"a{i}.png"), np.full((64, 64, 3), 100, np.uint8))
    cv2.imwrite(str(tmp_path / "b0.png"), np.full((128, 128, 3), 100, np.uint8))
    reads = []
    imread = cv2.imread
    monkeypatch.setattr(d.cv2, "imread", lambda p, *a: reads.append(p) or imread(p, *a))
    ds = d.DenoiseDataset(str(tmp_path), patch_size=96, config={"clean_prob": 1.0})
    small = [i for i, p in enumerate(ds.image_paths) if "/a" in p]
    for seed in range(20):
        d.random.seed(seed)
        for i in small:
            reads.clear()
            item = ds[i]
            assert reads[-1].endswith("b0.png")
            assert item["hr"].shape[-2:] == (96, 96)


def test_paired_dataset_names_an_unreadable_file(tmp_path):
    (tmp_path / "hr").mkdir()
    (tmp_path / "lr").mkdir()
    cv2.imwrite(str(tmp_path / "hr" / "x.png"), np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "lr" / "x.png").write_bytes(b"not a png")
    ds = d.PairedDataset(str(tmp_path / "hr"), str(tmp_path / "lr"))
    with pytest.raises(IOError, match="lr/x.png"):
        ds[0]
