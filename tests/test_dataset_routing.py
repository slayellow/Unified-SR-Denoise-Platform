"""resolve_dataset_kind: the task alone routes, loudly.

dataset_type used to override task, which let a denoise config claim task: sr
and still get the right dataset. An unknown task now raises instead of
silently becoming 'sr'.
"""
import glob

import pytest
import yaml

from tools.train import resolve_dataset_kind


def test_sr_and_denoise_route_by_task():
    assert resolve_dataset_kind({"task": "sr"}) == "sr"
    assert resolve_dataset_kind({"task": "denoise"}) == "denoise"
    assert resolve_dataset_kind({"task": "SR"}) == "sr"  # case-insensitive


def test_unknown_or_missing_task_raises():
    with pytest.raises(ValueError, match="unknown task"):
        resolve_dataset_kind({"task": "guide"})
    with pytest.raises(ValueError, match="unknown task"):
        resolve_dataset_kind({})


def test_every_shipped_config_routes():
    """No config in the repo may hit the new raise, and denoise configs must
    actually say so now that dataset_type no longer covers for them."""
    for f in glob.glob("configs/train/**/*.yaml", recursive=True) + \
             glob.glob("configs/finetune/**/*.yaml", recursive=True):
        cfg = yaml.safe_load(open(f, encoding="utf-8"))
        if not isinstance(cfg, dict) or "task" not in cfg:
            continue
        kind = resolve_dataset_kind(cfg)
        assert kind in ("sr", "denoise"), f
        if "/Denoise/" in f.replace("\\", "/"):
            assert kind == "denoise", f"{f} routes to {kind}"
