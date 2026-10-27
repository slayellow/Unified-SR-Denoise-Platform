"""Shared checkpoint loading for the tools.

Training wrappers leave prefixes on parameter names: DistributedDataParallel
adds "module." and torch.compile adds "_orig_mod.". A checkpoint saved from a
wrapped model then fails to load into the bare architecture. Every tool that
loads a checkpoint must strip these the same way.
"""
import torch

from src.models import build_model


def normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def check_model_config(checkpoint, model_cfg, path):
    """Raise ValueError when the checkpoint was trained with another model config than model_cfg.

    A strict load does not catch every mismatch: achromatic_output has no weights, so an IR
    checkpoint opened with the EO config loads cleanly and then puts colour on grey input.
    A checkpoint that saved no config (older runs) is not checked.
    """
    saved = (checkpoint.get("config") or {}).get("model") if isinstance(checkpoint, dict) else None
    if saved is None or saved == model_cfg:
        return
    keys = sorted(k for k in set(saved) | set(model_cfg) if saved.get(k) != model_cfg.get(k))
    raise ValueError(
        f"{path} was trained with another model config; differing keys: "
        + ", ".join(f"{k} (checkpoint {saved.get(k)!r}, --config {model_cfg.get(k)!r})" for k in keys)
    )


def load_model(path, model_cfg, device):
    """Checkpoint -> model in deploy mode (switch_to_deploy) and eval, for tools/evaluate, inference and export.

    A whole pickled nn.Module (2_compress_svd_pruning.py saves one) is used as it is. Otherwise the
    checkpoint must match model_cfg (check_model_config) and loads strictly.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        print("[Info] Loaded full model object.")
        model = checkpoint.to(device)
    else:
        check_model_config(checkpoint, model_cfg, path)
        model = build_model(model_cfg).to(device)
        model.load_state_dict(normalize_state_dict(checkpoint.get("model_state_dict", checkpoint)))
    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    return model.eval()
