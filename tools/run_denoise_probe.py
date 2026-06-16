#!/usr/bin/env python3
"""Denoise probe: Deploy vs Restormer/NAFNet Teachers on 42 real images."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from src.models import build_model


MODELS = {
    "deploy": {
        "config": "checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/train_config.yaml",
        "checkpoint": "checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/best.pth",
    },
    "restormer_teacher": {
        "config": "checkpoints/train_restormer_teacher_mc_g105_denoise_x1_dim48_tonesafe_lr1e4_260601/train_config.yaml",
        "checkpoint": "checkpoints/train_restormer_teacher_mc_g105_denoise_x1_dim48_tonesafe_lr1e4_260601/best.pth",
    },
    "nafnet_teacher": {
        "config": "checkpoints/train_nafnet_teacher_mc_g105_denoise_x1_width64_tonesafe_lr1e4_260605/train_config.yaml",
        "checkpoint": "checkpoints/train_nafnet_teacher_mc_g105_denoise_x1_width64_tonesafe_lr1e4_260605/best.pth",
    },
}


def load_model(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    model.eval()
    return model


def image_to_tensor(image_bgr, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    return tensor


def tensor_to_bgr(tensor):
    image = tensor.squeeze(0).detach().float().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def basic_metrics(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lap = cv2.Laplacian(image_bgr.astype(np.float32), cv2.CV_32F)
    return {
        "sharpness": float(np.var(lap)),
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
    }


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    input_root = Path("results/raw").resolve()
    output_root = Path("results/denoise_teacher_probe_analysis")

    if not input_root.exists():
        print(f"Error: Input directory not found: {input_root}")
        print(f"Current directory: {Path.cwd()}")
        return

    images = sorted([p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    if not images:
        print(f"Error: No images found in {input_root}")
        return

    print(f"Found {len(images)} images")

    # Load models
    print("\nLoading models...")
    models = {}
    for name, info in MODELS.items():
        print(f"  {name}...", end=" ", flush=True)
        config_path = PROJECT_ROOT / info["config"]
        checkpoint_path = PROJECT_ROOT / info["checkpoint"]
        if not config_path.exists() or not checkpoint_path.exists():
            print(f"SKIP (missing)")
            continue
        try:
            models[name] = load_model(config_path, checkpoint_path, device)
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")

    if not models:
        print("Error: No models loaded")
        return

    # Inference
    print(f"\nRunning inference on {len(images)} images x {len(models)} models...")
    rows = []
    start_time = time.time()

    with torch.no_grad():
        for idx, image_path in enumerate(images):
            rel_path = image_path.relative_to(input_root)
            raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                print(f"  [{idx+1}/{len(images)}] {rel_path}: LOAD FAILED")
                continue

            if (idx + 1) % max(1, len(images) // 10) == 0 or idx == len(images) - 1:
                print(f"  [{idx+1}/{len(images)}] {rel_path}")

            tensor = image_to_tensor(raw_bgr, device)
            row = {"relative_path": str(rel_path)}

            for model_name, model in models.items():
                try:
                    pred = model(tensor)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    output_bgr = tensor_to_bgr(pred)
                    if output_bgr.shape[:2] != raw_bgr.shape[:2]:
                        output_bgr = cv2.resize(output_bgr, (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

                    output_path = output_root / "outputs" / model_name / rel_path
                    save_image(output_path, output_bgr)

                    metrics = basic_metrics(output_bgr)
                    for k, v in metrics.items():
                        row[f"{model_name}_{k}"] = v
                except Exception as e:
                    print(f"    {model_name}: {e}")

            rows.append(row)

    # Save results
    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "metrics.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\nProcessed {len(images)} images x {len(models)} models in {elapsed:.1f}s")
    print(f"Output: {output_root}")
    print(f"Metrics: {output_root / 'metrics.csv'}")


if __name__ == "__main__":
    main()
