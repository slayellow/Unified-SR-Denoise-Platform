from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models import build_model  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run x4 SR real-sensor validation on center-cropped MC-G105 frames.")
    parser.add_argument("--input_root", default="results/260602_mc_g105_probe_42/raw")
    parser.add_argument("--output_root", default="results/sr_x4_baseline_analysis/real_sensor")
    parser.add_argument(
        "--config",
        default="checkpoints/csuav_deploy/finetune_svfocussrnet_eo_sr_x4_dim32_epoch100_bs_16_ga_2_lr_5e-5/train_config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/csuav_deploy/finetune_svfocussrnet_eo_sr_x4_dim32_epoch100_bs_16_ga_2_lr_5e-5/best.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop_w", type=int, default=320)
    parser.add_argument("--crop_h", type=int, default=180)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_iqa", action="store_true", help="Skip NIQE/BRISQUE/PIQE no-reference IQA metrics.")
    parser.add_argument("--iqa_device", default="cpu", help="Device for pyiqa no-reference metrics. Defaults to CPU.")
    return parser.parse_args()


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(normalize_state_dict(state_dict))

    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    model.eval()
    return model, config


def collect_images(input_root: Path, limit: int) -> list[Path]:
    images = sorted(path for path in input_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit > 0:
        images = images[:limit]
    return images


def center_crop(image: np.ndarray, crop_w: int, crop_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if crop_w > w or crop_h > h:
        raise ValueError(f"Crop {crop_w}x{crop_h} exceeds image size {w}x{h}.")
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()


def image_to_tensor(image_bgr: np.ndarray, device: torch.device, fp16: bool) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    if fp16 and device.type == "cuda":
        tensor = tensor.half()
    return tensor


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.squeeze(0).detach().float().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def to_gray_u8(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def metric_block(image: np.ndarray, prefix: str) -> dict[str, float]:
    gray = to_gray_u8(image)
    gray_f = gray.astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    edges = cv2.Canny(gray, 50, 150)
    low = cv2.GaussianBlur(gray_f, (0, 0), 3.0)
    high = gray_f - low
    local_mean = cv2.blur(gray_f, (31, 31))
    local_sq = cv2.blur(gray_f * gray_f, (31, 31))
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))

    edge_core = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    edge_band = cv2.dilate(edges, np.ones((9, 9), np.uint8), iterations=1) > 0
    ring_band = np.logical_and(edge_band, ~edge_core)
    if ring_band.any():
        ringing = float(np.mean(np.abs(high[ring_band])) * 255.0)
    else:
        ringing = 0.0

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    return {
        f"{prefix}_sharpness_laplacian_var": float(np.var(lap)),
        f"{prefix}_tenengrad": float(np.mean(sobel_mag * sobel_mag) * 255.0 * 255.0),
        f"{prefix}_edge_density": float(np.mean(edges > 0)),
        f"{prefix}_hf_energy": float(np.mean(np.abs(high)) * 255.0),
        f"{prefix}_ringing_proxy": ringing,
        f"{prefix}_local_contrast": float(np.mean(local_std) * 255.0),
        f"{prefix}_luma_mean": float(np.mean(ycrcb[:, :, 0])),
        f"{prefix}_luma_std": float(np.std(ycrcb[:, :, 0])),
        f"{prefix}_saturation_mean": float(np.mean(hsv[:, :, 1])),
    }


def ratio(numerator: float, denominator: float, min_denominator: float = 1e-8) -> float:
    if denominator < min_denominator:
        return float("nan")
    return float(numerator / denominator)


def pair_metrics(bicubic: np.ndarray, deploy: np.ndarray) -> dict[str, float]:
    bic_ycc = cv2.cvtColor(bicubic, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    dep_ycc = cv2.cvtColor(deploy, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    bic_y = bic_ycc[:, :, 0]
    dep_y = dep_ycc[:, :, 0]
    low_bic = cv2.GaussianBlur(bic_y, (0, 0), 15.0)
    low_dep = cv2.GaussianBlur(dep_y, (0, 0), 15.0)
    chroma_mae = np.mean(np.abs(dep_ycc[:, :, 1:] - bic_ycc[:, :, 1:]))
    rgb_mae = np.mean(np.abs(deploy.astype(np.float32) - bicubic.astype(np.float32)))
    diff = cv2.absdiff(deploy, bicubic)
    return {
        "deploy_minus_bicubic_luma_mean": float(np.mean(dep_y - bic_y)),
        "deploy_bicubic_lowfreq_luma_mae": float(np.mean(np.abs(low_dep - low_bic))),
        "deploy_bicubic_chroma_mae": float(chroma_mae),
        "deploy_bicubic_rgb_mae": float(rgb_mae),
        "deploy_bicubic_diff_p95": float(np.percentile(diff, 95)),
    }


class NoReferenceIqa:
    metric_names = ("niqe", "brisque", "piqe")

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled
        self.device = device
        self.metrics: dict[str, object] = {}
        self.errors: list[str] = []
        if not enabled:
            return

        if os.environ.get("HOME", "") in ("", "/"):
            os.environ["HOME"] = "/tmp"
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

        try:
            import pyiqa  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - depends on runtime package availability.
            self.errors.append(f"pyiqa import failed: {type(exc).__name__}: {exc}")
            return

        for name in self.metric_names:
            try:
                self.metrics[name] = pyiqa.create_metric(name, device=str(device))
            except Exception as exc:  # pragma: no cover - depends on runtime package availability.
                self.errors.append(f"{name} init failed: {type(exc).__name__}: {exc}")

    def compute(self, image: np.ndarray, prefix: str) -> dict[str, float]:
        scores = {f"{prefix}_{name}": float("nan") for name in self.metric_names}
        if not self.metrics:
            return scores

        tensor = image_to_tensor(image, self.device, fp16=False)
        with torch.no_grad():
            for name, metric in self.metrics.items():
                try:
                    value = metric(tensor)
                    scores[f"{prefix}_{name}"] = float(value.reshape(-1)[0].detach().cpu())
                except Exception as exc:  # pragma: no cover - metric-specific runtime behavior.
                    scores[f"{prefix}_{name}"] = float("nan")
                    self.errors.append(f"{name} compute failed: {type(exc).__name__}: {exc}")
        return scores


def classify_failures(row: dict[str, float]) -> tuple[list[str], int]:
    flags = []
    sharp_ratio = ratio(row["deploy_sharpness_laplacian_var"], row["bicubic_sharpness_laplacian_var"], 1.0)
    edge_ratio = ratio(row["deploy_edge_density"], row["bicubic_edge_density"], 0.002)
    hf_ratio = ratio(row["deploy_hf_energy"], row["bicubic_hf_energy"], 0.05)
    ring_ratio = ratio(row["deploy_ringing_proxy"], row["bicubic_ringing_proxy"], 0.05)
    contrast_ratio = ratio(row["deploy_local_contrast"], row["bicubic_local_contrast"], 0.05)

    if hf_ratio > 1.12 and (edge_ratio > 1.05 or row["deploy_edge_density"] - row["bicubic_edge_density"] > 0.01):
        flags.append("noise_or_false_texture_amplification")
    if sharp_ratio > 1.35 or ring_ratio > 1.18:
        flags.append("edge_oversharpening_or_ringing")
    if sharp_ratio < 0.85 and edge_ratio < 0.95:
        flags.append("edge_smoothing")
    if abs(row["deploy_minus_bicubic_luma_mean"]) > 2.0 or row["deploy_bicubic_chroma_mae"] > 2.0:
        flags.append("color_or_tone_instability")
    if contrast_ratio > 1.15:
        flags.append("local_contrast_amplification")
    if row["deploy_bicubic_diff_p95"] > 12.0:
        flags.append("large_local_deviation_from_bicubic")
    if not flags:
        flags.append("no_strong_failure_signal")
    return flags, sum(flag != "no_strong_failure_signal" for flag in flags)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    pad_h = 44
    canvas = np.full((out.shape[0] + pad_h, out.shape[1], 3), 245, dtype=np.uint8)
    canvas[pad_h:, :, :] = out
    cv2.putText(canvas, label, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def diff_heatmap(bicubic: np.ndarray, deploy: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(deploy, bicubic)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    colormap = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET)
    heat = cv2.applyColorMap(np.clip(gray * 4, 0, 255).astype(np.uint8), colormap)
    return heat


def make_comparison(input_crop: np.ndarray, bicubic: np.ndarray, deploy: np.ndarray) -> np.ndarray:
    h, w = bicubic.shape[:2]
    input_vis = cv2.resize(input_crop, (w, h), interpolation=cv2.INTER_NEAREST)
    heat = diff_heatmap(bicubic, deploy)
    panels = [
        add_label(input_vis, "Input crop 320x180 (nearest view)"),
        add_label(bicubic, "Bicubic x4"),
        add_label(deploy, "Deploy GPU x4"),
        add_label(heat, "Abs diff heatmap: Deploy vs Bicubic"),
    ]
    return np.concatenate(panels, axis=1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "sharpness_ratio",
        "edge_density_ratio",
        "hf_energy_ratio",
        "ringing_ratio",
        "local_contrast_ratio",
        "deploy_minus_bicubic_luma_mean",
        "deploy_bicubic_lowfreq_luma_mae",
        "deploy_bicubic_chroma_mae",
        "bicubic_niqe",
        "deploy_niqe",
        "deploy_minus_bicubic_niqe",
        "bicubic_brisque",
        "deploy_brisque",
        "deploy_minus_bicubic_brisque",
        "bicubic_piqe",
        "deploy_piqe",
        "deploy_minus_bicubic_piqe",
        "risk_score",
    ]
    return df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index().sort_values(group_cols)


def plot_failure_counts(failure_counts: Counter, output_path: Path) -> None:
    labels = [k for k, _ in failure_counts.most_common() if k != "no_strong_failure_signal"]
    values = [failure_counts[k] for k in labels]
    if not labels:
        labels = ["no_strong_failure_signal"]
        values = [failure_counts.get("no_strong_failure_signal", 0)]

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(labels) * 0.45)), facecolor="#FCFCFD")
    ax.set_facecolor("#FFFFFF")
    ax.barh(labels[::-1], values[::-1], color="#A3BEFA", edgecolor="#2E4780", linewidth=0.8)
    ax.set_xlabel("Frame count")
    ax.set_title("SR x4 real-sensor failure signal counts", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#E6E8F0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_overview(df: pd.DataFrame, comparisons_root: Path, output_path: Path, max_items: int = 8) -> list[str]:
    selected = df.sort_values(["risk_score", "deploy_bicubic_diff_p95"], ascending=False).head(max_items)
    thumbs = []
    used = []
    for _, row in selected.iterrows():
        comp = comparisons_root / row["comparison_path"]
        image = cv2.imread(str(comp), cv2.IMREAD_COLOR)
        if image is None:
            continue
        target_w = 1800
        scale = target_w / image.shape[1]
        target_h = max(1, int(image.shape[0] * scale))
        thumb = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
        used.append(row["relative_path"])

    if not thumbs:
        return []
    gap = 16
    width = max(t.shape[1] for t in thumbs)
    height = sum(t.shape[0] for t in thumbs) + gap * (len(thumbs) - 1)
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    y = 0
    for thumb in thumbs:
        canvas[y : y + thumb.shape[0], : thumb.shape[1]] = thumb
        y += thumb.shape[0] + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return used


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    preview = df.head(max_rows)
    try:
        return preview.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        def fmt(value: object) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value).replace("|", "\\|")

        headers = [str(col) for col in preview.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for _, row in preview.iterrows():
            lines.append("| " + " | ".join(fmt(row[col]) for col in preview.columns) + " |")
        return "\n".join(lines)


def mean_or_nan(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    return float(df[column].mean())


def fmt_float(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.3f}"


def write_report(
    output_root: Path,
    df: pd.DataFrame,
    summary_scene: pd.DataFrame,
    summary_zoom: pd.DataFrame,
    failure_counts: Counter,
    overview_items: list[str],
    args: argparse.Namespace,
    elapsed_infer: float,
) -> None:
    metrics_dir = output_root / "metrics"
    chart_rel = "metrics/failure_counts.png"
    overview_rel = "comparisons/overview_top_risk.jpg"
    per_image_csv = "metrics/per_image_metrics.csv"
    scene_csv = "metrics/summary_by_scene.csv"
    zoom_csv = "metrics/summary_by_zoom.csv"

    key = {
        "frames": int(len(df)),
        "crop": f"{args.crop_w}x{args.crop_h}",
        "output": f"{args.crop_w * args.scale}x{args.crop_h * args.scale}",
        "deploy_vram_note": "GPU inference was run batch-1 on the configured deploy x4 model.",
        "mean_hf_ratio": mean_or_nan(df, "hf_energy_ratio"),
        "mean_edge_ratio": mean_or_nan(df, "edge_density_ratio"),
        "mean_sharpness_ratio": mean_or_nan(df, "sharpness_ratio"),
        "mean_lowfreq_luma_mae": mean_or_nan(df, "deploy_bicubic_lowfreq_luma_mae"),
        "mean_chroma_mae": mean_or_nan(df, "deploy_bicubic_chroma_mae"),
        "mean_bicubic_niqe": mean_or_nan(df, "bicubic_niqe"),
        "mean_deploy_niqe": mean_or_nan(df, "deploy_niqe"),
        "mean_delta_niqe": mean_or_nan(df, "deploy_minus_bicubic_niqe"),
        "mean_bicubic_brisque": mean_or_nan(df, "bicubic_brisque"),
        "mean_deploy_brisque": mean_or_nan(df, "deploy_brisque"),
        "mean_delta_brisque": mean_or_nan(df, "deploy_minus_bicubic_brisque"),
        "mean_bicubic_piqe": mean_or_nan(df, "bicubic_piqe"),
        "mean_deploy_piqe": mean_or_nan(df, "deploy_piqe"),
        "mean_delta_piqe": mean_or_nan(df, "deploy_minus_bicubic_piqe"),
        "mean_risk_score": mean_or_nan(df, "risk_score"),
        "elapsed_infer_sec": elapsed_infer,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    failure_table = pd.DataFrame(
        [{"failure_signal": k, "frame_count": v} for k, v in failure_counts.most_common()]
    )
    top_risk = df.sort_values(["risk_score", "deploy_bicubic_diff_p95"], ascending=False)[
        [
            "relative_path",
            "scene",
            "zoom",
            "risk_score",
            "failure_labels",
            "hf_energy_ratio",
            "edge_density_ratio",
            "ringing_ratio",
            "deploy_niqe",
            "deploy_brisque",
            "deploy_piqe",
            "deploy_minus_bicubic_niqe",
            "deploy_minus_bicubic_brisque",
            "deploy_minus_bicubic_piqe",
            "deploy_minus_bicubic_luma_mean",
            "deploy_bicubic_lowfreq_luma_mae",
            "comparison_path",
        ]
    ]

    md = f"""# SR x4 Real Sensor 검증 - MC-G105 320x180 Crop

## 기술 요약

- 입력: `{args.input_root}`
- 샘플 수: `{key['frames']}`장
- Crop 기준: 원본 `1920x1080` frame의 중앙 `{key['crop']}`
- 출력 기준: Bicubic x4와 Deploy GPU x4 모두 `{key['output']}`
- Deploy checkpoint: `{args.checkpoint}`
- 실제 센서 입력에는 HR/GT가 없으므로 PSNR/SSIM은 계산하지 않고, no-reference IQA와 Deploy-vs-Bicubic proxy 지표 중심으로 판단한다.

## 전체 경향

| 지표 | 평균 |
|---|---:|
| Deploy / Bicubic sharpness ratio | {fmt_float(key['mean_sharpness_ratio'])} |
| Deploy / Bicubic edge density ratio | {fmt_float(key['mean_edge_ratio'])} |
| Deploy / Bicubic high-frequency ratio | {fmt_float(key['mean_hf_ratio'])} |
| Low-frequency luma MAE vs Bicubic | {fmt_float(key['mean_lowfreq_luma_mae'])} |
| Chroma MAE vs Bicubic | {fmt_float(key['mean_chroma_mae'])} |
| 평균 failure risk score | {fmt_float(key['mean_risk_score'])} |

## No-Reference IQA 요약

NIQE, BRISQUE, PIQE는 `pyiqa` 구현으로 계산했다. 세 지표 모두 낮을수록 좋은 방향으로 해석한다.

| 지표 | Bicubic 평균 | Deploy 평균 | Deploy - Bicubic |
|---|---:|---:|---:|
| NIQE | {fmt_float(key['mean_bicubic_niqe'])} | {fmt_float(key['mean_deploy_niqe'])} | {fmt_float(key['mean_delta_niqe'])} |
| BRISQUE | {fmt_float(key['mean_bicubic_brisque'])} | {fmt_float(key['mean_deploy_brisque'])} | {fmt_float(key['mean_delta_brisque'])} |
| PIQE | {fmt_float(key['mean_bicubic_piqe'])} | {fmt_float(key['mean_deploy_piqe'])} | {fmt_float(key['mean_delta_piqe'])} |

## Failure Signal Count

{dataframe_to_markdown(failure_table, max_rows=20)}

![Failure counts]({chart_rel})

## Top Risk Frame

{dataframe_to_markdown(top_risk, max_rows=12)}

## Scene별 요약

{dataframe_to_markdown(summary_scene, max_rows=30)}

## Zoom별 요약

{dataframe_to_markdown(summary_zoom, max_rows=30)}

## 시각 비교 Overview

![Top risk overview]({overview_rel})

포함 frame:

{chr(10).join(f'- `{path}`' for path in overview_items)}

## 산출 파일

- 전체 frame 지표: `{per_image_csv}`
- Scene별 요약: `{scene_csv}`
- Zoom별 요약: `{zoom_csv}`
- 입력 crop: `input_crop_320x180/`
- Bicubic x4: `bicubic/`
- Deploy GPU x4: `pred_deploy_gpu/`
- Frame별 비교 이미지: `comparisons/`

## 해석 및 한계

- Failure label은 절대 품질 판정이 아니라 screening용 threshold signal이다.
- `color_or_tone_instability`는 HR target이 없기 때문에 Deploy-vs-Bicubic low-frequency luma/chroma drift로 근사했다.
- `noise_or_false_texture_amplification`, `edge_oversharpening_or_ringing`은 high-frequency, edge density, sharpness, ringing proxy를 함께 사용해 분류했다.
- NIQE/BRISQUE/PIQE도 no-reference 지표라서 실제 임무 관점의 정답은 아니며, top-risk frame을 정성적으로 같이 확인해야 한다.
- SR 학습이나 MTKD 설계 변경 전에 본 보고서의 top-risk frame을 직접 확인해 domain mismatch, edge/texture hallucination, tone drift, model capacity 이슈를 분리하는 것이 좋다.
"""
    report_md = output_root / "report.md"
    report_md.write_text(md, encoding="utf-8")

    css = """
    body { font-family: Inter, Segoe UI, Arial, sans-serif; margin: 0; color: #1F2430; background: #FCFCFD; }
    main { max-width: 1180px; margin: 0 auto; padding: 32px 28px 56px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    h2 { margin-top: 34px; border-top: 1px solid #E6E8F0; padding-top: 24px; }
    p, li { line-height: 1.55; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }
    .card { background: #fff; border: 1px solid #E6E8F0; border-radius: 8px; padding: 14px; }
    .label { color: #6F768A; font-size: 12px; text-transform: uppercase; letter-spacing: .02em; }
    .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }
    th, td { border: 1px solid #E6E8F0; padding: 8px 10px; vertical-align: top; }
    th { background: #F4F5F7; text-align: left; }
    img { max-width: 100%; border: 1px solid #E6E8F0; background: #fff; }
    code { background: #F4F5F7; padding: 1px 4px; border-radius: 4px; }
    .muted { color: #6F768A; }
    """

    def html_table(data: pd.DataFrame, max_rows: int = 20) -> str:
        return data.head(max_rows).to_html(index=False, escape=True, float_format=lambda x: f"{x:.3f}")

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SR x4 Real Sensor 검증 - MC-G105</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>SR x4 Real Sensor 검증 - MC-G105</h1>
  <p class="muted">실제 raw frame 42장에서 중앙 {html.escape(key['crop'])} crop을 만들고, Bicubic x4와 Deploy GPU x4를 {html.escape(key['output'])} 기준으로 비교했다.</p>

  <h2>기술 요약</h2>
  <p>이 검증은 MC-G105 probe raw frame을 실제 센서 입력으로 사용한다. HR target이 없으므로 PSNR/SSIM 대신 no-reference IQA와 Deploy-vs-Bicubic drift 지표를 사용한다.</p>
  <div class="summary">
    <div class="card"><div class="label">Frames</div><div class="value">{key['frames']}</div></div>
    <div class="card"><div class="label">Mean HF Ratio</div><div class="value">{fmt_float(key['mean_hf_ratio'])}</div></div>
    <div class="card"><div class="label">Mean Edge Ratio</div><div class="value">{fmt_float(key['mean_edge_ratio'])}</div></div>
    <div class="card"><div class="label">Deploy NIQE</div><div class="value">{fmt_float(key['mean_deploy_niqe'])}</div></div>
    <div class="card"><div class="label">Deploy BRISQUE</div><div class="value">{fmt_float(key['mean_deploy_brisque'])}</div></div>
    <div class="card"><div class="label">Deploy PIQE</div><div class="value">{fmt_float(key['mean_deploy_piqe'])}</div></div>
    <div class="card"><div class="label">Mean Risk Score</div><div class="value">{fmt_float(key['mean_risk_score'])}</div></div>
  </div>

  <h2>No-Reference IQA 요약</h2>
  <p>NIQE, BRISQUE, PIQE는 <code>pyiqa</code>로 계산했다. 세 지표 모두 낮을수록 좋은 방향이다.</p>
  {html_table(pd.DataFrame([
      {'metric': 'NIQE', 'bicubic_mean': key['mean_bicubic_niqe'], 'deploy_mean': key['mean_deploy_niqe'], 'deploy_minus_bicubic': key['mean_delta_niqe']},
      {'metric': 'BRISQUE', 'bicubic_mean': key['mean_bicubic_brisque'], 'deploy_mean': key['mean_deploy_brisque'], 'deploy_minus_bicubic': key['mean_delta_brisque']},
      {'metric': 'PIQE', 'bicubic_mean': key['mean_bicubic_piqe'], 'deploy_mean': key['mean_deploy_piqe'], 'deploy_minus_bicubic': key['mean_delta_piqe']},
  ]), max_rows=10)}

  <h2>Failure Signal Count</h2>
  <p>전체 42장에 대한 threshold 기반 screening label이다. 한 frame에 여러 label이 동시에 붙을 수 있다.</p>
  <img src="{chart_rel}" alt="Failure counts chart">
  {html_table(failure_table, max_rows=20)}

  <h2>Top Risk 시각 증거</h2>
  <p>Overview 이미지는 screening score가 높은 frame을 모은 것이다. 각 row는 입력 crop, Bicubic x4, Deploy GPU x4, Deploy-vs-Bicubic diff heatmap 순서다.</p>
  <img src="{overview_rel}" alt="Top risk visual comparison overview">
  {html_table(top_risk, max_rows=12)}

  <h2>Scene/Zoom 요약</h2>
  <p>Scene과 zoom별 요약은 domain/content 민감도와 일반적인 모델 경향을 분리해서 보기 위한 것이다.</p>
  <h3>Scene별</h3>
  {html_table(summary_scene, max_rows=40)}
  <h3>Zoom별</h3>
  {html_table(summary_zoom, max_rows=40)}

  <h2>범위와 재현 정보</h2>
  <ul>
    <li>Input root: <code>{html.escape(args.input_root)}</code></li>
    <li>Deploy config: <code>{html.escape(args.config)}</code></li>
    <li>Deploy checkpoint: <code>{html.escape(args.checkpoint)}</code></li>
    <li>Per-image metrics: <code>{per_image_csv}</code></li>
    <li>Per-frame comparisons: <code>comparisons/</code></li>
  </ul>

  <h2>한계와 다음 판단</h2>
  <p>이 지표들은 진단용 proxy다. SR 학습이나 MTKD 설계를 바꾸기 전에 top-risk frame을 직접 확인해서 x4 이슈가 domain mismatch, edge/texture hallucination, tone drift, model capacity 중 어디에 가까운지 분리해야 한다.</p>
</main>
</body>
</html>
"""
    (output_root / "report.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    crop_root = output_root / f"input_crop_{args.crop_w}x{args.crop_h}"
    bicubic_root = output_root / "bicubic"
    deploy_root = output_root / "pred_deploy_gpu"
    comparisons_root = output_root / "comparisons"
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_root, args.limit)
    if not images:
        raise RuntimeError(f"No input images found under {input_root}")

    device = resolve_device(args.device)
    fp16 = args.fp16 and device.type == "cuda"
    model = None
    if not args.skip_inference:
        model, config = load_model(Path(args.config), Path(args.checkpoint), device)
        if int(config["model"].get("scale", args.scale)) != args.scale:
            raise ValueError(f"Config scale {config['model'].get('scale')} does not match requested x{args.scale}.")
        if fp16:
            model.half()
    iqa = NoReferenceIqa(enabled=not args.skip_iqa, device=resolve_device(args.iqa_device))
    if iqa.errors:
        print("IQA warnings:")
        for warning in iqa.errors[:5]:
            print(f"- {warning}")

    rows = []
    infer_start = time.time()
    with torch.no_grad():
        for image_path in images:
            rel = image_path.relative_to(input_root)
            rel_parts = rel.parts
            time_of_day = rel_parts[0] if len(rel_parts) > 0 else ""
            weather = rel_parts[1] if len(rel_parts) > 1 else ""
            scene = rel_parts[2] if len(rel_parts) > 2 else ""
            zoom = rel_parts[3] if len(rel_parts) > 3 else ""

            raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if raw is None:
                raise ValueError(f"Failed to read {image_path}")
            crop = center_crop(raw, args.crop_w, args.crop_h)
            bicubic = cv2.resize(crop, (args.crop_w * args.scale, args.crop_h * args.scale), interpolation=cv2.INTER_CUBIC)

            crop_path = crop_root / rel
            bicubic_path = bicubic_root / rel
            deploy_path = deploy_root / rel
            save_image(crop_path, crop)
            save_image(bicubic_path, bicubic)

            if args.skip_inference and deploy_path.exists():
                deploy = cv2.imread(str(deploy_path), cv2.IMREAD_COLOR)
            else:
                tensor = image_to_tensor(crop, device=device, fp16=fp16)
                output = model(tensor)
                if isinstance(output, tuple):
                    output = output[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                deploy = tensor_to_bgr(output)
                save_image(deploy_path, deploy)

            if deploy.shape[:2] != bicubic.shape[:2]:
                deploy = cv2.resize(deploy, (bicubic.shape[1], bicubic.shape[0]), interpolation=cv2.INTER_LINEAR)
                save_image(deploy_path, deploy)

            comparison = make_comparison(crop, bicubic, deploy)
            comparison_rel = rel.with_suffix(".compare.jpg")
            comparison_path = comparisons_root / comparison_rel
            save_image(comparison_path, comparison)

            row = {
                "relative_path": str(rel),
                "time_of_day": time_of_day,
                "weather": weather,
                "scene": scene,
                "zoom": zoom,
                "input_width": raw.shape[1],
                "input_height": raw.shape[0],
                "crop_width": args.crop_w,
                "crop_height": args.crop_h,
                "output_width": bicubic.shape[1],
                "output_height": bicubic.shape[0],
                "crop_path": str(crop_path.relative_to(output_root)),
                "bicubic_path": str(bicubic_path.relative_to(output_root)),
                "deploy_path": str(deploy_path.relative_to(output_root)),
                "comparison_path": str(comparison_path.relative_to(comparisons_root)),
            }
            row.update(metric_block(bicubic, "bicubic"))
            row.update(metric_block(deploy, "deploy"))
            row.update(pair_metrics(bicubic, deploy))
            row.update(iqa.compute(bicubic, "bicubic"))
            row.update(iqa.compute(deploy, "deploy"))
            for metric_name in NoReferenceIqa.metric_names:
                row[f"deploy_minus_bicubic_{metric_name}"] = (
                    row[f"deploy_{metric_name}"] - row[f"bicubic_{metric_name}"]
                )
            row["sharpness_ratio"] = ratio(
                row["deploy_sharpness_laplacian_var"], row["bicubic_sharpness_laplacian_var"], 1.0
            )
            row["edge_density_ratio"] = ratio(row["deploy_edge_density"], row["bicubic_edge_density"], 0.002)
            row["hf_energy_ratio"] = ratio(row["deploy_hf_energy"], row["bicubic_hf_energy"], 0.05)
            row["ringing_ratio"] = ratio(row["deploy_ringing_proxy"], row["bicubic_ringing_proxy"], 0.05)
            row["local_contrast_ratio"] = ratio(row["deploy_local_contrast"], row["bicubic_local_contrast"], 0.05)
            labels, risk_score = classify_failures(row)
            row["failure_labels"] = ";".join(labels)
            row["risk_score"] = risk_score
            rows.append(row)

    elapsed_infer = time.time() - infer_start
    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir / "per_image_metrics.csv", index=False)

    summary_scene = summarize_group(df, ["time_of_day", "weather", "scene"])
    summary_zoom = summarize_group(df, ["zoom"])
    summary_scene.to_csv(metrics_dir / "summary_by_scene.csv", index=False)
    summary_zoom.to_csv(metrics_dir / "summary_by_zoom.csv", index=False)

    failure_counts = Counter()
    for labels in df["failure_labels"]:
        failure_counts.update(labels.split(";"))
    plot_failure_counts(failure_counts, metrics_dir / "failure_counts.png")
    overview_items = make_overview(df, comparisons_root, comparisons_root / "overview_top_risk.jpg")
    write_report(output_root, df, summary_scene, summary_zoom, failure_counts, overview_items, args, elapsed_infer)

    print(f"Processed {len(df)} images")
    print(f"Output root: {output_root}")
    print(f"Report: {output_root / 'report.html'}")


if __name__ == "__main__":
    main()
