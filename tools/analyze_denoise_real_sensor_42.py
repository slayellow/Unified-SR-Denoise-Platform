from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, List

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


DEFAULT_MODELS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Deploy/Teacher denoise outputs on 42 real frames.")
    parser.add_argument("--input_root", default="results/raw")
    parser.add_argument("--output_root", default="results/denoise_teacher_probe_analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iqa_device", default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_iqa", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
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


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device, fp16: bool) -> tuple[torch.nn.Module, dict]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(normalize_state_dict(state_dict), strict=True)
    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    if fp16 and device.type == "cuda":
        model.half()
    model.eval()
    return model, config


def collect_images(input_root: Path, limit: int) -> list[Path]:
    images = sorted(path for path in input_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit > 0:
        images = images[:limit]
    return images


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


def gray_u8(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sobel_mag01(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    sx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sx * sx + sy * sy)


def high_freq(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    low = cv2.GaussianBlur(gray_f, (0, 0), sigma)
    return gray_f - low


def image_metrics(image: np.ndarray, prefix: str) -> dict[str, float]:
    gray = gray_u8(image)
    gray_f = gray.astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    mag = sobel_mag01(gray)
    edges = cv2.Canny(gray, 50, 150)
    hf = high_freq(gray)
    local_mean = cv2.blur(gray_f, (31, 31))
    local_sq = cv2.blur(gray_f * gray_f, (31, 31))
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))

    edge_core = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    edge_band = cv2.dilate(edges, np.ones((9, 9), np.uint8), iterations=1) > 0
    ring_band = np.logical_and(edge_band, ~edge_core)
    ringing = float(np.mean(np.abs(hf[ring_band])) * 255.0) if ring_band.any() else 0.0

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    return {
        f"{prefix}_sharpness_laplacian_var": float(np.var(lap)),
        f"{prefix}_tenengrad": float(np.mean(mag * mag) * 255.0 * 255.0),
        f"{prefix}_edge_density": float(np.mean(edges > 0)),
        f"{prefix}_hf_energy": float(np.mean(np.abs(hf)) * 255.0),
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


def pair_metrics(raw: np.ndarray, output: np.ndarray) -> dict[str, float]:
    raw_ycc = cv2.cvtColor(raw, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    out_ycc = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    raw_y = raw_ycc[:, :, 0]
    out_y = out_ycc[:, :, 0]
    raw_gray = gray_u8(raw)
    out_gray = gray_u8(output)
    raw_mag = sobel_mag01(raw_gray)
    out_mag = sobel_mag01(out_gray)
    raw_hf = high_freq(raw_gray)
    out_hf = high_freq(out_gray)

    flat_threshold = float(np.percentile(raw_mag, 30))
    flat_mask = raw_mag <= flat_threshold
    edge_threshold = float(np.percentile(raw_mag, 85))
    strong_edge_mask = raw_mag >= edge_threshold

    raw_flat_hf = float(np.mean(np.abs(raw_hf[flat_mask])) * 255.0) if flat_mask.any() else float("nan")
    out_flat_hf = float(np.mean(np.abs(out_hf[flat_mask])) * 255.0) if flat_mask.any() else float("nan")
    raw_edge_grad = float(np.mean(raw_mag[strong_edge_mask]) * 255.0) if strong_edge_mask.any() else float("nan")
    out_edge_grad = float(np.mean(out_mag[strong_edge_mask]) * 255.0) if strong_edge_mask.any() else float("nan")

    low_raw = cv2.GaussianBlur(raw_y, (0, 0), 15.0)
    low_out = cv2.GaussianBlur(out_y, (0, 0), 15.0)
    diff = cv2.absdiff(output, raw)
    return {
        "raw_output_rgb_mae": float(np.mean(np.abs(output.astype(np.float32) - raw.astype(np.float32)))),
        "raw_output_diff_p95": float(np.percentile(diff, 95)),
        "output_minus_raw_luma_mean": float(np.mean(out_y - raw_y)),
        "raw_output_lowfreq_luma_mae": float(np.mean(np.abs(low_out - low_raw))),
        "raw_output_chroma_mae": float(np.mean(np.abs(out_ycc[:, :, 1:] - raw_ycc[:, :, 1:]))),
        "flat_raw_hf": raw_flat_hf,
        "flat_output_hf": out_flat_hf,
        "flat_hf_ratio": ratio(out_flat_hf, raw_flat_hf, 0.02),
        "strong_edge_raw_grad": raw_edge_grad,
        "strong_edge_output_grad": out_edge_grad,
        "strong_edge_grad_ratio": ratio(out_edge_grad, raw_edge_grad, 0.02),
    }


class NoReferenceIqa:
    metric_names = ("niqe", "brisque", "piqe")

    def __init__(self, enabled: bool, device: torch.device) -> None:
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
        except Exception as exc:
            self.errors.append(f"pyiqa import failed: {type(exc).__name__}: {exc}")
            return
        for name in self.metric_names:
            try:
                self.metrics[name] = pyiqa.create_metric(name, device=str(device))
            except Exception as exc:
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
                except Exception as exc:
                    scores[f"{prefix}_{name}"] = float("nan")
                    self.errors.append(f"{name} compute failed: {type(exc).__name__}: {exc}")
        return scores


def failure_labels(row: dict[str, float]) -> tuple[str, int]:
    labels = []
    if row["flat_hf_ratio"] > 0.98:
        labels.append("noise_under_reduction")
    if row["flat_hf_ratio"] < 0.80 and row["strong_edge_grad_ratio"] < 0.92:
        labels.append("oversmoothing_risk")
    if row["strong_edge_grad_ratio"] > 1.08 or row["sharpness_ratio"] > 1.15 or row["ringing_ratio"] > 1.15:
        labels.append("oversharpening_or_ringing_risk")
    if abs(row["output_minus_raw_luma_mean"]) > 2.0 or row["raw_output_chroma_mae"] > 2.0:
        labels.append("tone_or_color_shift")
    if row["raw_output_diff_p95"] > 14.0:
        labels.append("large_raw_deviation")
    if not labels:
        labels.append("balanced")
    return ";".join(labels), sum(label != "balanced" for label in labels)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    pad_h = 42
    canvas = np.full((out.shape[0] + pad_h, out.shape[1], 3), 245, dtype=np.uint8)
    canvas[pad_h:, :, :] = out
    cv2.putText(canvas, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def make_comparison(raw: np.ndarray, outputs: dict[str, np.ndarray], model_names: Optional[List[str]] = None, max_panel_w: int = 620) -> np.ndarray:
    if model_names is None:
        model_names = list(outputs.keys())
    scale = max_panel_w / raw.shape[1]
    panel_h = max(1, int(raw.shape[0] * scale))
    panel_size = (max_panel_w, panel_h)
    panels = [add_label(cv2.resize(raw, panel_size, interpolation=cv2.INTER_AREA), "Raw input")]
    for name in model_names:
        if name in outputs:
            label = name.replace("_", " ").title()
            panels.append(add_label(cv2.resize(outputs[name], panel_size, interpolation=cv2.INTER_AREA), label))
    return np.concatenate(panels, axis=1)


def make_overview(df: pd.DataFrame, comparisons_root: Path, output_path: Path, max_items: int = 8) -> list[str]:
    if "stkd_minus_deploy_flat_hf_ratio" in df.columns:
        selected = df.sort_values(["stkd_minus_deploy_flat_hf_ratio", "stkd_minus_deploy_niqe"]).head(max_items)
    else:
        selected = df.head(max_items)
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
        thumbs.append(cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA))
        used.append(row["relative_path"])
    if not thumbs:
        return []
    gap = 16
    h = sum(t.shape[0] for t in thumbs) + gap * (len(thumbs) - 1)
    canvas = np.full((h, thumbs[0].shape[1], 3), 250, dtype=np.uint8)
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


def plot_model_bars(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = ["flat_hf_ratio", "strong_edge_grad_ratio", "niqe", "brisque", "piqe", "risk_score"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor="#FCFCFD")
    for ax, metric in zip(axes.flat, metrics):
        values = summary.set_index("model")[metric]
        ax.bar(values.index, values.values, color=["#A6A8B8", "#9CC6E8", "#9AD1B1"], edgecolor="#303846")
        ax.set_title(metric, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_model_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "niqe",
        "brisque",
        "piqe",
        "flat_hf_ratio",
        "strong_edge_grad_ratio",
        "sharpness_ratio",
        "edge_density_ratio",
        "hf_energy_ratio",
        "ringing_ratio",
        "raw_output_lowfreq_luma_mae",
        "raw_output_chroma_mae",
        "raw_output_diff_p95",
        "risk_score",
    ]
    return long_df.groupby("model", dropna=False)[cols].mean().reset_index()


def summarize_group(long_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = [
        "flat_hf_ratio",
        "strong_edge_grad_ratio",
        "niqe",
        "brisque",
        "piqe",
        "raw_output_lowfreq_luma_mae",
        "raw_output_chroma_mae",
        "risk_score",
    ]
    return long_df.groupby(group_cols, dropna=False)[cols].mean().reset_index().sort_values(group_cols)


def write_report(output_root: Path, long_df: pd.DataFrame, wide_df: pd.DataFrame, summary: pd.DataFrame, overview_items: list[str]) -> None:
    metrics_dir = output_root / "metrics"
    summary_dict = {
        "frames": int(wide_df.shape[0]),
        "models": summary.to_dict(orient="records"),
        "checkpoint_mode": "best.pth at analysis time",
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    comparison_cols = [
        "relative_path",
        "time_of_day",
        "weather",
        "scene",
        "zoom",
        "deploy_flat_hf_ratio",
        "mtkd_flat_hf_ratio",
        "stkd_flat_hf_ratio",
        "deploy_strong_edge_grad_ratio",
        "mtkd_strong_edge_grad_ratio",
        "stkd_strong_edge_grad_ratio",
        "deploy_niqe",
        "mtkd_niqe",
        "stkd_niqe",
        "mtkd_minus_deploy_flat_hf_ratio",
        "stkd_minus_deploy_flat_hf_ratio",
        "mtkd_minus_deploy_niqe",
        "stkd_minus_deploy_niqe",
        "comparison_path",
    ]
    top = wide_df.sort_values(["stkd_minus_deploy_flat_hf_ratio", "stkd_minus_deploy_niqe"]).head(12)[comparison_cols]
    flags = Counter()
    for labels in long_df["failure_labels"]:
        flags.update(str(labels).split(";"))
    flag_df = pd.DataFrame([{"failure_signal": k, "frame_count": v} for k, v in flags.most_common()])

    md = f"""# Denoise Real Sensor 42장 KD 효과 분석

## 분석 기준

- 입력: `results/260602_mc_g105_probe_42/raw`
- 샘플 수: `{wide_df.shape[0]}`장
- 비교 모델: Deploy, MTKD, STKD
- checkpoint 기준: 분석 시점의 각 모델 `best.pth`
- 실제 센서 입력에는 HR/GT가 없으므로 PSNR/SSIM/LPIPS 대신 no-reference IQA와 Raw-vs-Output proxy를 사용했다.

## 모델별 요약

낮을수록 좋은 지표: NIQE, BRISQUE, PIQE, flat_hf_ratio, raw_output_lowfreq_luma_mae, raw_output_chroma_mae, risk_score.

`flat_hf_ratio`는 Raw의 평탄 영역 고주파 대비 출력의 평탄 영역 고주파 비율이다. 낮을수록 평탄 영역 noise-like HF가 줄었다는 뜻이다.

`strong_edge_grad_ratio`는 Raw의 강한 edge 위치에서 출력 edge 강도가 얼마나 유지되는지를 보는 값이다. 1에 가까울수록 edge 보존, 너무 낮으면 smoothing, 너무 높으면 sharpening 가능성이 있다.

{dataframe_to_markdown(summary, max_rows=10)}

![Model summary](metrics/model_summary_bars.png)

## Failure Signal Count

{dataframe_to_markdown(flag_df, max_rows=20)}

## STKD 개선 폭이 큰 Frame 예시

{dataframe_to_markdown(top, max_rows=12)}

## 시각 비교 Overview

![Overview](comparisons/overview_stkd_improvement.jpg)

포함 frame:

{chr(10).join(f'- `{path}`' for path in overview_items)}

## 산출 파일

- 모델별 long metrics: `metrics/per_model_metrics.csv`
- frame별 wide metrics: `metrics/per_frame_comparison.csv`
- scene별 요약: `metrics/summary_by_scene_model.csv`
- zoom별 요약: `metrics/summary_by_zoom_model.csv`
- 출력 이미지: `outputs/{{deploy,mtkd,stkd}}/`
- frame별 비교 이미지: `comparisons/`

## 해석 주의

- 이 분석은 real input 기반이라 GT 정답 비교가 아니다.
- NIQE/BRISQUE/PIQE는 자연 영상 품질 proxy이며, 임무 장비에서의 edge fidelity를 완전히 대변하지 않는다.
- 최종 판단은 flat noise 감소, edge 유지, tone/color shift, 비교 이미지를 함께 봐야 한다.
"""
    (output_root / "report.md").write_text(md, encoding="utf-8")

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Denoise Real Sensor 42 KD Analysis</title>
<style>
body {{ font-family: Inter, Segoe UI, Arial, sans-serif; margin: 0; color: #1F2430; background: #FCFCFD; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 56px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
h2 {{ margin-top: 34px; border-top: 1px solid #E6E8F0; padding-top: 24px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
th, td {{ border: 1px solid #E6E8F0; padding: 8px 10px; vertical-align: top; }}
th {{ background: #F4F5F7; text-align: left; }}
img {{ max-width: 100%; border: 1px solid #E6E8F0; background: #fff; }}
code {{ background: #F4F5F7; padding: 1px 4px; border-radius: 4px; }}
.muted {{ color: #6F768A; }}
</style></head><body><main>
<h1>Denoise Real Sensor 42장 KD 효과 분석</h1>
<p class="muted">Deploy, MTKD, STKD를 동일한 42장 real sensor raw input에 적용해 no-reference IQA와 Raw-vs-Output proxy를 비교했다.</p>
<h2>모델별 요약</h2>
{summary.to_html(index=False, escape=True, float_format=lambda x: f"{x:.3f}")}
<img src="metrics/model_summary_bars.png" alt="Model summary bars">
<h2>STKD 개선 폭이 큰 Frame 예시</h2>
{top.to_html(index=False, escape=True, float_format=lambda x: f"{x:.3f}")}
<h2>시각 비교 Overview</h2>
<img src="comparisons/overview_stkd_improvement.jpg" alt="Overview">
<h2>산출 파일</h2>
<ul>
<li><code>metrics/per_model_metrics.csv</code></li>
<li><code>metrics/per_frame_comparison.csv</code></li>
<li><code>outputs/{{deploy,mtkd,stkd}}/</code></li>
<li><code>comparisons/</code></li>
</ul>
</main></body></html>
"""
    (output_root / "report.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    outputs_root = output_root / "outputs"
    comparisons_root = output_root / "comparisons"
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_root, args.limit)
    if not images:
        raise RuntimeError(f"No images found under {input_root}")

    device = resolve_device(args.device)
    fp16 = args.fp16 and device.type == "cuda"
    iqa = NoReferenceIqa(enabled=not args.skip_iqa, device=resolve_device(args.iqa_device))
    if iqa.errors:
        print("IQA warnings:")
        for warning in iqa.errors[:5]:
            print(f"- {warning}")

    model_infos = {
        name: {"config": Path(info["config"]), "checkpoint": Path(info["checkpoint"])}
        for name, info in DEFAULT_MODELS.items()
    }
    for name, info in model_infos.items():
        if not info["config"].exists():
            raise FileNotFoundError(f"{name} config not found: {info['config']}")
        if not info["checkpoint"].exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {info['checkpoint']}")

    raw_metric_cache: dict[str, dict[str, float]] = {}
    output_images_by_rel: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    start_time = time.time()

    for model_name, info in model_infos.items():
        print(f"Loading {model_name}: {info['checkpoint']}")
        model = None
        if not args.skip_inference:
            model, _ = load_model(info["config"], info["checkpoint"], device, fp16)

        with torch.no_grad():
            for image_path in images:
                rel = image_path.relative_to(input_root)
                rel_str = str(rel)
                rel_parts = rel.parts
                time_of_day = rel_parts[0] if len(rel_parts) > 0 else ""
                weather = rel_parts[1] if len(rel_parts) > 1 else ""
                scene = rel_parts[2] if len(rel_parts) > 2 else ""
                zoom = rel_parts[3] if len(rel_parts) > 3 else ""
                raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if raw is None:
                    raise ValueError(f"Failed to read {image_path}")

                output_path = outputs_root / model_name / rel
                if args.skip_inference and output_path.exists():
                    output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
                else:
                    tensor = image_to_tensor(raw, device, fp16)
                    pred = model(tensor)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    output = tensor_to_bgr(pred)
                    if output.shape[:2] != raw.shape[:2]:
                        output = cv2.resize(output, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
                    save_image(output_path, output)

                output_images_by_rel.setdefault(rel_str, {})[model_name] = output

                if rel_str not in raw_metric_cache:
                    raw_metrics = image_metrics(raw, "raw")
                    raw_metrics.update(iqa.compute(raw, "raw"))
                    raw_metric_cache[rel_str] = raw_metrics
                raw_metrics = raw_metric_cache[rel_str]
                out_metrics = image_metrics(output, "output")
                pair = pair_metrics(raw, output)
                row = {
                    "relative_path": rel_str,
                    "time_of_day": time_of_day,
                    "weather": weather,
                    "scene": scene,
                    "zoom": zoom,
                    "model": model_name,
                    "config": str(info["config"]),
                    "checkpoint": str(info["checkpoint"]),
                    "output_path": str(output_path.relative_to(output_root)),
                }
                row.update(raw_metrics)
                row.update(out_metrics)
                row.update(pair)
                row.update(iqa.compute(output, "output"))
                for metric_name in NoReferenceIqa.metric_names:
                    row[metric_name] = row[f"output_{metric_name}"]
                    row[f"output_minus_raw_{metric_name}"] = row[f"output_{metric_name}"] - row[f"raw_{metric_name}"]
                row["sharpness_ratio"] = ratio(row["output_sharpness_laplacian_var"], row["raw_sharpness_laplacian_var"], 1.0)
                row["edge_density_ratio"] = ratio(row["output_edge_density"], row["raw_edge_density"], 0.002)
                row["hf_energy_ratio"] = ratio(row["output_hf_energy"], row["raw_hf_energy"], 0.05)
                row["ringing_ratio"] = ratio(row["output_ringing_proxy"], row["raw_ringing_proxy"], 0.05)
                labels, risk = failure_labels(row)
                row["failure_labels"] = labels
                row["risk_score"] = risk
                rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Build comparison images once all model outputs are ready.
    comparison_paths = {}
    model_names = list(model_infos.keys())
    for image_path in images:
        rel = image_path.relative_to(input_root)
        rel_str = str(rel)
        raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        comp = make_comparison(raw, output_images_by_rel[rel_str], model_names=model_names)
        comp_path = comparisons_root / rel.with_suffix(".compare.jpg")
        save_image(comp_path, comp)
        comparison_paths[rel_str] = str(comp_path.relative_to(comparisons_root))

    long_df = pd.DataFrame(rows)
    long_df["comparison_path"] = long_df["relative_path"].map(comparison_paths)
    long_df.to_csv(metrics_dir / "per_model_metrics.csv", index=False)

    # Wide frame table for deploy-vs-KD deltas.
    base_cols = ["relative_path", "time_of_day", "weather", "scene", "zoom", "comparison_path"]
    metric_cols = [
        "niqe",
        "brisque",
        "piqe",
        "flat_hf_ratio",
        "strong_edge_grad_ratio",
        "sharpness_ratio",
        "edge_density_ratio",
        "hf_energy_ratio",
        "ringing_ratio",
        "raw_output_lowfreq_luma_mae",
        "raw_output_chroma_mae",
        "raw_output_diff_p95",
        "risk_score",
        "failure_labels",
    ]
    frames = []
    for rel, group in long_df.groupby("relative_path"):
        base = group.iloc[0][base_cols].to_dict()
        for _, row in group.iterrows():
            prefix = row["model"]
            for col in metric_cols:
                base[f"{prefix}_{col}"] = row[col]
        for model_name in model_names:
            if model_name != "deploy" and "deploy" in base:
                for col in metric_cols:
                    if col == "failure_labels":
                        continue
                    base[f"{model_name}_minus_deploy_{col}"] = base.get(f"{model_name}_{col}", float("nan")) - base.get(f"deploy_{col}", float("nan"))
        frames.append(base)
    wide_df = pd.DataFrame(frames)
    wide_df.to_csv(metrics_dir / "per_frame_comparison.csv", index=False)

    summary = summarize_model_metrics(long_df)
    summary.to_csv(metrics_dir / "summary_by_model.csv", index=False)
    summarize_group(long_df, ["model", "time_of_day", "weather", "scene"]).to_csv(
        metrics_dir / "summary_by_scene_model.csv", index=False
    )
    summarize_group(long_df, ["model", "zoom"]).to_csv(metrics_dir / "summary_by_zoom_model.csv", index=False)
    plot_model_bars(summary, metrics_dir / "model_summary_bars.png")
    overview_items = make_overview(wide_df, comparisons_root, comparisons_root / "overview_stkd_improvement.jpg")
    write_report(output_root, long_df, wide_df, summary, overview_items)

    print(f"Processed {len(images)} images x {len(model_infos)} models")
    print(f"Elapsed sec: {time.time() - start_time:.1f}")
    print(f"Output root: {output_root}")
    print(f"Report: {output_root / 'report.html'}")


if __name__ == "__main__":
    main()
