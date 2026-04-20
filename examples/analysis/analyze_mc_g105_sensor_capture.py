from __future__ import annotations
import argparse
import logging
import re
import sys
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import koreanize_matplotlib


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageAnalysisResult:
    """Container for per-image sensor statistics."""

    file_path: str
    scene_label: str
    time_label: str
    zoom_label: str
    width: int
    height: int
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    mean_y: float
    std_y: float
    mean_cr: float
    mean_cb: float
    tint_rg: float
    tint_bg: float
    edge_density: float
    high_freq_energy: float
    dark_region_ratio: float
    dark_noise_std: float
    hot_pixel_ratio: float
    hot_pixel_count: int
    temporal_noise: float = 0.0


def setup_logging(verbose: bool) -> None:
    """Configure logging for console output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze MC-G105 / VISCA output frame captures and summarize sensor-like artifacts."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing extracted frame images. Nested folders are supported.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mc_g105_analysis",
        help="Directory to save CSV, plots, and markdown summary.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional cap on the number of images to analyze. 0 means all images.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def collect_image_paths(input_dir: Path, max_images: int) -> List[Path]:
    """Collect image files recursively from the input directory."""
    image_paths = sorted(
        path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_images > 0:
        image_paths = image_paths[:max_images]
    return image_paths


def infer_time_label(path: Path) -> str:
    """Infer a coarse day/night label from path parts."""
    lowered_parts = [part.lower() for part in path.parts]
    if any("night" in part for part in lowered_parts):
        return "night"
    if any("day" in part for part in lowered_parts):
        return "day"
    return "unknown"


def infer_zoom_label(path: Path) -> str:
    """Infer zoom label like 1x/7x from file or directory names."""
    joined = " / ".join(path.parts).lower()
    match = re.search(r"(?<!\d)(\d{1,2})x(?!\d)", joined)
    if match:
        return f"{match.group(1)}x"
    return "unknown"

def group_bursts(image_paths: List[Path]) -> List[List[Path]]:
    """Group images into bursts based on directory and file prefixes."""
    bursts: Dict[Tuple[Path, str], List[Path]] = {}
    for p in image_paths:
        stem = p.stem
        # Pattern like frame_1401 where last digit is burst index
        prefix = re.sub(r'\d$', '', stem)
        key = (p.parent, prefix)
        if key not in bursts:
            bursts[key] = []
        bursts[key].append(p)
    return [sorted(v) for v in bursts.values()]


def infer_scene_label(path: Path) -> str:
    """Infer a scene label from the parent directory structure."""
    if len(path.parts) >= 2:
        return path.parent.name
    return "default"


def compute_hot_pixel_mask(rgb_image: np.ndarray) -> np.ndarray:
    """Estimate hot-pixel-like outliers from intensity and local neighborhood mismatch."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)
    local_median = cv2.medianBlur(gray, 3).astype(np.float32)
    residual = gray_f - local_median
    bright_threshold = np.percentile(gray_f, 99.8)
    residual_threshold = max(18.0, np.percentile(residual, 99.5))
    hot_mask = (gray_f >= bright_threshold) & (residual >= residual_threshold)
    return hot_mask


def compute_high_freq_energy(gray_image: np.ndarray) -> float:
    """Compute a simple normalized high-frequency energy score using Laplacian variance."""
    lap = cv2.Laplacian(gray_image, cv2.CV_32F)
    return float(np.var(lap))

def compute_temporal_noise(burst_paths: List[Path]) -> float:
    """Calculate average temporal standard deviation across a stack of images."""
    if len(burst_paths) <= 1:
        return 0.0
    stack = []
    for p in burst_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stack.append(img.astype(np.float32))
    if len(stack) <= 1:
        return 0.0
    stack_array = np.stack(stack, axis=0)
    std_map = np.std(stack_array, axis=0)
    return float(np.mean(std_map))


def analyze_image(image_path: Path) -> ImageAnalysisResult:
    """Analyze one image and return sensor-oriented summary statistics."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32)
    h, w = rgb.shape[:2]

    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    dark_mask = y < 40.0
    dark_region_ratio = float(np.mean(dark_mask))
    dark_noise_std = float(np.std(y[dark_mask])) if np.any(dark_mask) else 0.0

    hot_mask = compute_hot_pixel_mask(rgb)
    hot_pixel_count = int(np.sum(hot_mask))
    hot_pixel_ratio = float(hot_pixel_count / (h * w))

    result = ImageAnalysisResult(
        file_path=str(image_path),
        scene_label=infer_scene_label(image_path),
        time_label=infer_time_label(image_path),
        zoom_label=infer_zoom_label(image_path),
        width=w,
        height=h,
        mean_r=float(np.mean(r)),
        mean_g=float(np.mean(g)),
        mean_b=float(np.mean(b)),
        std_r=float(np.std(r)),
        std_g=float(np.std(g)),
        std_b=float(np.std(b)),
        mean_y=float(np.mean(y)),
        std_y=float(np.std(y)),
        mean_cr=float(np.mean(cr)),
        mean_cb=float(np.mean(cb)),
        tint_rg=float(np.mean(r) - np.mean(g)),
        tint_bg=float(np.mean(b) - np.mean(g)),
        edge_density=edge_density,
        high_freq_energy=compute_high_freq_energy(gray),
        dark_region_ratio=dark_region_ratio,
        dark_noise_std=dark_noise_std,
        hot_pixel_ratio=hot_pixel_ratio,
        hot_pixel_count=hot_pixel_count,
        temporal_noise=0.0
    )
    return result


def build_dataframe(results: Iterable[ImageAnalysisResult]) -> pd.DataFrame:
    """Convert analysis results into a pandas DataFrame."""
    return pd.DataFrame([asdict(result) for result in results])


def save_group_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Save grouped summary statistics by time/zoom labels."""
    group_cols = ["time_label", "zoom_label"]
    metric_cols = [
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "mean_y",
        "std_y",
        "tint_rg",
        "tint_bg",
        "edge_density",
        "high_freq_energy",
        "dark_region_ratio",
        "dark_noise_std",
        "hot_pixel_ratio",
        "hot_pixel_count",
        "temporal_noise",
    ]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    summary_path = output_dir / "group_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved grouped summary to %s", summary_path)
    return summary


def save_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a few compact plots for quick inspection."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, group in df.groupby("time_label"):
        plt.scatter(group["mean_y"], group["dark_noise_std"], label=label, alpha=0.7)
    plt.xlabel("Mean Y")
    plt.ylabel("Dark Region Noise Std")
    plt.title("Brightness vs Dark-Region Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "brightness_vs_dark_noise.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for label, group in df.groupby("zoom_label"):
        if group["temporal_noise"].sum() > 0:
            plt.scatter(group.index, group["temporal_noise"], label=label, alpha=0.7)
    plt.ylabel("Temporal Noise Std")
    plt.title("Burst Group Temporal Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "temporal_noise_analysis.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for label, group in df.groupby("zoom_label"):
        plt.scatter(group["edge_density"], group["hot_pixel_ratio"], label=label, alpha=0.7)
    plt.xlabel("Edge Density")
    plt.ylabel("Hot Pixel Ratio")
    plt.title("Edge Density vs Hot Pixel Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "edge_vs_hot_pixel.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(df["tint_rg"], bins=20, alpha=0.6, label="R-G")
    plt.hist(df["tint_bg"], bins=20, alpha=0.6, label="B-G")
    plt.xlabel("Tint Bias")
    plt.ylabel("Count")
    plt.title("Color Tint Bias Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "tint_distribution.png", dpi=150)
    plt.close()

    LOGGER.info("Saved plots to %s", plot_dir)


def render_summary_markdown(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Render a markdown summary for quick human review."""
    top_hot = df.sort_values("hot_pixel_count", ascending=False).head(5)
    lines = [
        "# MC-G105 Capture Analysis Summary",
        "",
        f"- Total images: {len(df)}",
        f"- Time labels: {sorted(df['time_label'].unique().tolist())}",
        f"- Zoom labels: {sorted(df['zoom_label'].unique().tolist())}",
        "",
        "## Group Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Top Hot-Pixel Candidates",
        "",
        top_hot[["file_path", "time_label", "zoom_label", "hot_pixel_count", "hot_pixel_ratio", "dark_noise_std", "tint_rg", "tint_bg"]].to_markdown(index=False),
        "",
        "## Interpretation Hints",
        "",
        "- `dark_noise_std`가 night/tele에서 높아지면 gain/저조도 노이즈 영향 가능성이 큼",
        "- `temporal_noise`가 높으면 실제 픽셀 단위의 시간적 변동성이 큼 (AWGN 시뮬레이션 지표)",
        "- `tint_rg`, `tint_bg`가 특정 방향으로 지속적으로 치우치면 고정 tint bias 가능성이 큼",
        "- `hot_pixel_count`가 dark scene에서 반복적으로 높으면 fixed defect pixel 가능성이 큼",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run frame analysis and save machine/human-readable outputs."""
    args = parse_args()
    setup_logging(args.verbose)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", input_dir)
        return 1

    image_paths = collect_image_paths(input_dir, args.max_images)
    if not image_paths:
        LOGGER.error("No image files found under %s", input_dir)
        return 1

    image_bursts = group_bursts(image_paths)
    LOGGER.info("Found %d images, grouped into %d bursts", len(image_paths), len(image_bursts))

    results: List[ImageAnalysisResult] = []
    for burst in image_bursts:
        t_noise = compute_temporal_noise(burst)
        for image_path in burst:
            try:
                res = analyze_image(image_path)
                res.temporal_noise = t_noise
                results.append(res)
            except Exception as exc:
                LOGGER.warning("Skipping %s due to error: %s", image_path, exc)

    if not results:
        LOGGER.error("No images were successfully analyzed.")
        return 1

    df = build_dataframe(results)
    per_image_path = output_dir / "per_image_metrics.csv"
    df.to_csv(per_image_path, index=False)
    LOGGER.info("Saved per-image metrics to %s", per_image_path)

    summary = save_group_summary(df, output_dir)
    save_plots(df, output_dir)

    summary_markdown = render_summary_markdown(df, summary)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_markdown, encoding="utf-8")
    LOGGER.info("Saved markdown summary to %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())