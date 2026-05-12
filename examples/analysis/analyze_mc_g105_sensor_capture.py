from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    koreanize_matplotlib = None


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EPS = 1e-6

TIME_LABELS = {"morning", "afternoon", "day", "night"}
WEATHER_LABELS = {"sunny", "cloudy", "clear", "overcast", "indoor", "unknown"}
FLAT_SCENES = {"flat", "white_wall", "whitewall"}
DARK_SCENES = {"dark", "dark_frame", "lens_covered"}
OUTDOOR_SCENES = {"building", "mixed", "road_shaded", "road", "urban"}

SUMMARY_METRICS = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "mean_y",
    "std_y",
    "mean_cr",
    "mean_cb",
    "std_cr",
    "std_cb",
    "tint_rg",
    "tint_bg",
    "g_over_r",
    "g_over_b",
    "green_excess",
    "green_excess_norm",
    "chroma_mottle_std",
    "edge_density",
    "high_freq_energy",
    "dark_region_ratio",
    "dark_noise_std",
    "hot_pixel_ratio",
    "hot_pixel_count",
    "y_clip_low_ratio",
    "y_clip_high_ratio",
    "rgb_clip_low_ratio",
    "rgb_clip_high_ratio",
    "flat_shading_range_norm",
    "flat_center_corner_y_ratio",
    "flat_center_g_over_r",
    "flat_center_g_over_b",
    "burst_temporal_noise_mean",
    "burst_temporal_noise_p90",
    "burst_mean_y_range",
    "burst_green_excess_range",
    "burst_high_freq_range",
    "hot_persistence_ratio_25",
    "hot_persistence_ratio_50",
    "hot_persistence_ratio_75",
]


@dataclass
class ImageMetadata:
    file_path: str
    relative_path: str
    time_label: str
    weather_label: str
    scene_label: str
    scene_kind: str
    zoom_label: str
    frame_index: str
    frame_number: int
    burst_id: str


@dataclass
class ImageAnalysisResult:
    """Per-image sensor/proxy statistics."""

    file_path: str
    relative_path: str
    time_label: str
    weather_label: str
    scene_label: str
    scene_kind: str
    zoom_label: str
    frame_index: str
    frame_number: int
    burst_id: str
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
    std_cr: float
    std_cb: float
    tint_rg: float
    tint_bg: float
    g_over_r: float
    g_over_b: float
    green_excess: float
    green_excess_norm: float
    chroma_mottle_std: float
    edge_density: float
    high_freq_energy: float
    dark_region_ratio: float
    dark_noise_std: float
    hot_pixel_ratio: float
    hot_pixel_count: int
    y_clip_low_ratio: float
    y_clip_high_ratio: float
    rgb_clip_low_ratio: float
    rgb_clip_high_ratio: float
    r_clip_high_ratio: float
    g_clip_high_ratio: float
    b_clip_high_ratio: float
    flat_shading_range_norm: float
    flat_center_y: float
    flat_corner_y: float
    flat_center_corner_y_ratio: float
    flat_center_g_over_r: float
    flat_center_g_over_b: float


@dataclass
class BurstAnalysisResult:
    """Burst-level temporal, stability, and persistence statistics."""

    burst_id: str
    time_label: str
    weather_label: str
    scene_label: str
    scene_kind: str
    zoom_label: str
    frame_count: int
    width: int
    height: int
    burst_temporal_noise_mean: float
    burst_temporal_noise_median: float
    burst_temporal_noise_p90: float
    burst_temporal_noise_max: float
    burst_mean_y_std: float
    burst_mean_y_range: float
    burst_mean_y_drift: float
    burst_g_over_r_range: float
    burst_g_over_b_range: float
    burst_green_excess_range: float
    burst_edge_density_range: float
    burst_high_freq_range: float
    burst_high_freq_rel_range: float
    unstable_luma: bool
    unstable_color: bool
    unstable_detail: bool
    unstable_auto: bool
    hot_persistence_count_25: int
    hot_persistence_count_50: int
    hot_persistence_count_75: int
    hot_persistence_ratio_25: float
    hot_persistence_ratio_50: float
    hot_persistence_ratio_75: float
    map_temporal_std: str
    map_hot_persistence: str
    map_hot_fixed_mask: str
    map_flat_shading: str
    map_flat_green_excess: str


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MC-G105 NV12-derived frame captures using folder metadata. "
            "Expected layout: {time}/{weather}/{scene}/{zoom}/{frame}.png"
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing captured images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mc_g105_analysis",
        help="Directory to save CSV, plots, maps, and markdown summary.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional cap on the number of images to analyze. 0 means all images.",
    )
    parser.add_argument(
        "--previous_group_summary",
        type=str,
        default="",
        help="Optional group_summary.csv from an earlier run for current-vs-previous deltas.",
    )
    parser.add_argument(
        "--no_maps",
        action="store_true",
        help="Disable temporal/hot-pixel/flat heatmap image outputs.",
    )
    parser.add_argument(
        "--save_outdoor_maps",
        action="store_true",
        help="Also save maps for outdoor bursts. By default maps are saved for flat/dark only.",
    )
    parser.add_argument(
        "--unstable_y_range",
        type=float,
        default=12.0,
        help="Burst mean-Y range above this value is flagged as auto-exposure unstable.",
    )
    parser.add_argument(
        "--unstable_color_range",
        type=float,
        default=8.0,
        help="Burst green-excess range above this value is flagged as color unstable.",
    )
    parser.add_argument(
        "--unstable_detail_rel_range",
        type=float,
        default=0.35,
        help="Relative high-frequency range above this value is flagged as detail/focus unstable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def normalize_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + EPS))


def collect_image_paths(input_dir: Path, max_images: int) -> list[Path]:
    image_paths = sorted(
        path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_images > 0:
        image_paths = image_paths[:max_images]
    return image_paths


def infer_time_label(parts: Iterable[str]) -> str:
    lowered = [normalize_label(part) for part in parts]
    for label in lowered:
        if label in TIME_LABELS:
            return label
    if any("morning" in label for label in lowered):
        return "morning"
    if any("afternoon" in label for label in lowered):
        return "afternoon"
    if any("night" in label for label in lowered):
        return "night"
    if any("day" in label for label in lowered):
        return "day"
    return "unknown"


def infer_weather_label(parts: Iterable[str]) -> str:
    lowered = [normalize_label(part) for part in parts]
    for label in lowered:
        if label in WEATHER_LABELS:
            return label
    return "unknown"


def infer_zoom_label(parts: Iterable[str]) -> str:
    joined = "/".join(parts).lower()
    match = re.search(r"(?<!\d)(\d{1,2})x(?!\d)", joined)
    if match:
        return f"{match.group(1)}x"
    return "unknown"


def infer_scene_kind(scene_label: str) -> str:
    scene = normalize_label(scene_label)
    if scene in FLAT_SCENES:
        return "calibration_flat"
    if scene in DARK_SCENES:
        return "calibration_dark"
    if scene in OUTDOOR_SCENES:
        return "outdoor"
    return "unknown"


def infer_frame_number(frame_index: str) -> int:
    match = re.search(r"(\d+)$", frame_index)
    if match:
        return int(match.group(1))
    return -1


def parse_metadata(image_path: Path, input_dir: Path) -> ImageMetadata:
    rel_path = image_path.relative_to(input_dir)
    parts = list(rel_path.parts)
    normalized_parts = [normalize_label(part) for part in parts]

    frame_index = image_path.stem
    frame_number = infer_frame_number(frame_index)

    if len(parts) >= 5:
        time_label = normalize_label(parts[0])
        weather_label = normalize_label(parts[1])
        scene_label = normalize_label(parts[2])
        zoom_label = normalize_label(parts[3])
    else:
        time_label = infer_time_label(normalized_parts)
        weather_label = infer_weather_label(normalized_parts)
        zoom_label = infer_zoom_label(normalized_parts)
        scene_label = normalize_label(image_path.parent.name)
        if scene_label == zoom_label and image_path.parent.parent != image_path.parent:
            scene_label = normalize_label(image_path.parent.parent.name)

    if time_label not in TIME_LABELS:
        inferred_time = infer_time_label(normalized_parts)
        time_label = inferred_time if inferred_time != "unknown" else time_label
    if weather_label not in WEATHER_LABELS:
        inferred_weather = infer_weather_label(normalized_parts)
        weather_label = inferred_weather if inferred_weather != "unknown" else weather_label
    if not re.fullmatch(r"\d{1,2}x", zoom_label):
        zoom_label = infer_zoom_label(normalized_parts)

    burst_id = str(rel_path.parent).replace("\\", "/")
    scene_kind = infer_scene_kind(scene_label)

    return ImageMetadata(
        file_path=str(image_path),
        relative_path=str(rel_path).replace("\\", "/"),
        time_label=time_label,
        weather_label=weather_label,
        scene_label=scene_label,
        scene_kind=scene_kind,
        zoom_label=zoom_label,
        frame_index=frame_index,
        frame_number=frame_number,
        burst_id=burst_id,
    )


def read_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_hot_pixel_mask_from_gray(gray: np.ndarray) -> np.ndarray:
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    gray_f = gray_u8.astype(np.float32)
    local_median = cv2.medianBlur(gray_u8, 3).astype(np.float32)
    residual = gray_f - local_median

    bright_threshold = max(
        float(np.percentile(gray_f, 99.8)),
        float(np.mean(gray_f) + 3.0 * np.std(gray_f)),
    )
    residual_threshold = max(12.0, float(np.percentile(residual, 99.5)))
    return (gray_f >= bright_threshold) & (residual >= residual_threshold)


def compute_high_freq_energy(gray_image: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_image, cv2.CV_32F)
    return float(np.var(lap))


def compute_edge_density(gray_image: np.ndarray) -> float:
    edges = cv2.Canny(gray_image, 80, 160)
    return float(np.mean(edges > 0))


def compute_chroma_mottle_std(cr: np.ndarray, cb: np.ndarray) -> float:
    cr_low = cv2.GaussianBlur(cr, (0, 0), sigmaX=3.0)
    cb_low = cv2.GaussianBlur(cb, (0, 0), sigmaX=3.0)
    cr_residual = cr - cr_low
    cb_residual = cb - cb_low
    return float(np.sqrt(np.mean(cr_residual**2 + cb_residual**2)))


def corner_stack(array: np.ndarray) -> np.ndarray:
    h, w = array.shape[:2]
    ch = max(1, h // 5)
    cw = max(1, w // 5)
    corners = [
        array[:ch, :cw],
        array[:ch, w - cw :],
        array[h - ch :, :cw],
        array[h - ch :, w - cw :],
    ]
    return np.concatenate([corner.reshape(-1, *corner.shape[2:]) for corner in corners])


def center_crop(array: np.ndarray) -> np.ndarray:
    h, w = array.shape[:2]
    y0 = h // 4
    y1 = max(y0 + 1, (3 * h) // 4)
    x0 = w // 4
    x1 = max(x0 + 1, (3 * w) // 4)
    return array[y0:y1, x0:x1]


def compute_flat_metrics(rgb_f: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float, float]:
    sigma = max(3.0, min(y.shape[:2]) / 24.0)
    low_freq_y = cv2.GaussianBlur(y.astype(np.float32), (0, 0), sigmaX=sigma)
    shading_range_norm = safe_divide(
        float(np.percentile(low_freq_y, 95) - np.percentile(low_freq_y, 5)),
        float(np.mean(low_freq_y)),
    )

    center_y = float(np.mean(center_crop(y)))
    corner_y = float(np.mean(corner_stack(y)))
    center_corner_y_ratio = safe_divide(center_y, corner_y)

    center_rgb = center_crop(rgb_f)
    center_r = float(np.mean(center_rgb[:, :, 0]))
    center_g = float(np.mean(center_rgb[:, :, 1]))
    center_b = float(np.mean(center_rgb[:, :, 2]))
    center_g_over_r = safe_divide(center_g, center_r)
    center_g_over_b = safe_divide(center_g, center_b)

    return (
        shading_range_norm,
        center_y,
        corner_y,
        center_corner_y_ratio,
        center_g_over_r,
        center_g_over_b,
    )


def analyze_image(image_path: Path, metadata: ImageMetadata) -> ImageAnalysisResult:
    rgb = read_rgb(image_path)
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

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))
    mean_y = float(np.mean(y))
    green_excess = float(mean_g - ((mean_r + mean_b) / 2.0))

    dark_mask = y < 40.0
    dark_region_ratio = float(np.mean(dark_mask))
    dark_noise_std = float(np.std(y[dark_mask])) if np.any(dark_mask) else 0.0

    hot_mask = compute_hot_pixel_mask_from_gray(gray)
    hot_pixel_count = int(np.sum(hot_mask))
    hot_pixel_ratio = float(hot_pixel_count / (h * w))

    (
        flat_shading_range_norm,
        flat_center_y,
        flat_corner_y,
        flat_center_corner_y_ratio,
        flat_center_g_over_r,
        flat_center_g_over_b,
    ) = compute_flat_metrics(rgb_f, y)

    return ImageAnalysisResult(
        file_path=metadata.file_path,
        relative_path=metadata.relative_path,
        time_label=metadata.time_label,
        weather_label=metadata.weather_label,
        scene_label=metadata.scene_label,
        scene_kind=metadata.scene_kind,
        zoom_label=metadata.zoom_label,
        frame_index=metadata.frame_index,
        frame_number=metadata.frame_number,
        burst_id=metadata.burst_id,
        width=w,
        height=h,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=float(np.std(r)),
        std_g=float(np.std(g)),
        std_b=float(np.std(b)),
        mean_y=mean_y,
        std_y=float(np.std(y)),
        mean_cr=float(np.mean(cr)),
        mean_cb=float(np.mean(cb)),
        std_cr=float(np.std(cr)),
        std_cb=float(np.std(cb)),
        tint_rg=float(mean_r - mean_g),
        tint_bg=float(mean_b - mean_g),
        g_over_r=safe_divide(mean_g, mean_r),
        g_over_b=safe_divide(mean_g, mean_b),
        green_excess=green_excess,
        green_excess_norm=safe_divide(green_excess, mean_y),
        chroma_mottle_std=compute_chroma_mottle_std(cr, cb),
        edge_density=compute_edge_density(gray),
        high_freq_energy=compute_high_freq_energy(gray),
        dark_region_ratio=dark_region_ratio,
        dark_noise_std=dark_noise_std,
        hot_pixel_ratio=hot_pixel_ratio,
        hot_pixel_count=hot_pixel_count,
        y_clip_low_ratio=float(np.mean(y <= 3.0)),
        y_clip_high_ratio=float(np.mean(y >= 252.0)),
        rgb_clip_low_ratio=float(np.mean(np.any(rgb_f <= 3.0, axis=2))),
        rgb_clip_high_ratio=float(np.mean(np.any(rgb_f >= 252.0, axis=2))),
        r_clip_high_ratio=float(np.mean(r >= 252.0)),
        g_clip_high_ratio=float(np.mean(g >= 252.0)),
        b_clip_high_ratio=float(np.mean(b >= 252.0)),
        flat_shading_range_norm=flat_shading_range_norm,
        flat_center_y=flat_center_y,
        flat_corner_y=flat_corner_y,
        flat_center_corner_y_ratio=flat_center_corner_y_ratio,
        flat_center_g_over_r=flat_center_g_over_r,
        flat_center_g_over_b=flat_center_g_over_b,
    )


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return slug.strip("_")[:180] or "unknown"


def save_heatmap(array: np.ndarray, path: Path, title: str, cmap: str = "magma") -> str:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        return ""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(np.nan_to_num(arr), cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def load_gray_stack(paths: list[Path]) -> tuple[np.ndarray, list[Path]]:
    stack: list[np.ndarray] = []
    used_paths: list[Path] = []
    base_shape: Optional[tuple[int, int]] = None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if base_shape is None:
            base_shape = gray.shape
        if gray.shape != base_shape:
            LOGGER.warning("Skipping shape-mismatched burst frame: %s", path)
            continue
        stack.append(gray.astype(np.float32))
        used_paths.append(path)
    if not stack:
        return np.empty((0, 0, 0), dtype=np.float32), []
    return np.stack(stack, axis=0), used_paths


def load_rgb_stack(paths: list[Path]) -> np.ndarray:
    stack: list[np.ndarray] = []
    base_shape: Optional[tuple[int, int, int]] = None
    for path in paths:
        try:
            rgb = read_rgb(path).astype(np.float32)
        except ValueError:
            continue
        if base_shape is None:
            base_shape = rgb.shape
        if rgb.shape != base_shape:
            continue
        stack.append(rgb)
    if not stack:
        return np.empty((0, 0, 0, 0), dtype=np.float32)
    return np.stack(stack, axis=0)


def should_save_maps(scene_kind: str, args: argparse.Namespace) -> bool:
    if args.no_maps:
        return False
    if scene_kind in {"calibration_flat", "calibration_dark"}:
        return True
    return bool(args.save_outdoor_maps)


def analyze_burst(
    burst_id: str,
    burst_paths: list[Path],
    burst_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> BurstAnalysisResult:
    burst_df = burst_df.sort_values(["frame_number", "relative_path"])
    first = burst_df.iloc[0]
    stack, used_paths = load_gray_stack(burst_paths)

    if stack.size == 0:
        h = int(first["height"])
        w = int(first["width"])
        temporal_std_map = np.zeros((h, w), dtype=np.float32)
    else:
        temporal_std_map = np.std(stack, axis=0) if stack.shape[0] > 1 else np.zeros_like(stack[0])
        h, w = temporal_std_map.shape[:2]

    hot_persistence = np.zeros((h, w), dtype=np.float32)
    if used_paths:
        masks = []
        for frame in stack:
            masks.append(compute_hot_pixel_mask_from_gray(frame.astype(np.uint8)))
        if masks:
            hot_persistence = np.mean(np.stack(masks, axis=0), axis=0).astype(np.float32)

    pixel_count = max(1, h * w)
    hot_count_25 = int(np.sum(hot_persistence >= 0.25))
    hot_count_50 = int(np.sum(hot_persistence >= 0.50))
    hot_count_75 = int(np.sum(hot_persistence >= 0.75))

    mean_y_values = burst_df["mean_y"].to_numpy(dtype=np.float32)
    green_excess_values = burst_df["green_excess"].to_numpy(dtype=np.float32)
    g_over_r_values = burst_df["g_over_r"].to_numpy(dtype=np.float32)
    g_over_b_values = burst_df["g_over_b"].to_numpy(dtype=np.float32)
    edge_values = burst_df["edge_density"].to_numpy(dtype=np.float32)
    high_freq_values = burst_df["high_freq_energy"].to_numpy(dtype=np.float32)

    mean_y_range = float(np.max(mean_y_values) - np.min(mean_y_values))
    green_excess_range = float(np.max(green_excess_values) - np.min(green_excess_values))
    high_freq_range = float(np.max(high_freq_values) - np.min(high_freq_values))
    high_freq_ref = float(np.median(high_freq_values)) + EPS
    high_freq_rel_range = float(high_freq_range / high_freq_ref)

    unstable_luma = bool(mean_y_range > args.unstable_y_range)
    unstable_color = bool(green_excess_range > args.unstable_color_range)
    unstable_detail = bool(high_freq_rel_range > args.unstable_detail_rel_range)
    unstable_auto = bool(unstable_luma or unstable_color or unstable_detail)

    map_temporal_std = ""
    map_hot_persistence = ""
    map_hot_fixed_mask = ""
    map_flat_shading = ""
    map_flat_green_excess = ""

    scene_kind = str(first["scene_kind"])
    if should_save_maps(scene_kind, args):
        map_dir = output_dir / "maps"
        slug = slugify(burst_id)
        map_temporal_std = save_heatmap(
            temporal_std_map,
            map_dir / f"{slug}_temporal_std.png",
            f"{burst_id} temporal std",
            cmap="magma",
        )
        map_hot_persistence = save_heatmap(
            hot_persistence,
            map_dir / f"{slug}_hot_persistence.png",
            f"{burst_id} hot-pixel persistence",
            cmap="inferno",
        )
        if scene_kind == "calibration_flat":
            rgb_stack = load_rgb_stack(used_paths)
            if rgb_stack.size > 0:
                mean_rgb = np.mean(rgb_stack, axis=0)
                ycrcb = cv2.cvtColor(np.clip(mean_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
                mean_y = ycrcb[:, :, 0].astype(np.float32)
                sigma = max(3.0, min(mean_y.shape[:2]) / 24.0)
                flat_shading = cv2.GaussianBlur(mean_y, (0, 0), sigmaX=sigma)
                green_excess = mean_rgb[:, :, 1] - ((mean_rgb[:, :, 0] + mean_rgb[:, :, 2]) / 2.0)
                map_flat_shading = save_heatmap(
                    flat_shading,
                    map_dir / f"{slug}_flat_shading.png",
                    f"{burst_id} flat shading map",
                    cmap="viridis",
                )
                map_flat_green_excess = save_heatmap(
                    green_excess,
                    map_dir / f"{slug}_flat_green_excess.png",
                    f"{burst_id} green excess map",
                    cmap="coolwarm",
                )

    return BurstAnalysisResult(
        burst_id=burst_id,
        time_label=str(first["time_label"]),
        weather_label=str(first["weather_label"]),
        scene_label=str(first["scene_label"]),
        scene_kind=scene_kind,
        zoom_label=str(first["zoom_label"]),
        frame_count=int(len(burst_df)),
        width=w,
        height=h,
        burst_temporal_noise_mean=float(np.mean(temporal_std_map)),
        burst_temporal_noise_median=float(np.median(temporal_std_map)),
        burst_temporal_noise_p90=float(np.percentile(temporal_std_map, 90)),
        burst_temporal_noise_max=float(np.max(temporal_std_map)),
        burst_mean_y_std=float(np.std(mean_y_values)),
        burst_mean_y_range=mean_y_range,
        burst_mean_y_drift=float(mean_y_values[-1] - mean_y_values[0]) if len(mean_y_values) else 0.0,
        burst_g_over_r_range=float(np.max(g_over_r_values) - np.min(g_over_r_values)),
        burst_g_over_b_range=float(np.max(g_over_b_values) - np.min(g_over_b_values)),
        burst_green_excess_range=green_excess_range,
        burst_edge_density_range=float(np.max(edge_values) - np.min(edge_values)),
        burst_high_freq_range=high_freq_range,
        burst_high_freq_rel_range=high_freq_rel_range,
        unstable_luma=unstable_luma,
        unstable_color=unstable_color,
        unstable_detail=unstable_detail,
        unstable_auto=unstable_auto,
        hot_persistence_count_25=hot_count_25,
        hot_persistence_count_50=hot_count_50,
        hot_persistence_count_75=hot_count_75,
        hot_persistence_ratio_25=float(hot_count_25 / pixel_count),
        hot_persistence_ratio_50=float(hot_count_50 / pixel_count),
        hot_persistence_ratio_75=float(hot_count_75 / pixel_count),
        map_temporal_std=map_temporal_std,
        map_hot_persistence=map_hot_persistence,
        map_hot_fixed_mask=map_hot_fixed_mask,
        map_flat_shading=map_flat_shading,
        map_flat_green_excess=map_flat_green_excess,
    )


def build_dataframe(results: Iterable[ImageAnalysisResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(result) for result in results])


def summarize_group(group: pd.DataFrame, metric_cols: list[str]) -> dict[str, object]:
    record: dict[str, object] = {
        "image_count": int(len(group)),
        "burst_count": int(group["burst_id"].nunique()),
    }
    for metric in metric_cols:
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if values.empty:
            continue
        record[f"{metric}_mean"] = float(values.mean())
        record[f"{metric}_median"] = float(values.median())
        record[f"{metric}_p10"] = float(values.quantile(0.10))
        record[f"{metric}_p90"] = float(values.quantile(0.90))
        record[f"{metric}_std"] = float(values.std(ddof=0))
    return record


def save_group_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    group_cols = ["time_label", "weather_label", "scene_label", "scene_kind", "zoom_label"]
    metric_cols = [col for col in SUMMARY_METRICS if col in df.columns]
    records: list[dict[str, object]] = []

    for keys, group in df.groupby(group_cols, dropna=False):
        record = dict(zip(group_cols, keys))
        record.update(summarize_group(group, metric_cols))
        records.append(record)

    summary = pd.DataFrame(records).sort_values(group_cols)
    summary_path = output_dir / "group_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved grouped summary to %s", summary_path)
    return summary


def save_scene_contrast_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = [
        "mean_y",
        "std_y",
        "g_over_r",
        "g_over_b",
        "green_excess",
        "chroma_mottle_std",
        "edge_density",
        "high_freq_energy",
        "hot_pixel_ratio",
        "dark_noise_std",
    ]
    metrics = [metric for metric in metrics if metric in df.columns]
    if not metrics:
        return pd.DataFrame()

    flat_df = df[df["scene_kind"] == "calibration_flat"]
    dark_df = df[df["scene_kind"] == "calibration_dark"]
    comparison_rows: list[dict[str, object]] = []

    for keys, group in df.groupby(["scene_label", "scene_kind", "zoom_label"], dropna=False):
        scene_label, scene_kind, zoom_label = keys
        if scene_kind in {"calibration_flat", "calibration_dark"}:
            continue
        record: dict[str, object] = {
            "scene_label": scene_label,
            "scene_kind": scene_kind,
            "zoom_label": zoom_label,
            "image_count": int(len(group)),
        }
        flat_ref = flat_df[flat_df["zoom_label"] == zoom_label]
        dark_ref = dark_df[dark_df["zoom_label"] == zoom_label]

        for metric in metrics:
            group_value = float(group[metric].median())
            record[f"{metric}_median"] = group_value
            if not flat_ref.empty and metric not in {"hot_pixel_ratio", "dark_noise_std"}:
                record[f"{metric}_delta_vs_flat"] = group_value - float(flat_ref[metric].median())
            if not dark_ref.empty and metric in {"hot_pixel_ratio", "dark_noise_std"}:
                record[f"{metric}_delta_vs_dark"] = group_value - float(dark_ref[metric].median())
        comparison_rows.append(record)

    contrast = pd.DataFrame(comparison_rows)
    contrast_path = output_dir / "scene_contrast_summary.csv"
    contrast.to_csv(contrast_path, index=False)
    LOGGER.info("Saved scene contrast summary to %s", contrast_path)
    return contrast


def build_hot_persistence_map(paths: list[Path]) -> np.ndarray:
    """Build an in-memory persistence map for overlap statistics only."""
    stack, _ = load_gray_stack(paths)
    if stack.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    masks = [
        compute_hot_pixel_mask_from_gray(frame.astype(np.uint8))
        for frame in stack
    ]
    if not masks:
        return np.empty((0, 0), dtype=np.float32)
    return np.mean(np.stack(masks, axis=0), axis=0).astype(np.float32)


def save_hot_pixel_overlap_summary(
    per_image_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Summarize hot-pixel candidate overlap without saving sensor-specific masks.

    A per-device hot-pixel mask is not emitted because the deployment target has
    multiple sensors and the denoise model should remain sensor-general. The CSV
    keeps only counts/ratios that are useful for choosing augmentation ranges.
    """
    if per_image_df.empty:
        return pd.DataFrame()

    persistence_by_burst: dict[str, np.ndarray] = {}
    meta_by_burst: dict[str, dict[str, object]] = {}
    for burst_id, group in per_image_df.groupby("burst_id", dropna=False):
        burst_key = str(burst_id)
        paths = [Path(path) for path in group["file_path"].tolist()]
        persistence = build_hot_persistence_map(paths)
        if persistence.size == 0:
            continue
        persistence_by_burst[burst_key] = persistence
        first = group.iloc[0]
        meta_by_burst[burst_key] = {
            "time_label": first["time_label"],
            "weather_label": first["weather_label"],
            "scene_label": first["scene_label"],
            "scene_kind": first["scene_kind"],
            "zoom_label": first["zoom_label"],
            "pixel_count": int(persistence.size),
        }

    dark_keys = [
        key
        for key, meta in meta_by_burst.items()
        if meta["scene_kind"] == "calibration_dark"
    ]
    flat_keys = [
        key
        for key, meta in meta_by_burst.items()
        if meta["scene_kind"] == "calibration_flat"
    ]
    thresholds = [0.25, 0.50, 0.75, 0.90]
    records: list[dict[str, object]] = []

    for idx, key_a in enumerate(dark_keys):
        for key_b in dark_keys[idx + 1:]:
            map_a = persistence_by_burst[key_a]
            map_b = persistence_by_burst[key_b]
            if map_a.shape != map_b.shape:
                continue
            meta_a = meta_by_burst[key_a]
            meta_b = meta_by_burst[key_b]
            for threshold in thresholds:
                mask_a = map_a >= threshold
                mask_b = map_b >= threshold
                count_a = int(mask_a.sum())
                count_b = int(mask_b.sum())
                intersection = int(np.logical_and(mask_a, mask_b).sum())
                union = int(np.logical_or(mask_a, mask_b).sum())
                records.append({
                    "comparison_type": "dark_vs_dark",
                    "threshold": threshold,
                    "burst_a": key_a,
                    "burst_b": key_b,
                    "scene_a": meta_a["scene_label"],
                    "scene_b": meta_b["scene_label"],
                    "zoom_a": meta_a["zoom_label"],
                    "zoom_b": meta_b["zoom_label"],
                    "count_a": count_a,
                    "count_b": count_b,
                    "intersection_count": intersection,
                    "union_count": union,
                    "jaccard": safe_divide(intersection, union),
                    "intersection_ratio_a": safe_divide(intersection, count_a),
                    "intersection_ratio_b": safe_divide(intersection, count_b),
                })

    for threshold in thresholds:
        dark_masks = []
        for dark_key in dark_keys:
            dark_masks.append(persistence_by_burst[dark_key] >= threshold)
        if not dark_masks:
            continue
        dark_union = np.logical_or.reduce(dark_masks)
        dark_count = int(dark_union.sum())
        for flat_key in flat_keys:
            flat_map = persistence_by_burst[flat_key]
            if flat_map.shape != dark_union.shape:
                continue
            flat_mask = flat_map >= threshold
            flat_count = int(flat_mask.sum())
            intersection = int(np.logical_and(flat_mask, dark_union).sum())
            union = int(np.logical_or(flat_mask, dark_union).sum())
            meta = meta_by_burst[flat_key]
            records.append({
                "comparison_type": "flat_vs_dark_union",
                "threshold": threshold,
                "burst_a": flat_key,
                "burst_b": "dark_union",
                "scene_a": meta["scene_label"],
                "scene_b": "dark",
                "zoom_a": meta["zoom_label"],
                "zoom_b": "all",
                "count_a": flat_count,
                "count_b": dark_count,
                "intersection_count": intersection,
                "union_count": union,
                "jaccard": safe_divide(intersection, union),
                "intersection_ratio_a": safe_divide(intersection, flat_count),
                "intersection_ratio_b": safe_divide(intersection, dark_count),
            })

    overlap = pd.DataFrame(records)
    overlap_path = output_dir / "hot_pixel_overlap_summary.csv"
    overlap.to_csv(overlap_path, index=False)
    LOGGER.info("Saved hot-pixel overlap summary to %s", overlap_path)
    return overlap


def save_previous_delta(
    current_summary: pd.DataFrame,
    previous_group_summary: str,
    output_dir: Path,
) -> pd.DataFrame:
    previous_path = Path(previous_group_summary).expanduser().resolve()
    if not previous_path.exists():
        LOGGER.warning("Previous group summary does not exist: %s", previous_path)
        return pd.DataFrame()

    previous = pd.read_csv(previous_path)
    key_cols = ["time_label", "weather_label", "scene_label", "scene_kind", "zoom_label"]
    if any(col not in previous.columns for col in key_cols):
        LOGGER.warning("Previous summary is missing expected key columns: %s", previous_path)
        return pd.DataFrame()

    merged = current_summary.merge(previous, on=key_cols, how="inner", suffixes=("_current", "_previous"))
    candidate_metrics = [
        "mean_y_mean",
        "std_y_mean",
        "g_over_r_mean",
        "g_over_b_mean",
        "green_excess_mean",
        "chroma_mottle_std_mean",
        "hot_pixel_ratio_mean",
        "burst_temporal_noise_mean_mean",
        "hot_persistence_ratio_50_mean",
    ]

    delta_records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        record = {col: row[col] for col in key_cols}
        for metric in candidate_metrics:
            current_col = f"{metric}_current"
            previous_col = f"{metric}_previous"
            if current_col in merged.columns and previous_col in merged.columns:
                current_value = float(row[current_col])
                previous_value = float(row[previous_col])
                record[f"{metric}_current"] = current_value
                record[f"{metric}_previous"] = previous_value
                record[f"{metric}_delta"] = current_value - previous_value
        delta_records.append(record)

    delta = pd.DataFrame(delta_records)
    delta_path = output_dir / "previous_run_delta.csv"
    delta.to_csv(delta_path, index=False)
    LOGGER.info("Saved previous-run delta to %s", delta_path)
    return delta


def save_metric_barplot(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    if metric not in df.columns or df.empty:
        return
    pivot = (
        df.groupby(["scene_label", "zoom_label"], dropna=False)[metric]
        .median()
        .reset_index()
        .pivot(index="scene_label", columns="zoom_label", values=metric)
    )
    if pivot.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax = pivot.plot(kind="bar", figsize=(11, 6), rot=30)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scene")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_plots(df: pd.DataFrame, burst_df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, group in df.groupby("scene_label"):
        plt.scatter(group["mean_y"], group["std_y"], label=label, alpha=0.65, s=18)
    plt.xlabel("Mean Y")
    plt.ylabel("Std Y")
    plt.title("Brightness vs Luma Variation")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "brightness_vs_luma_std.png", dpi=150)
    plt.close()

    save_metric_barplot(
        df,
        "green_excess",
        plot_dir / "green_excess_by_scene_zoom.png",
        "Green Excess by Scene / Zoom",
        "G - (R+B)/2",
    )
    save_metric_barplot(
        df,
        "chroma_mottle_std",
        plot_dir / "chroma_mottle_by_scene_zoom.png",
        "Chroma Mottle Proxy by Scene / Zoom",
        "Chroma residual RMS",
    )
    save_metric_barplot(
        df,
        "high_freq_energy",
        plot_dir / "high_freq_by_scene_zoom.png",
        "High-Frequency Energy by Scene / Zoom",
        "Laplacian variance",
    )
    save_metric_barplot(
        df,
        "hot_pixel_ratio",
        plot_dir / "single_frame_hot_ratio_by_scene_zoom.png",
        "Single-Frame Hot-Pixel Candidate Ratio",
        "Ratio",
    )

    if not burst_df.empty:
        save_metric_barplot(
            burst_df,
            "burst_temporal_noise_mean",
            plot_dir / "temporal_noise_by_scene_zoom.png",
            "Burst Temporal Noise by Scene / Zoom",
            "Mean temporal std",
        )
        save_metric_barplot(
            burst_df,
            "hot_persistence_ratio_50",
            plot_dir / "hot_persistence_50_by_scene_zoom.png",
            "Persistent Hot-Pixel Candidate Ratio",
            "Persistence >= 50%",
        )

    LOGGER.info("Saved plots to %s", plot_dir)


def format_value(value: object) -> str:
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.6f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    available = [col for col in columns if col in df.columns]
    if not available or df.empty:
        return "_No data._"

    view = df[available].head(max_rows)
    header = "| " + " | ".join(available) + " |"
    sep = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(format_value(row[col]) for col in available) + " |")
    return "\n".join([header, sep, *rows])


def render_summary_markdown(
    df: pd.DataFrame,
    burst_df: pd.DataFrame,
    group_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    hot_overlap_summary: pd.DataFrame,
    previous_delta: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
) -> str:
    scene_counts = (
        df.groupby(["scene_label", "zoom_label"], dropna=False)
        .size()
        .reset_index(name="image_count")
        .sort_values(["scene_label", "zoom_label"])
    )
    top_hot = df.sort_values("hot_pixel_count", ascending=False).head(8)
    flat_rows = group_summary[group_summary["scene_kind"] == "calibration_flat"]
    dark_bursts = burst_df[burst_df["scene_kind"] == "calibration_dark"]
    unstable_bursts = burst_df[burst_df["unstable_auto"].astype(bool)]

    lines = [
        "# MC-G105 Capture Analysis Summary",
        "",
        f"- Input: `{input_dir}`",
        f"- Output: `{output_dir}`",
        f"- Total images: {len(df)}",
        f"- Total bursts: {df['burst_id'].nunique()}",
        f"- Scenes: {', '.join(sorted(df['scene_label'].unique().tolist()))}",
        "",
        "## Scene / Zoom Counts",
        "",
        dataframe_to_markdown(scene_counts, ["scene_label", "zoom_label", "image_count"], max_rows=30),
        "",
        "## Flat Baseline",
        "",
        dataframe_to_markdown(
            flat_rows,
            [
                "time_label",
                "weather_label",
                "scene_label",
                "zoom_label",
                "image_count",
                "g_over_r_mean",
                "g_over_b_mean",
                "green_excess_mean",
                "flat_shading_range_norm_mean",
                "chroma_mottle_std_mean",
            ],
            max_rows=20,
        ),
        "",
        "## Dark / Hot-Pixel Persistence",
        "",
        dataframe_to_markdown(
            dark_bursts,
            [
                "burst_id",
                "zoom_label",
                "frame_count",
                "hot_persistence_count_25",
                "hot_persistence_count_50",
                "hot_persistence_count_75",
                "hot_persistence_ratio_50",
                "burst_temporal_noise_mean",
            ],
            max_rows=20,
        ),
        "",
        "## Group Summary Preview",
        "",
        dataframe_to_markdown(
            group_summary,
            [
                "time_label",
                "weather_label",
                "scene_label",
                "scene_kind",
                "zoom_label",
                "image_count",
                "mean_y_mean",
                "std_y_mean",
                "green_excess_mean",
                "hot_pixel_ratio_mean",
                "burst_temporal_noise_mean_mean",
            ],
            max_rows=20,
        ),
        "",
        "## Top Single-Frame Hot-Pixel Candidates",
        "",
        dataframe_to_markdown(
            top_hot,
            [
                "relative_path",
                "scene_label",
                "zoom_label",
                "hot_pixel_count",
                "hot_pixel_ratio",
                "dark_noise_std",
                "green_excess",
            ],
            max_rows=8,
        ),
        "",
        "## Auto-Stability Flags",
        "",
        dataframe_to_markdown(
            unstable_bursts,
            [
                "burst_id",
                "scene_label",
                "zoom_label",
                "unstable_luma",
                "unstable_color",
                "unstable_detail",
                "burst_mean_y_range",
                "burst_green_excess_range",
                "burst_high_freq_rel_range",
            ],
            max_rows=20,
        ),
        "",
        "## Hot-Pixel Candidate Overlap",
        "",
        dataframe_to_markdown(
            hot_overlap_summary,
            [
                "comparison_type",
                "threshold",
                "burst_a",
                "burst_b",
                "zoom_a",
                "zoom_b",
                "count_a",
                "count_b",
                "intersection_count",
                "jaccard",
                "intersection_ratio_a",
            ],
            max_rows=24,
        ),
        "",
        "## Scene Contrast Preview",
        "",
        dataframe_to_markdown(
            contrast_summary,
            [
                "scene_label",
                "zoom_label",
                "image_count",
                "green_excess_delta_vs_flat",
                "chroma_mottle_std_delta_vs_flat",
                "high_freq_energy_delta_vs_flat",
                "hot_pixel_ratio_delta_vs_dark",
            ],
            max_rows=20,
        ),
        "",
        "## Previous Run Delta",
        "",
        dataframe_to_markdown(
            previous_delta,
            [
                "time_label",
                "weather_label",
                "scene_label",
                "zoom_label",
                "green_excess_mean_delta",
                "hot_pixel_ratio_mean_delta",
                "burst_temporal_noise_mean_mean_delta",
            ],
            max_rows=20,
        ),
        "",
        "## Interpretation Notes",
        "",
        "- `flat`은 실내 흰 벽 baseline이므로 야외 색 편향 전체를 대표하지 않는다.",
        "- `dark`는 렌즈 가림 baseline이며 Auto exposure 때문에 물리 dark current로 해석하지 않는다.",
        "- `hot_persistence_ratio_50`은 같은 위치가 burst의 50% 이상에서 hot-pixel 후보로 반복된 비율이다.",
        "- `unstable_luma`, `unstable_color`, `unstable_detail`은 밝기/색/디테일 변화를 분리해서 표시한다.",
        "- hot-pixel overlap summary는 특정 센서 mask를 저장하지 않고 count/ratio만 저장한다.",
        "- NV12 원본 plane이 아니라 저장된 이미지 분석이면 Y/Cr/Cb 지표는 RGB 변환 이후 proxy다.",
    ]
    return "\n".join(lines)


def main() -> int:
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

    LOGGER.info("Found %d images under %s", len(image_paths), input_dir)

    results: list[ImageAnalysisResult] = []
    for image_path in image_paths:
        metadata = parse_metadata(image_path, input_dir)
        try:
            results.append(analyze_image(image_path, metadata))
        except Exception as exc:
            LOGGER.warning("Skipping %s due to error: %s", image_path, exc)

    if not results:
        LOGGER.error("No images were successfully analyzed.")
        return 1

    per_image_df = build_dataframe(results)
    burst_results: list[BurstAnalysisResult] = []
    for burst_id, burst_frame_df in per_image_df.groupby("burst_id", dropna=False):
        burst_paths = [Path(path) for path in burst_frame_df["file_path"].tolist()]
        burst_results.append(analyze_burst(str(burst_id), burst_paths, burst_frame_df, output_dir, args))

    burst_df = pd.DataFrame([asdict(result) for result in burst_results])
    burst_metric_cols = [
        col
        for col in burst_df.columns
        if col
        not in {
            "time_label",
            "weather_label",
            "scene_label",
            "scene_kind",
            "zoom_label",
            "width",
            "height",
        }
    ]
    per_image_df = per_image_df.merge(burst_df[burst_metric_cols], on="burst_id", how="left")

    per_image_path = output_dir / "per_image_metrics.csv"
    burst_path = output_dir / "burst_metrics.csv"
    per_image_df.to_csv(per_image_path, index=False)
    burst_df.to_csv(burst_path, index=False)
    LOGGER.info("Saved per-image metrics to %s", per_image_path)
    LOGGER.info("Saved burst metrics to %s", burst_path)

    group_summary = save_group_summary(per_image_df, output_dir)
    contrast_summary = save_scene_contrast_summary(per_image_df, output_dir)
    hot_overlap_summary = save_hot_pixel_overlap_summary(per_image_df, output_dir)
    previous_delta = (
        save_previous_delta(group_summary, args.previous_group_summary, output_dir)
        if args.previous_group_summary
        else pd.DataFrame()
    )
    save_plots(per_image_df, burst_df, output_dir)

    summary_markdown = render_summary_markdown(
        per_image_df,
        burst_df,
        group_summary,
        contrast_summary,
        hot_overlap_summary,
        previous_delta,
        input_dir,
        output_dir,
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_markdown, encoding="utf-8")
    LOGGER.info("Saved markdown summary to %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
