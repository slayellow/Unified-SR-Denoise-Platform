     1|import argparse
     2|import logging
     3|import re
     4|import sys
     5|from dataclasses import asdict, dataclass
     6|from pathlib import Path
     7|from typing import Iterable
     8|
     9|import cv2
    10|import matplotlib.pyplot as plt
    11|import numpy as np
    12|import pandas as pd
    13|
    14|
    15|LOGGER = logging.getLogger(__name__)
    16|IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    17|
    18|
    19|@dataclass(slots=True)
    20|class ImageAnalysisResult:
    21|    """Container for per-image sensor statistics."""
    22|
    23|    file_path: str
    24|    scene_label: str
    25|    time_label: str
    26|    zoom_label: str
    27|    width: int
    28|    height: int
    29|    mean_r: float
    30|    mean_g: float
    31|    mean_b: float
    32|    std_r: float
    33|    std_g: float
    34|    std_b: float
    35|    mean_y: float
    36|    std_y: float
    37|    mean_cr: float
    38|    mean_cb: float
    39|    tint_rg: float
    40|    tint_bg: float
    41|    edge_density: float
    42|    high_freq_energy: float
    43|    dark_region_ratio: float
    44|    dark_noise_std: float
    45|    hot_pixel_ratio: float
    46|    hot_pixel_count: int
    47|
    48|
    49|def setup_logging(verbose: bool) -> None:
    50|    """Configure logging for console output."""
    51|    level = logging.DEBUG if verbose else logging.INFO
    52|    logging.basicConfig(
    53|        level=level,
    54|        format="[%(levelname)s] %(message)s",
    55|    )
    56|
    57|
    58|def parse_args() -> argparse.Namespace:
    59|    """Parse command-line arguments."""
    60|    parser = argparse.ArgumentParser(
    61|        description="Analyze MC-G105 / VISCA output frame captures and summarize sensor-like artifacts."
    62|    )
    63|    parser.add_argument(
    64|        "--input_dir",
    65|        type=str,
    66|        required=True,
    67|        help="Directory containing extracted frame images. Nested folders are supported.",
    68|    )
    69|    parser.add_argument(
    70|        "--output_dir",
    71|        type=str,
    72|        default="results/mc_g105_analysis",
    73|        help="Directory to save CSV, plots, and markdown summary.",
    74|    )
    75|    parser.add_argument(
    76|        "--max_images",
    77|        type=int,
    78|        default=0,
    79|        help="Optional cap on the number of images to analyze. 0 means all images.",
    80|    )
    81|    parser.add_argument(
    82|        "--verbose",
    83|        action="store_true",
    84|        help="Enable verbose logging.",
    85|    )
    86|    return parser.parse_args()
    87|
    88|
    89|def collect_image_paths(input_dir: Path, max_images: int) -> list[Path]:
    90|    """Collect image files recursively from the input directory."""
    91|    image_paths = sorted(
    92|        path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    93|    )
    94|    if max_images > 0:
    95|        image_paths = image_paths[:max_images]
    96|    return image_paths
    97|
    98|
    99|def infer_time_label(path: Path) -> str:
   100|    """Infer a coarse day/night label from path parts."""
   101|    lowered_parts = [part.lower() for part in path.parts]
   102|    if any("night" in part for part in lowered_parts):
   103|        return "night"
   104|    if any("day" in part for part in lowered_parts):
   105|        return "day"
   106|    return "unknown"
   107|
   108|
   109|def infer_zoom_label(path: Path) -> str:
   110|    """Infer zoom label like 1x/7x from file or directory names."""
   111|    joined = " / ".join(path.parts).lower()
   112|    match = re.search(r"(?<!\d)(\d{1,2})x(?!\d)", joined)
   113|    if match:
   114|        return f"{match.group(1)}x"
   115|    return "unknown"
   116|
   117|
   118|def infer_scene_label(path: Path) -> str:
   119|    """Infer a scene label from the parent directory structure."""
   120|    if len(path.parts) >= 2:
   121|        return path.parent.name
   122|    return "default"
   123|
   124|
   125|def compute_hot_pixel_mask(rgb_image: np.ndarray) -> np.ndarray:
   126|    """Estimate hot-pixel-like outliers from intensity and local neighborhood mismatch."""
   127|    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
   128|    gray_f = gray.astype(np.float32)
   129|    local_median = cv2.medianBlur(gray, 3).astype(np.float32)
   130|    residual = gray_f - local_median
   131|    bright_threshold = np.percentile(gray_f, 99.8)
   132|    residual_threshold = max(18.0, np.percentile(residual, 99.5))
   133|    hot_mask = (gray_f >= bright_threshold) & (residual >= residual_threshold)
   134|    return hot_mask
   135|
   136|
   137|def compute_high_freq_energy(gray_image: np.ndarray) -> float:
   138|    """Compute a simple normalized high-frequency energy score using Laplacian variance."""
   139|    lap = cv2.Laplacian(gray_image, cv2.CV_32F)
   140|    return float(np.var(lap))
   141|
   142|
   143|def analyze_image(image_path: Path) -> ImageAnalysisResult:
   144|    """Analyze one image and return sensor-oriented summary statistics."""
   145|    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
   146|    if bgr is None:
   147|        raise ValueError(f"Failed to read image: {image_path}")
   148|
   149|    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
   150|    rgb_f = rgb.astype(np.float32)
   151|    h, w = rgb.shape[:2]
   152|
   153|    r = rgb_f[:, :, 0]
   154|    g = rgb_f[:, :, 1]
   155|    b = rgb_f[:, :, 2]
   156|
   157|    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
   158|    y = ycrcb[:, :, 0]
   159|    cr = ycrcb[:, :, 1]
   160|    cb = ycrcb[:, :, 2]
   161|
   162|    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
   163|    edges = cv2.Canny(gray, 80, 160)
   164|    edge_density = float(np.mean(edges > 0))
   165|
   166|    dark_mask = y < 40.0
   167|    dark_region_ratio = float(np.mean(dark_mask))
   168|    dark_noise_std = float(np.std(y[dark_mask])) if np.any(dark_mask) else 0.0
   169|
   170|    hot_mask = compute_hot_pixel_mask(rgb)
   171|    hot_pixel_count = int(np.sum(hot_mask))
   172|    hot_pixel_ratio = float(hot_pixel_count / (h * w))
   173|
   174|    result = ImageAnalysisResult(
   175|        file_path=str(image_path),
   176|        scene_label=infer_scene_label(image_path),
   177|        time_label=infer_time_label(image_path),
   178|        zoom_label=infer_zoom_label(image_path),
   179|        width=w,
   180|        height=h,
   181|        mean_r=float(np.mean(r)),
   182|        mean_g=float(np.mean(g)),
   183|        mean_b=float(np.mean(b)),
   184|        std_r=float(np.std(r)),
   185|        std_g=float(np.std(g)),
   186|        std_b=float(np.std(b)),
   187|        mean_y=float(np.mean(y)),
   188|        std_y=float(np.std(y)),
   189|        mean_cr=float(np.mean(cr)),
   190|        mean_cb=float(np.mean(cb)),
   191|        tint_rg=float(np.mean(r) - np.mean(g)),
   192|        tint_bg=float(np.mean(b) - np.mean(g)),
   193|        edge_density=edge_density,
   194|        high_freq_energy=compute_high_freq_energy(gray),
   195|        dark_region_ratio=dark_region_ratio,
   196|        dark_noise_std=dark_noise_std,
   197|        hot_pixel_ratio=hot_pixel_ratio,
   198|        hot_pixel_count=hot_pixel_count,
   199|    )
   200|    return result
   201|
   202|
   203|def build_dataframe(results: Iterable[ImageAnalysisResult]) -> pd.DataFrame:
   204|    """Convert analysis results into a pandas DataFrame."""
   205|    return pd.DataFrame([asdict(result) for result in results])
   206|
   207|
   208|def save_group_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
   209|    """Save grouped summary statistics by time/zoom labels."""
   210|    group_cols = ["time_label", "zoom_label"]
   211|    metric_cols = [
   212|        "mean_r",
   213|        "mean_g",
   214|        "mean_b",
   215|        "std_r",
   216|        "std_g",
   217|        "std_b",
   218|        "mean_y",
   219|        "std_y",
   220|        "tint_rg",
   221|        "tint_bg",
   222|        "edge_density",
   223|        "high_freq_energy",
   224|        "dark_region_ratio",
   225|        "dark_noise_std",
   226|        "hot_pixel_ratio",
   227|        "hot_pixel_count",
   228|    ]
   229|    summary = df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
   230|    summary_path = output_dir / "group_summary.csv"
   231|    summary.to_csv(summary_path, index=False)
   232|    LOGGER.info("Saved grouped summary to %s", summary_path)
   233|    return summary
   234|
   235|
   236|def save_plots(df: pd.DataFrame, output_dir: Path) -> None:
   237|    """Generate a few compact plots for quick inspection."""
   238|    plot_dir = output_dir / "plots"
   239|    plot_dir.mkdir(parents=True, exist_ok=True)
   240|
   241|    plt.figure(figsize=(10, 6))
   242|    for label, group in df.groupby("time_label"):
   243|        plt.scatter(group["mean_y"], group["dark_noise_std"], label=label, alpha=0.7)
   244|    plt.xlabel("Mean Y")
   245|    plt.ylabel("Dark Region Noise Std")
   246|    plt.title("Brightness vs Dark-Region Noise")
   247|    plt.legend()
   248|    plt.tight_layout()
   249|    plt.savefig(plot_dir / "brightness_vs_dark_noise.png", dpi=150)
   250|    plt.close()
   251|
   252|    plt.figure(figsize=(10, 6))
   253|    for label, group in df.groupby("zoom_label"):
   254|        plt.scatter(group["edge_density"], group["hot_pixel_ratio"], label=label, alpha=0.7)
   255|    plt.xlabel("Edge Density")
   256|    plt.ylabel("Hot Pixel Ratio")
   257|    plt.title("Edge Density vs Hot Pixel Ratio")
   258|    plt.legend()
   259|    plt.tight_layout()
   260|    plt.savefig(plot_dir / "edge_vs_hot_pixel.png", dpi=150)
   261|    plt.close()
   262|
   263|    plt.figure(figsize=(10, 6))
   264|    plt.hist(df["tint_rg"], bins=20, alpha=0.6, label="R-G")
   265|    plt.hist(df["tint_bg"], bins=20, alpha=0.6, label="B-G")
   266|    plt.xlabel("Tint Bias")
   267|    plt.ylabel("Count")
   268|    plt.title("Color Tint Bias Distribution")
   269|    plt.legend()
   270|    plt.tight_layout()
   271|    plt.savefig(plot_dir / "tint_distribution.png", dpi=150)
   272|    plt.close()
   273|
   274|    LOGGER.info("Saved plots to %s", plot_dir)
   275|
   276|
   277|def render_summary_markdown(df: pd.DataFrame, summary: pd.DataFrame) -> str:
   278|    """Render a markdown summary for quick human review."""
   279|    top_hot = df.sort_values("hot_pixel_count", ascending=False).head(5)
   280|    lines = [
   281|        "# MC-G105 Capture Analysis Summary",
   282|        "",
   283|        f"- Total images: {len(df)}",
   284|        f"- Time labels: {sorted(df['time_label'].unique().tolist())}",
   285|        f"- Zoom labels: {sorted(df['zoom_label'].unique().tolist())}",
   286|        "",
   287|        "## Group Summary",
   288|        "",
   289|        summary.to_markdown(index=False),
   290|        "",
   291|        "## Top Hot-Pixel Candidates",
   292|        "",
   293|        top_hot[["file_path", "time_label", "zoom_label", "hot_pixel_count", "hot_pixel_ratio", "dark_noise_std", "tint_rg", "tint_bg"]].to_markdown(index=False),
   294|        "",
   295|        "## Interpretation Hints",
   296|        "",
   297|        "- `dark_noise_std`가 night/tele에서 높아지면 gain/저조도 노이즈 영향 가능성이 큼",
   298|        "- `tint_rg`, `tint_bg`가 특정 방향으로 지속적으로 치우치면 고정 tint bias 가능성이 큼",
   299|        "- `hot_pixel_count`가 dark scene에서 반복적으로 높으면 fixed defect pixel 가능성이 큼",
   300|        "- `edge_density`는 장면 복잡도 차이를 해석할 때 참고용으로 사용",
   301|    ]
   302|    return "\n".join(lines)
   303|
   304|
   305|def main() -> int:
   306|    """Run frame analysis and save machine/human-readable outputs."""
   307|    args = parse_args()
   308|    setup_logging(args.verbose)
   309|
   310|    input_dir = Path(args.input_dir).expanduser().resolve()
   311|    output_dir = Path(args.output_dir).expanduser().resolve()
   312|    output_dir.mkdir(parents=True, exist_ok=True)
   313|
   314|    if not input_dir.exists():
   315|        LOGGER.error("Input directory does not exist: %s", input_dir)
   316|        return 1
   317|
   318|    image_paths = collect_image_paths(input_dir, args.max_images)
   319|    if not image_paths:
   320|        LOGGER.error("No image files found under %s", input_dir)
   321|        return 1
   322|
   323|    LOGGER.info("Found %d images to analyze", len(image_paths))
   324|
   325|    results: list[ImageAnalysisResult] = []
   326|    for image_path in image_paths:
   327|        try:
   328|            results.append(analyze_image(image_path))
   329|        except Exception as exc:  # pragma: no cover - defensive logging path
   330|            LOGGER.warning("Skipping %s due to error: %s", image_path, exc)
   331|
   332|    if not results:
   333|        LOGGER.error("No images were successfully analyzed.")
   334|        return 1
   335|
   336|    df = build_dataframe(results)
   337|    per_image_path = output_dir / "per_image_metrics.csv"
   338|    df.to_csv(per_image_path, index=False)
   339|    LOGGER.info("Saved per-image metrics to %s", per_image_path)
   340|
   341|    summary = save_group_summary(df, output_dir)
   342|    save_plots(df, output_dir)
   343|
   344|    summary_markdown = render_summary_markdown(df, summary)
   345|    summary_path = output_dir / "summary.md"
   346|    summary_path.write_text(summary_markdown, encoding="utf-8")
   347|    LOGGER.info("Saved markdown summary to %s", summary_path)
   348|
   349|    return 0
   350|
   351|
   352|if __name__ == "__main__":
   353|    sys.exit(main())
   354|