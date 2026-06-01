from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


LOGGER = logging.getLogger(__name__)
DEFAULT_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
torch = None
build_model = None


def ensure_runtime_imports() -> None:
    global torch
    global build_model

    if torch is None:
        import torch as torch_module

        torch = torch_module
    if build_model is None:
        from src.models import build_model as build_model_fn

        build_model = build_model_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run image inference over an MC-G105 capture tree and preserve the "
            "input directory structure under the output root."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="results/mc_g105_capture_frames",
        help="Root directory containing MC-G105 input frames.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for inference outputs. Relative input paths are preserved below this path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: cuda, cuda:0, a numeric cuda index, or cpu.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 inference on CUDA.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=0,
        help="Optional tile size for memory-safe scale-1 inference. 0 disables tiling.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=32,
        help="Tile overlap in pixels when --tile_size is enabled.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files whose output path already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of images to process. 0 means all images.",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        default=",".join(DEFAULT_SUFFIXES),
        help="Comma-separated image suffix allowlist.",
    )
    parser.add_argument(
        "--status_csv",
        type=str,
        default="",
        help="Optional CSV path for per-file inference status.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List how many files would be processed without loading the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def resolve_device(device_arg: str) -> torch.device:
    ensure_runtime_imports()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_arg}")
        LOGGER.warning("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    if device_arg.startswith("cuda"):
        LOGGER.warning("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ensure_runtime_imports()

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    LOGGER.info("Loading model: %s", config["model"]["name"])
    model = build_model(config["model"]).to(device)

    LOGGER.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(normalize_state_dict(state_dict))

    if hasattr(model, "switch_to_deploy"):
        LOGGER.info("Switching model to deploy mode.")
        model.switch_to_deploy()

    model.eval()
    return model, config


def collect_images(input_root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(path for path in input_root.rglob("*") if path.suffix.lower() in suffixes)


def load_image(path: Path, device: torch.device, use_fp16: bool) -> torch.Tensor:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device)
    if use_fp16 and device.type == "cuda":
        tensor = tensor.half()
    return tensor


def save_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.squeeze(0).detach().float().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).round().astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def run_model(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    output = model(image)
    if isinstance(output, tuple):
        output = output[0]
    return output


def make_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int,
    tile_overlap: int,
) -> torch.Tensor:
    _, channels, height, width = image.shape
    stride = max(tile_size - tile_overlap, 1)
    y_starts = make_starts(height, tile_size, stride)
    x_starts = make_starts(width, tile_size, stride)

    output = torch.zeros((1, channels, height, width), dtype=torch.float32, device=image.device)
    weight = torch.zeros((1, 1, height, width), dtype=torch.float32, device=image.device)

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tile = image[:, :, y0:y1, x0:x1]
            tile_output = run_model(model, tile).float()
            output[:, :, y0:y1, x0:x1] += tile_output
            weight[:, :, y0:y1, x0:x1] += 1.0

    return output / weight.clamp_min(1.0)


def infer_one(
    model: torch.nn.Module,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    use_fp16: bool,
    tile_size: int,
    tile_overlap: int,
) -> tuple[int, int, float]:
    image = load_image(image_path, device=device, use_fp16=use_fp16)
    _, _, height, width = image.shape

    start = time.time()
    if tile_size > 0:
        output = run_tiled(model, image, tile_size=tile_size, tile_overlap=tile_overlap)
    else:
        output = run_model(model, image)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000.0

    save_image(output, output_path)
    return width, height, elapsed_ms


def write_status_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["relative_path", "status", "width", "height", "elapsed_ms", "error"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    suffixes = {suffix.strip().lower() for suffix in args.suffixes.split(",") if suffix.strip()}
    image_paths = collect_images(input_root, suffixes=suffixes)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    LOGGER.info("Input root: %s", input_root)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Found %d image(s).", len(image_paths))
    if args.dry_run:
        for path in image_paths[:10]:
            LOGGER.info("Would process: %s -> %s", path, output_root / path.relative_to(input_root))
        if len(image_paths) > 10:
            LOGGER.info("... and %d more.", len(image_paths) - 10)
        return

    device = resolve_device(args.device)
    use_fp16 = args.fp16 and device.type == "cuda"
    model, config = load_model(Path(args.config), Path(args.checkpoint), device)
    if use_fp16:
        LOGGER.info("Using FP16 inference.")
        model.half()

    if args.tile_size > 0 and int(config["model"].get("scale", 1)) != 1:
        raise ValueError("--tile_size currently supports scale=1 models only.")

    rows: list[dict[str, str | int | float]] = []
    processed = 0
    skipped = 0
    failed = 0

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="MC-G105 tree inference"):
            rel_path = image_path.relative_to(input_root)
            output_path = output_root / rel_path

            if args.skip_existing and output_path.exists():
                skipped += 1
                rows.append({
                    "relative_path": str(rel_path),
                    "status": "skipped",
                    "width": "",
                    "height": "",
                    "elapsed_ms": "",
                    "error": "",
                })
                continue

            try:
                width, height, elapsed_ms = infer_one(
                    model=model,
                    image_path=image_path,
                    output_path=output_path,
                    device=device,
                    use_fp16=use_fp16,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                )
                processed += 1
                rows.append({
                    "relative_path": str(rel_path),
                    "status": "ok",
                    "width": width,
                    "height": height,
                    "elapsed_ms": f"{elapsed_ms:.3f}",
                    "error": "",
                })
            except Exception as exc:  # noqa: BLE001
                failed += 1
                LOGGER.exception("Failed on %s", image_path)
                rows.append({
                    "relative_path": str(rel_path),
                    "status": "failed",
                    "width": "",
                    "height": "",
                    "elapsed_ms": "",
                    "error": str(exc),
                })

    if args.status_csv:
        write_status_csv(Path(args.status_csv), rows)

    LOGGER.info("Done. processed=%d skipped=%d failed=%d output=%s", processed, skipped, failed, output_root)
    if args.status_csv:
        LOGGER.info("Status CSV saved to %s", args.status_csv)


if __name__ == "__main__":
    main()
