from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create luma-guarded Teacher outputs by matching the low-frequency "
            "Y channel of Teacher results to the original input frames."
        )
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Directory containing original input frames.",
    )
    parser.add_argument(
        "--teacher_dir",
        type=str,
        required=True,
        help="Directory containing Teacher inference outputs with matching file names.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save luma-guarded outputs.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=24.0,
        help="Gaussian sigma for low-frequency luma estimation. Larger values preserve only broader tone.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Correction strength. 1.0 fully matches low-frequency luma; 0.5 applies half correction.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Match files recursively under teacher_dir and preserve relative paths.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="",
        help="Optional CSV path for before/after luma-shift metrics.",
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


def collect_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in root.glob(pattern) if path.suffix.lower() in IMAGE_SUFFIXES)


def lowpass_luma(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.full_like(y, y.mean(), dtype=np.float32)
    return cv2.GaussianBlur(
        y,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )


def apply_luma_guard(original_bgr: np.ndarray, teacher_bgr: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    if original_bgr.shape[:2] != teacher_bgr.shape[:2]:
        raise ValueError(f"Shape mismatch: original={original_bgr.shape}, teacher={teacher_bgr.shape}")

    original_ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    teacher_ycrcb = cv2.cvtColor(teacher_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    original_low = lowpass_luma(original_ycrcb[:, :, 0], sigma)
    teacher_low = lowpass_luma(teacher_ycrcb[:, :, 0], sigma)
    correction = (original_low - teacher_low) * strength

    guarded = teacher_ycrcb.copy()
    guarded[:, :, 0] = np.clip(guarded[:, :, 0] + correction, 0.0, 255.0)

    return cv2.cvtColor(guarded.round().astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def luma_metrics(original_bgr: np.ndarray, teacher_bgr: np.ndarray, guarded_bgr: np.ndarray) -> dict[str, float]:
    original_y = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    teacher_y = cv2.cvtColor(teacher_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    guarded_y = cv2.cvtColor(guarded_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

    teacher_delta = teacher_y - original_y
    guarded_delta = guarded_y - original_y
    return {
        "teacher_mean_delta_y": float(teacher_delta.mean()),
        "teacher_mae_y": float(np.abs(teacher_delta).mean()),
        "guarded_mean_delta_y": float(guarded_delta.mean()),
        "guarded_mae_y": float(np.abs(guarded_delta).mean()),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    original_dir = Path(args.original_dir)
    teacher_dir = Path(args.teacher_dir)
    output_dir = Path(args.output_dir)

    teacher_paths = collect_images(teacher_dir, recursive=args.recursive)
    if not teacher_paths:
        raise FileNotFoundError(f"No images found under teacher_dir: {teacher_dir}")

    rows: list[dict[str, float | str]] = []
    written = 0
    skipped = 0

    for teacher_path in teacher_paths:
        rel_path = teacher_path.relative_to(teacher_dir)
        original_path = original_dir / rel_path
        if not original_path.exists():
            LOGGER.warning("Skipping %s: missing original %s", rel_path, original_path)
            skipped += 1
            continue

        original_bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        teacher_bgr = cv2.imread(str(teacher_path), cv2.IMREAD_COLOR)
        if original_bgr is None or teacher_bgr is None:
            LOGGER.warning("Skipping %s: failed to read original or teacher image", rel_path)
            skipped += 1
            continue

        guarded_bgr = apply_luma_guard(
            original_bgr=original_bgr,
            teacher_bgr=teacher_bgr,
            sigma=args.sigma,
            strength=args.strength,
        )

        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), guarded_bgr)

        metrics = luma_metrics(original_bgr, teacher_bgr, guarded_bgr)
        rows.append({"file": str(rel_path), **metrics})
        written += 1

    if args.metrics_csv:
        csv_path = Path(args.metrics_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "file",
            "teacher_mean_delta_y",
            "teacher_mae_y",
            "guarded_mean_delta_y",
            "guarded_mae_y",
        ]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    LOGGER.info("Wrote %d images to %s; skipped %d.", written, output_dir, skipped)
    if args.metrics_csv:
        LOGGER.info("Metrics saved to %s.", args.metrics_csv)


if __name__ == "__main__":
    main()
