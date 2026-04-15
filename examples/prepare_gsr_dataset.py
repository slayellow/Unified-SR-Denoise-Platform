"""
Guided Super Resolution (GSR) 데이터셋 준비 스크립트

기능:
1. FLIR 데이터셋 파일명 통일 (HR_unified, RGB_unified 폴더로 복사)
   - HR: FLIR_XXXXX_PreviewData.jpeg -> FLIR_XXXXX.png
   - RGB: FLIR_XXXXX_RGB.jpg -> FLIR_XXXXX.png
2. HR(IR) 이미지를 bicubic downsampling하여 LRx2, LRx4, LRx8 생성
3. 데이터 무결성 검증

사용법:
    python examples/prepare_gsr_dataset.py --root /path/to/GSR_Dataset
    python examples/prepare_gsr_dataset.py --root /path/to/GSR_Dataset --step unify   # FLIR 리네임만
    python examples/prepare_gsr_dataset.py --root /path/to/GSR_Dataset --step lr       # LR 생성만
    python examples/prepare_gsr_dataset.py --root /path/to/GSR_Dataset --step verify   # 검증만
"""

import os
import re
import argparse
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


# ──────────────────────────────────────────────
# 데이터셋 정의
# ──────────────────────────────────────────────

DATASET_CONFIG = {
    "FLIR": {
        "splits": ["train"],
        "hr_dir": "HR_unified",  # 통일된 폴더 사용
        "rgb_dir": "RGB_unified",
    },
    "LLVIP": {
        "splits": ["train", "val", "test"],
        "hr_dir": "HR",
        "rgb_dir": "RGB",
    },
    "M3FD": {
        "splits": ["train", "val"],
        "hr_dir": "HR",
        "rgb_dir": "RGB",
    },
}

SCALES = [2, 4, 8]

FLIR_ID_PATTERN = re.compile(r"FLIR_(\d{5})_")


# ──────────────────────────────────────────────
# Step 1: FLIR 파일명 통일
# ──────────────────────────────────────────────

def extract_flir_id(filename):
    """FLIR_XXXXX_PreviewData.jpeg 또는 FLIR_XXXXX_RGB.jpg에서 숫자 ID 추출."""
    m = FLIR_ID_PATTERN.search(filename)
    return m.group(1) if m else None


def copy_and_convert(args_tuple):
    """단일 이미지를 읽어서 PNG로 저장."""
    src_path, dst_path = args_tuple
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"WARN: 읽기 실패 - {src_path}"
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(str(dst_path), img)
    return None


def unify_flir(root, workers=8):
    """FLIR HR/RGB 파일명을 통일하여 HR_unified/RGB_unified에 복사."""
    flir_root = Path(root) / "FLIR"
    hr_src = flir_root / "HR" / "train"
    rgb_src = flir_root / "RGB" / "train"

    if not hr_src.exists() or not rgb_src.exists():
        print(f"ERROR: FLIR 원본 폴더를 찾을 수 없습니다: {hr_src} 또는 {rgb_src}")
        return

    hr_dst = flir_root / "HR_unified" / "train"
    rgb_dst = flir_root / "RGB_unified" / "train"

    # ID -> 파일 경로 매핑
    hr_files = {extract_flir_id(f.name): f for f in hr_src.iterdir() if f.is_file()}
    rgb_files = {extract_flir_id(f.name): f for f in rgb_src.iterdir() if f.is_file()}

    # None 키 제거 (파싱 실패)
    hr_files.pop(None, None)
    rgb_files.pop(None, None)

    # 공통 ID만 사용
    common_ids = sorted(set(hr_files.keys()) & set(rgb_files.keys()))
    hr_only = set(hr_files.keys()) - set(rgb_files.keys())
    rgb_only = set(rgb_files.keys()) - set(hr_files.keys())

    if hr_only:
        print(f"WARN: HR에만 존재하는 ID {len(hr_only)}개 (스킵): {sorted(hr_only)[:5]}...")
    if rgb_only:
        print(f"WARN: RGB에만 존재하는 ID {len(rgb_only)}개 (스킵): {sorted(rgb_only)[:5]}...")

    print(f"FLIR: 공통 ID {len(common_ids)}개 처리 시작")

    # 복사 작업 목록 생성
    tasks = []
    for fid in common_ids:
        new_name = f"FLIR_{fid}.png"
        tasks.append((hr_files[fid], str(hr_dst / new_name)))
        tasks.append((rgb_files[fid], str(rgb_dst / new_name)))

    os.makedirs(hr_dst, exist_ok=True)
    os.makedirs(rgb_dst, exist_ok=True)

    # 멀티프로세싱으로 복사
    with Pool(workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(copy_and_convert, tasks),
            total=len(tasks),
            desc="FLIR 파일명 통일",
        ))

    warnings = [r for r in results if r is not None]
    if warnings:
        print(f"\n경고 {len(warnings)}건:")
        for w in warnings[:10]:
            print(f"  {w}")

    print(f"FLIR 통일 완료: HR_unified={len(list(hr_dst.iterdir()))}, RGB_unified={len(list(rgb_dst.iterdir()))}")


# ──────────────────────────────────────────────
# Step 2: Bicubic LR 생성
# ──────────────────────────────────────────────

def downscale_single(args_tuple):
    """단일 이미지를 bicubic downsampling하여 저장."""
    src_path, dst_path, scale = args_tuple
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"WARN: 읽기 실패 - {src_path}"

    h, w = img.shape[:2]
    new_w, new_h = w // scale, h // scale

    if new_w < 1 or new_h < 1:
        return f"WARN: 이미지가 너무 작아 x{scale} 다운샘플링 불가 - {src_path} ({w}x{h})"

    lr = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(str(dst_path), lr)
    return None


def generate_lr(root, workers=8):
    """각 데이터셋의 HR(IR) 이미지를 bicubic downsampling하여 LR 생성."""
    root = Path(root)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    for dataset_name, cfg in DATASET_CONFIG.items():
        dataset_dir = root / dataset_name
        hr_dir_name = cfg["hr_dir"]

        for split in cfg["splits"]:
            hr_dir = dataset_dir / hr_dir_name / split
            if not hr_dir.exists():
                print(f"WARN: HR 디렉토리 없음, 스킵 - {hr_dir}")
                continue

            hr_files = sorted([
                f for f in hr_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_exts
            ])

            if not hr_files:
                print(f"WARN: 이미지 없음, 스킵 - {hr_dir}")
                continue

            for scale in SCALES:
                lr_dir = dataset_dir / f"LRx{scale}" / split
                os.makedirs(lr_dir, exist_ok=True)

                # 이미 생성된 파일 수 확인 (이어하기 지원)
                existing = set(f.stem for f in lr_dir.iterdir() if f.is_file())
                tasks = []
                for hr_file in hr_files:
                    out_name = f"{hr_file.stem}.png"
                    if hr_file.stem in existing:
                        continue  # 이미 존재하면 스킵
                    tasks.append((str(hr_file), str(lr_dir / out_name), scale))

                if not tasks:
                    print(f"  {dataset_name}/{split}/LRx{scale}: 이미 완료 ({len(hr_files)}장)")
                    continue

                print(f"  {dataset_name}/{split}/LRx{scale}: {len(tasks)}장 생성 중...")

                with Pool(workers) as pool:
                    results = list(tqdm(
                        pool.imap_unordered(downscale_single, tasks),
                        total=len(tasks),
                        desc=f"{dataset_name}/{split}/LRx{scale}",
                    ))

                warnings = [r for r in results if r is not None]
                if warnings:
                    for w in warnings[:5]:
                        print(f"    {w}")


# ──────────────────────────────────────────────
# Step 3: 데이터 무결성 검증
# ──────────────────────────────────────────────

def verify_dataset(root):
    """데이터셋 무결성 검증 및 요약 테이블 출력."""
    root = Path(root)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    print("\n" + "=" * 90)
    print(f"{'Dataset':<10} {'Split':<8} {'HR':>6} {'RGB':>6} {'LRx2':>6} {'LRx4':>6} {'LRx8':>6} {'HR res':>12} {'LRx2 res':>12}")
    print("-" * 90)

    all_ok = True

    for dataset_name, cfg in DATASET_CONFIG.items():
        dataset_dir = root / dataset_name
        hr_dir_name = cfg["hr_dir"]
        rgb_dir_name = cfg["rgb_dir"]

        for split in cfg["splits"]:
            hr_dir = dataset_dir / hr_dir_name / split
            rgb_dir = dataset_dir / rgb_dir_name / split

            def count_images(d):
                if not d.exists():
                    return 0
                return len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in image_exts])

            hr_count = count_images(hr_dir)
            rgb_count = count_images(rgb_dir)
            lr_counts = {}
            for s in SCALES:
                lr_dir = dataset_dir / f"LRx{s}" / split
                lr_counts[s] = count_images(lr_dir)

            # HR 해상도 샘플
            hr_res_str = "-"
            lr2_res_str = "-"
            hr_files = sorted([f for f in hr_dir.iterdir() if f.is_file() and f.suffix.lower() in image_exts]) if hr_dir.exists() else []
            if hr_files:
                sample = cv2.imread(str(hr_files[0]), cv2.IMREAD_UNCHANGED)
                if sample is not None:
                    h, w = sample.shape[:2]
                    hr_res_str = f"{w}x{h}"

            lr2_dir = dataset_dir / "LRx2" / split
            lr2_files = sorted([f for f in lr2_dir.iterdir() if f.is_file() and f.suffix.lower() in image_exts]) if lr2_dir.exists() else []
            if lr2_files:
                sample = cv2.imread(str(lr2_files[0]), cv2.IMREAD_UNCHANGED)
                if sample is not None:
                    h, w = sample.shape[:2]
                    lr2_res_str = f"{w}x{h}"

            # 파일 수 일치 검증
            expected = hr_count
            mismatch = False
            if rgb_count != expected:
                mismatch = True
            for s in SCALES:
                if lr_counts[s] != expected:
                    mismatch = True

            mark = " ✗" if mismatch else ""

            print(f"{dataset_name:<10} {split:<8} {hr_count:>6} {rgb_count:>6} "
                  f"{lr_counts[2]:>6} {lr_counts[4]:>6} {lr_counts[8]:>6} "
                  f"{hr_res_str:>12} {lr2_res_str:>12}{mark}")

            if mismatch:
                all_ok = False

            # FLIR: HR_unified와 RGB_unified 파일명 매칭 검증
            if dataset_name == "FLIR":
                hr_names = set(f.name for f in hr_dir.iterdir() if f.is_file()) if hr_dir.exists() else set()
                rgb_names = set(f.name for f in rgb_dir.iterdir() if f.is_file()) if rgb_dir.exists() else set()
                if hr_names != rgb_names:
                    diff = hr_names.symmetric_difference(rgb_names)
                    print(f"  WARN: FLIR HR/RGB 파일명 불일치 {len(diff)}건")
                    all_ok = False

    print("=" * 90)
    if all_ok:
        print("✓ 모든 데이터셋 검증 통과")
    else:
        print("✗ 일부 불일치 발견 - 위 표의 ✗ 항목 확인")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="GSR 데이터셋 준비 (FLIR 리네임 + LR 생성)")
    parser.add_argument("--root", type=str, required=True,
                        help="GSR_Dataset 루트 경로 (FLIR, LLVIP, M3FD 포함)")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "unify", "lr", "verify"],
                        help="실행할 단계 (default: all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="멀티프로세싱 워커 수 (default: 8)")
    return parser.parse_args()


def main():
    args = get_args()
    root = args.root

    # 데이터셋 존재 확인
    for name in DATASET_CONFIG:
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            print(f"WARN: 데이터셋 폴더 없음 - {d}")

    if args.step in ("all", "unify"):
        print("\n[Step 1] FLIR 파일명 통일")
        print("-" * 50)
        unify_flir(root, workers=args.workers)

    if args.step in ("all", "lr"):
        print("\n[Step 2] Bicubic LR 생성 (x2, x4, x8)")
        print("-" * 50)
        generate_lr(root, workers=args.workers)

    if args.step in ("all", "verify"):
        print("\n[Step 3] 데이터 무결성 검증")
        print("-" * 50)
        verify_dataset(root)


if __name__ == "__main__":
    main()
