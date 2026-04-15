"""
Denoise Validation 데이터 생성 스크립트

val/HR 이미지에 degradation pipeline (scale=1)을 적용하여
val_denoise/HR (clean) + val_denoise/LR (noisy) 쌍을 생성합니다.

Usage:
    python3 examples/generate_denoise_val.py \
        --hr_dir /mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val/HR \
        --out_dir /mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val_denoise \
        --data_config configs/data/denoise_eo.yaml \
        --seed 42
"""
import os
import sys
import argparse
import glob
import random
import shutil

import cv2
import numpy as np
import yaml

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.datasets import SRDataset


def main():
    parser = argparse.ArgumentParser(description="Generate Denoise Validation Data")
    parser.add_argument("--hr_dir", type=str,
                        default="/mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val/HR",
                        help="Clean HR 이미지 디렉토리")
    parser.add_argument("--out_dir", type=str,
                        default="/mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val_denoise",
                        help="출력 디렉토리 (하위에 HR/, LR/ 생성)")
    parser.add_argument("--data_config", type=str,
                        default="configs/data/denoise_eo.yaml",
                        help="열화 파이프라인 설정 yaml")
    parser.add_argument("--seed", type=int, default=42,
                        help="재현 가능한 노이즈를 위한 시드")
    args = parser.parse_args()

    # Config 로드
    with open(args.data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)

    # 시드 고정
    random.seed(args.seed)
    np.random.seed(args.seed)

    # SRDataset의 degradation pipeline 재사용 (scale=1)
    sr_ds = SRDataset(dataset_root=[], scale_factor=1, config=data_cfg)

    # 출력 디렉토리 생성
    out_hr = os.path.join(args.out_dir, 'HR')
    out_lr = os.path.join(args.out_dir, 'LR')
    os.makedirs(out_hr, exist_ok=True)
    os.makedirs(out_lr, exist_ok=True)

    # HR 이미지 수집
    extensions = ('.png', '.jpg', '.jpeg')
    hr_files = sorted([
        f for f in glob.glob(os.path.join(args.hr_dir, '*'))
        if f.lower().endswith(extensions)
    ])

    print(f"HR 이미지: {len(hr_files)}장")
    print(f"열화 config: {args.data_config}")
    print(f"출력 경로: {args.out_dir}")
    print(f"시드: {args.seed}")
    print()

    for i, hr_path in enumerate(hr_files):
        fname = os.path.basename(hr_path)
        stem, _ = os.path.splitext(fname)
        out_name = f"{stem}.png"

        img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [SKIP] 읽기 실패: {hr_path}")
            continue

        # 열화 적용 (전체 이미지, crop 없음)
        noisy = sr_ds.degradation_pipeline(img)

        # 저장 (HR은 원본 복사, LR은 noisy)
        cv2.imwrite(os.path.join(out_hr, out_name), img)
        cv2.imwrite(os.path.join(out_lr, out_name), noisy)

        if (i + 1) % 20 == 0 or (i + 1) == len(hr_files):
            print(f"  [{i+1}/{len(hr_files)}] {out_name}")

    print(f"\n완료: HR {len(hr_files)}장 → {out_hr}")
    print(f"완료: LR {len(hr_files)}장 → {out_lr}")


if __name__ == '__main__':
    main()
