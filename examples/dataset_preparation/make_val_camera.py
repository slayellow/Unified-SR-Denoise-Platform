"""Generate a denoise validation set through the camera model: <out_dir>/HR holds the targets, <out_dir>/LR the noisy inputs.

The target follows camera_model.target in the data config (camera / optics / source),
so the validation pairs are scored against the same target as training. The LR images
do not depend on the target mode: the same seed gives the same LR for every mode.

Deterministic: image i uses seed (base_seed + i), so the set regenerates bit-identically.

    PYTHONPATH=. python3 examples/dataset_preparation/make_val_camera.py --data_config configs/data/bicubicpp/denoise.yaml \
        --hr_root /mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val_denoise/HR \
        --out_dir /mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val_denoise_cam_a9nr5
"""

import argparse
import os

import cv2
import numpy as np
import yaml

from src.data.degradations import apply_camera_model, validate_camera_model_cfg

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_config', required=True, help='data config with an enabled camera_model block')
    parser.add_argument('--hr_root', required=True, help='folder of clean source images')
    parser.add_argument('--out_dir', required=True, help='output folder; HR/ and LR/ are created inside')
    parser.add_argument('--seed', type=int, default=2026, help='base seed; image i uses seed + i')
    args = parser.parse_args(args)

    with open(args.data_config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cam = (cfg or {}).get('camera_model', {}) or {}
    if not cam.get('enabled', False):
        raise ValueError(f"{args.data_config}: camera_model is missing or disabled")
    validate_camera_model_cfg(cam)

    names = [n for n in sorted(os.listdir(args.hr_root)) if os.path.splitext(n)[1].lower() in IMAGE_EXTS]
    if not names:
        raise FileNotFoundError(f"no images under {args.hr_root}")
    hr_dir, lr_dir = os.path.join(args.out_dir, 'HR'), os.path.join(args.out_dir, 'LR')
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    for i, name in enumerate(names):
        img = cv2.imread(os.path.join(args.hr_root, name), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot read {os.path.join(args.hr_root, name)}")
        lr, target = apply_camera_model(img, cam, rng=np.random.default_rng(args.seed + i))
        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(hr_dir, f'{stem}.png'), target)
        cv2.imwrite(os.path.join(lr_dir, f'{stem}.png'), lr)
    print(f'Images : {len(names)} -> {args.out_dir} (seed {args.seed})')
    print('Make Val Finish!')


if __name__ == '__main__':
    main()
