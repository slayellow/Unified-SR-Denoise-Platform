"""Generate a validation LR set from val HR with a training degradation recipe.

The legacy val/LRx{2,3,4} folders were generated long ago with an unrecorded,
much heavier recipe (bicubic-deviation RMS ~20 DN vs ~2 DN for the weak-noise
recipe). Judging a finetune that adapts to the weak-noise distribution against
that legacy set measured the wrong thing -- the owner's call is that the
weak-noise recipe is closer to the field, so Phase-1 and Phase-2 validate on LR
generated with exactly the recipe (and softening split) they train on.

Deterministic: each image is degraded under seed (base_seed + index), so the
set regenerates bit-identically and eo/ir sets differ only by their recipe.
The val sets on disk were built before the 2026-10-16 recipe fixes (noise
rounding, see docs/CONFIG_RATIONALE.md 4.4); they regenerate bit-identically
only at commit de502f6.

--denoise_checkpoint mirrors the deployment softening the trainer applies (see
Trainer._soften). Deployment feeds SR through a denoise net on one camera and
not on another, and the same weights serve both, so training splits each batch
between the two; a validation set drawn from only one of them would score a
distribution the model is not being trained for. Here the split is by index
rather than by coin flip -- every other image is denoised -- so the set stays
reproducible and both paths are represented in the same proportion.

    python3 examples/dataset_preparation/make_val_lr.py --data_config configs/data/bicubicpp/sr_finetune_eo_phase-2.yaml \
        --scale 3 --hr_root .../val/HR --out_dir .../val/LRx3_weaknoise_eo_v3 \
        --denoise_checkpoint checkpoints/csuav_deploy/train_svfocusdenoise_adv_x1_dim32_block3_epoch300_lr1e-4_charbonnier_edge_ssim_hotpixel_260309/best.pth
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.datasets import apply_configured_degradation  # noqa: E402


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_config", required=True)
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--hr_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--denoise_checkpoint", default=None,
                    help="deployed denoise net; every other image goes through it")
    ap.add_argument("--denoise_fraction", type=float, default=0.5,
                    help="share of images to soften (matches train denoise_prob)")
    ap.add_argument("--denoise_noise_sigma", default="2,15",
                    help="sensor noise added before the denoise net, matching "
                         "Trainer._soften; empty string disables")
    args = ap.parse_args(args)

    denoise = None
    if args.denoise_checkpoint:
        import torch
        from src.models import build_model
        from src.engine.checkpoint import normalize_state_dict
        ck = torch.load(args.denoise_checkpoint, map_location="cpu", weights_only=False)
        mcfg = (ck.get("config") or {}).get("model")
        if not mcfg:
            raise ValueError(f"{args.denoise_checkpoint} carries no model config")
        denoise = build_model(dict(mcfg))
        denoise.load_state_dict(normalize_state_dict(ck.get("model_state_dict", ck)))
        if hasattr(denoise, "switch_to_deploy"):
            denoise.switch_to_deploy()
        denoise.eval()

    cfg = yaml.safe_load(open(args.data_config, encoding="utf-8"))
    hrs = sorted(p for p in glob.glob(os.path.join(args.hr_root, "*"))
                 if p.lower().endswith((".png", ".jpg", ".jpeg")))
    if not hrs:
        raise FileNotFoundError(f"no HR images under {args.hr_root}")
    os.makedirs(args.out_dir, exist_ok=True)

    for i, h in enumerate(hrs):
        hr = cv2.imread(h, cv2.IMREAD_COLOR)
        H, W = hr.shape[:2]
        hr = hr[:H - H % args.scale, :W - W % args.scale]
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)
        lr = apply_configured_degradation(img_hr=hr, cfg=cfg, scale_factor=args.scale)
        # Softened on a fixed stride, not at random: reproducibility is the
        # whole point of this script, and a gate whose membership changes
        # between runs would move the score without the model moving.
        if denoise is not None and args.denoise_fraction > 0:
            stride = max(int(round(1.0 / args.denoise_fraction)), 1)
            if i % stride == 0:
                import torch
                t = torch.from_numpy(lr[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255.0
                if args.denoise_noise_sigma:
                    lo, hi = (float(v) for v in args.denoise_noise_sigma.split(","))
                    sig = random.uniform(lo, hi) / 255.0
                    t = (t + torch.randn(t.shape, generator=torch.Generator().manual_seed(
                        args.seed + i)) * sig).clamp(0, 1)
                with torch.no_grad():
                    o = denoise(t)
                    if isinstance(o, tuple):
                        o = o[0]
                o = (o.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                lr = o[:, :, ::-1].copy()
        out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(h))[0] + ".png")
        cv2.imwrite(out, lr)
    softened = "none" if denoise is None else f"{args.denoise_fraction:.0%} via {os.path.basename(args.denoise_checkpoint)}"
    print(f"{args.out_dir}: {len(hrs)} images (seed {args.seed}, recipe {args.data_config}, softened {softened})")


if __name__ == "__main__":
    main()
