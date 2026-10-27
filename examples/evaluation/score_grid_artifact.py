"""Score the upsampler-period grid (checkerboard) an SR model leaves in its output.

A PixelShuffle or strided ConvTranspose writes each output pixel from one of
`period` x `period` phases. When the phases disagree the output carries a
pattern that repeats every `period` pixels, which shows as a spike at k/period
cycles/px in the power spectrum. The score is that spike against the median of
the bins around it: ~1 means no grid, the 2026-09-22 IR x4 board capture scored
72 (1/4 horizontal). A phase mean over the whole frame (the 09-22 metric) cancels
out, because the pattern's sign follows local content.

Sharpness is reported next to it so a grid fix that just blurs is visible: the
mean Sobel magnitude of the model output over that of a bicubic upscale of the
same input.

    PYTHONPATH=. python3 examples/evaluation/score_grid_artifact.py \
        --ckpt checkpoints/bicubicpp_deploy/finetune_bicubicplusplus_ir_sr_x4_dim32_v3/best.pth

Design and test cases: docs/modules/20260928-MOD-GridScore.md.
"""

import argparse
import hashlib
import json
import os

import cv2
import numpy as np
import torch

from src.engine.checkpoint import normalize_state_dict
from src.models import build_model

DEFAULT_FRAMES = "results/release_v3/eval/ir92_raw_600"
DEFAULT_MANIFEST = "results/release_v3/eval/ir92_p2_frames.sha256"


def power_spectrum(img: np.ndarray) -> np.ndarray:
    """Hann-windowed power spectrum of a 2-D float image."""
    img = img.astype(np.float64) - img.mean()
    win = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    return np.abs(np.fft.fft2(img * win)) ** 2


def peak_ratio(spec: np.ndarray, fy: float, fx: float) -> float:
    """Max power within 1 bin of (fy, fx) cycles/px over the median of a 17x17 ring around it."""
    h, w = spec.shape
    y, x = int(round(fy * h)) % h, int(round(fx * w)) % w
    ys = [(y + d) % h for d in range(-8, 9)]
    xs = [(x + d) % w for d in range(-8, 9)]
    nb = spec[np.ix_(ys, xs)].copy()
    peak = nb[7:10, 7:10].max()
    nb[6:11, 6:11] = np.nan
    return float(peak / np.nanmedian(nb))


def grid_score(spec: np.ndarray, period: int) -> dict:
    """Peak ratios at every k/period <= 1/2 along h (x only), v (y only) and d (both); raises ValueError for period < 2."""
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    out = {}
    for k in range(1, period // 2 + 1):
        f = k / period
        name = f"{k}/{period}" if k * 2 != period or period == 2 else "1/2"
        out[f"{name}h"] = peak_ratio(spec, 0, f)
        out[f"{name}v"] = peak_ratio(spec, f, 0)
        out[f"{name}d"] = peak_ratio(spec, f, f)
    return out


def sharpness_ratio(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean Sobel magnitude of img over that of ref (same size, float)."""

    def mag(a):
        a = a.astype(np.float32)
        return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)).mean()

    return float(mag(img) / mag(ref))


def load_model(ckpt: str, device: str) -> tuple:
    """Model from ckpt's model_state_dict (EMA when training saved it) and the model config saved in it.

    The saved config, not a train_config.yaml beside the file: a checkpoint copied into a release folder has
    no yaml beside it, and a yaml beside it can belong to another run.
    """
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = (sd.get("config") or {}).get("model")
    if cfg is None:
        raise ValueError(f"{ckpt} has no saved model config (config.model); it cannot be scored")
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(normalize_state_dict(sd["model_state_dict"]))
    return model, cfg["scale"], sd.get("epoch")


def upscale(model, lr_rgb: np.ndarray, device: str) -> np.ndarray:
    """uint8 HxWx3 RGB in, float luma (0..255, rounded like an 8-bit output) out."""
    t = torch.from_numpy(np.ascontiguousarray(lr_rgb)).permute(2, 0, 1)[None].float().to(device) / 255
    with torch.no_grad():
        o = model(t)
    o = (o[0] if isinstance(o, tuple) else o).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return np.round(cv2.cvtColor(o * 255, cv2.COLOR_RGB2GRAY))


def score(model, scale: int, frames: list, crop: tuple, device: str) -> dict:
    """Grid score of the summed spectrum over frames, plus the median sharpness ratio vs bicubic."""
    x, y, w, h = crop
    spec, sharp = 0, []
    for p in frames:
        lr = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)[y : y + h, x : x + w]
        sr = upscale(model, lr, device)
        bic = cv2.resize(cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY), (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        spec = spec + power_spectrum(sr)
        sharp.append(sharpness_ratio(sr, bic.astype(np.float32)))
    out = grid_score(spec, scale)
    out["sharpness"] = float(np.median(sharp))
    return out


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="checkpoint with its model config saved in it (every Trainer checkpoint)")
    parser.add_argument("--frames", default=DEFAULT_FRAMES, help="directory of raw sensor PNG frames")
    parser.add_argument("--crop", default="160,166,320,180", help="x,y,w,h of the LR crop the board feeds the model")
    parser.add_argument("--index", default="0:600:12", help="start:stop:step over the sorted frame list")
    parser.add_argument("--min_frames", type=int, default=50, help="refuse to score fewer frames than this")
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST, help="sha256sum list the selection must equal, in order; '' skips"
    )
    parser.add_argument("--device", default="cpu", help="cpu keeps IEEE FP32; cuda convs may run in TF32")
    parser.add_argument("--json", default=None, help="write the scores here")
    args = parser.parse_args(args)

    names = sorted(n for n in os.listdir(args.frames) if n.endswith(".png"))
    frames = [os.path.join(args.frames, n) for n in names[slice(*map(int, args.index.split(":")))]]
    if len(frames) < max(args.min_frames, 1):
        raise ValueError(
            f"{len(frames)} frames selected from {args.frames} with --index {args.index}, need {args.min_frames}"
        )
    if args.manifest:
        with open(args.manifest) as f:
            want = f.read().split()
        got = []
        for p in frames:
            with open(p, "rb") as f:
                got += [hashlib.sha256(f.read()).hexdigest(), os.path.basename(p)]
        if got != want:
            raise ValueError(
                f"frames selected from {args.frames} with --index {args.index} differ from {args.manifest}"
            )
    model, scale, epoch = load_model(args.ckpt, args.device)
    out = score(model, scale, frames, tuple(map(int, args.crop.split(","))), args.device)
    out.update(
        ckpt=args.ckpt,
        epoch=epoch,
        n_frames=len(frames),
        frames=args.frames,
        index=args.index,
        manifest=args.manifest,
        device=args.device,
    )
    for k, v in out.items():
        print(f"{k} : {v:.2f}" if isinstance(v, float) else f"{k} : {v}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    print("Grid Score Finish!")


if __name__ == "__main__":
    main()
