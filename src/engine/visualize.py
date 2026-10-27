"""Run artifacts rendered during training, ultralytics-style.

Everything here draws into checkpoints/<save_name>/plots/:

  results.png          metric curves, redrawn every epoch from training_log.csv
  train_batch.png      one degraded LR|HR training batch, rendered once at start
                       (the visual counterpart of the [Degradation] log line --
                       a wrong degradation config is obvious here in a second)
  dataset_stats.png    image count, resolution scatter, brightness histogram
  probe_ep{N}.png      bicubic | SR | GT on fixed validation images, every
                       save_interval epochs, with PSNR under the repo metric

The Trainer wraps every call in try/except: a rendering failure must never
kill a training run. Functions here therefore raise freely.

matplotlib is imported lazily with the Agg backend so importing this module
costs nothing and never needs a display.
"""
import csv
import math
import os
import random

import cv2
import numpy as np
import torch


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------- results.png

_PANELS = [
    ("train_loss", "Train loss", "log"),
    ("val_loss", "Val loss", "log"),
    ("psnr", "PSNR (dB)", "linear"),
    ("ssim", "SSIM", "linear"),
    ("lpips", "LPIPS (lower better)", "linear"),
    ("lr", "Learning rate", "log"),
]


def render_results(csv_path, out_path, title=None):
    """training_log.csv -> six-panel curve figure. Cheap (~0.3s), safe to call
    every epoch; written atomically so a viewer never sees a half-drawn file."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]

    def col(name):
        return [float(r[name]) if r.get(name) else math.nan for r in rows]

    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=110)
    for ax, (name, label, scale) in zip(axes.flat, _PANELS):
        vals = col(name)
        ax.plot(epochs, vals, lw=1.6, color="tab:blue")
        finite = [(e, v) for e, v in zip(epochs, vals) if not math.isnan(v)]
        if name == "psnr" and finite:
            be, bv = max(finite, key=lambda t: t[1])
            ax.plot(be, bv, "o", color="tab:red", ms=5)
            ax.annotate(f"best {bv:.2f} @ {be}", xy=(be, bv), fontsize=8,
                        color="tab:red", xytext=(4, -12), textcoords="offset points")
        if scale == "log" and finite and min(v for _, v in finite) > 0:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("epoch", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(title or os.path.basename(os.path.dirname(os.path.abspath(csv_path))), fontsize=12)
    fig.tight_layout()
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, out_path)


# ------------------------------------------------------------ train_batch.png

def _to_u8(chw):
    """float CHW [0,1] -> uint8 HWC BGR for cv2."""
    img = (chw.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _label(img, text):
    cv2.rectangle(img, (0, 0), (8 + 9 * len(text), 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_train_batch(batch, out_path, max_items=8):
    """Grid of LR|HR pairs exactly as the model will see them, degradations
    applied. LR is nearest-upscaled to HR height so artifacts stay visible."""
    lrs, hrs = batch["lr"], batch["hr"]
    n = min(max_items, lrs.shape[0])
    tiles = []
    for i in range(n):
        hr = _to_u8(hrs[i])
        lr = _to_u8(lrs[i])
        lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST)
        tiles.append(np.hstack([_label(lr, "LR (degraded)"), _label(hr, "HR")]))
    cols = 2
    rows = [np.hstack(tiles[i:i + cols]) if len(tiles[i:i + cols]) == cols
            else np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(out_path, np.vstack(rows))


# ---------------------------------------------------------- dataset_stats.png

def render_dataset_stats(image_paths, out_path, sample=2000, decode_sample=128):
    """Counts, resolution scatter and brightness histogram. Header-only reads
    for sizes (cheap); decoding is capped at `decode_sample` thumbnails."""
    from PIL import Image

    rng = random.Random(0)
    paths = list(image_paths)
    size_paths = paths if len(paths) <= sample else rng.sample(paths, sample)
    sizes = []
    for p in size_paths:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)  # header only, no decode
        except Exception:
            pass

    bright = []
    for p in (paths if len(paths) <= decode_sample else rng.sample(paths, decode_sample)):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            bright.append(cv2.resize(img, (64, 64)).mean())

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=110)
    axes[0].axis("off")
    axes[0].text(0.05, 0.6, f"images: {len(paths):,}", fontsize=16)
    axes[0].text(0.05, 0.35, f"resolutions sampled: {len(sizes):,}\n"
                             f"brightness sampled: {len(bright):,}", fontsize=10, color="gray")
    if sizes:
        w, h = zip(*sizes)
        axes[1].scatter(w, h, s=4, alpha=0.25)
        axes[1].set_title("resolution (W x H)", fontsize=11)
        axes[1].set_xlabel("width"); axes[1].set_ylabel("height")
        axes[1].grid(alpha=0.3)
    if bright:
        axes[2].hist(bright, bins=32, color="tab:blue", alpha=0.8)
        axes[2].set_title("mean brightness (0-255)", fontsize=11)
        axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- probe_ep{N}

def render_probe(model, samples, scale, device, out_path):
    """bicubic | SR | GT side by side for fixed validation samples, PSNR under
    the repo metric (src.metrics). `samples` is a list of PairedDataset items."""
    from src.metrics import create_psnr

    metric = create_psnr(scale, device)
    model_was_training = model.training
    model.eval()
    tiles = []
    with torch.no_grad():
        for s in samples:
            lr = s["lr"].unsqueeze(0).to(device)
            hr = s["hr"].unsqueeze(0).to(device)
            sr = model(lr)
            if isinstance(sr, tuple):
                sr = sr[0]
            sr = sr.clamp(0, 1)
            if sr.shape[-2:] != hr.shape[-2:]:
                h = min(sr.shape[-2], hr.shape[-2]); w = min(sr.shape[-1], hr.shape[-1])
                sr, hr = sr[..., :h, :w], hr[..., :h, :w]
            bic = torch.nn.functional.interpolate(
                lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
            p_bic = metric(bic, hr).item()
            p_sr = metric(sr, hr).item()
            tiles.append(np.hstack([
                _label(_to_u8(bic[0]), f"bicubic {p_bic:.2f} dB"),
                _label(_to_u8(sr[0]), f"SR {p_sr:.2f} dB"),
                _label(_to_u8(hr[0]), "GT"),
            ]))
    if model_was_training:
        model.train()
    if tiles:
        cv2.imwrite(out_path, np.vstack(tiles))


# ------------------------------------------------------------- RNG insulation

class preserved_rng:
    """Restore python/numpy/torch RNG state on exit, so rendering a sample
    batch at startup does not shift the training's random stream -- two runs
    with the same seed stay bit-identical whether or not plots are enabled."""

    def __enter__(self):
        self._py = random.getstate()
        self._np = np.random.get_state()
        self._torch = torch.get_rng_state()
        self._cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return self

    def __exit__(self, *exc):
        random.setstate(self._py)
        np.random.set_state(self._np)
        torch.set_rng_state(self._torch)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)
        return False
