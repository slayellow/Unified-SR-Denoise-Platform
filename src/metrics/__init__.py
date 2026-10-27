"""One definition of PSNR and SSIM for the whole repository.

The same Bicubic++ x2 checkpoint scored 25.85 dB inside the trainer, 26.50 dB
under tools/evaluate.py and 27.83 dB under the convention SR papers report --
a 1.97 dB spread with no change to the model. That is wider than most
improvements worth chasing, so any number carried between two scripts was
meaningless.

Canonical spec, used everywhere:

  luminance only, BT.601 limited range (``color_space='ycbcr'``), which is what
  BasicSR and the SR literature report;
  ``crop_border=scale``, the usual SR shave;
  inputs are ``(N, 3, H, W)`` float tensors in [0, 1].

Measured contributions on the x2 validation set: the colour space is worth
+1.97 dB and the border shave +0.003 dB. The shave is here so published tables
line up, not because it moves the number.

Switching to this spec makes every metric jump by roughly +2 dB relative to
logs written before it. That is a change of ruler, not of model.
"""

import pyiqa

__all__ = ["METRIC_SPEC", "create_psnr", "create_ssim", "create_lpips", "create_niqe"]

# Recorded into checkpoints so a log always says which ruler produced it.
METRIC_SPEC = "y-ycbcr/crop=scale"

_Y_KWARGS = {"test_y_channel": True, "color_space": "ycbcr"}


def create_psnr(scale, device):
    """Full-reference PSNR on the luminance channel, shaved by `scale`."""
    return pyiqa.create_metric("psnr", device=device, crop_border=scale, **_Y_KWARGS)


def create_ssim(scale, device):
    """Full-reference SSIM on the luminance channel, shaved by `scale`."""
    return pyiqa.create_metric("ssim", device=device, crop_border=scale, **_Y_KWARGS)


def create_lpips(device):
    """Perceptual distance. Operates on RGB; no luminance or shave options."""
    return pyiqa.create_metric("lpips", device=device)


def create_niqe(scale, device):
    """No-reference quality of the reconstruction alone."""
    return pyiqa.create_metric("niqe", device=device, crop_border=scale)
