"""Losses that batch F touched: SSIMLoss (pyiqa-backed) and HotPixelMaskedLoss.

SSIMLoss must agree with the pyiqa SSIM the Validator reports (same
implementation, RGB view) and stay differentiable. HotPixelMaskedLoss must
work at SR scales -- its mask is detected on the LR input, and before the fix
the elementwise product with an SR-sized loss map was a shape error for any
scale above 1.
"""
import pytest
import torch

from src.losses.losses import HotPixelMaskedLoss, SSIMLoss


def test_ssim_loss_matches_pyiqa_and_backprops():
    torch.manual_seed(0)
    a = torch.rand(2, 3, 32, 32, requires_grad=True)
    b = torch.rand(2, 3, 32, 32)

    loss = SSIMLoss()(a, b)
    from pyiqa.archs.ssim_arch import ssim
    expected = 1.0 - ssim(a.detach(), b, data_range=1.0).mean()
    assert torch.allclose(loss, expected, atol=1e-6)

    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()

    # identical images -> zero loss
    same = torch.rand(1, 3, 24, 24)
    assert SSIMLoss()(same, same).abs() < 1e-6


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_hot_pixel_masked_loss_handles_sr_scales(scale):
    torch.manual_seed(0)
    h = w = 16
    lr = torch.zeros(1, 3, h, w)
    lr[0, :, 4, 5] = 1.0  # one hot pixel on an otherwise black input
    pred = torch.rand(1, 3, h * scale, w * scale)
    target = torch.rand(1, 3, h * scale, w * scale)

    loss = HotPixelMaskedLoss()(pred, target, lr)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss >= 0

    # no hot pixels -> zero; missing input -> zero
    assert HotPixelMaskedLoss()(pred, target, torch.zeros_like(lr)) == 0
    assert HotPixelMaskedLoss()(pred, target, None) == 0


def test_gray_perceptual_has_zero_chroma_gradient():
    """IR is achromatic; the gray mode must be unable to push chroma. The
    channel-mean collapse hands every input channel the identical gradient,
    so the difference must be exactly zero, not merely small."""
    torch.manual_seed(0)
    pred = torch.rand(1, 3, 32, 32, requires_grad=True)
    target = torch.rand(1, 3, 32, 32)
    from src.losses.losses import PerceptualLoss
    loss = PerceptualLoss(layers={8: 1.0}, gray=True)(pred, target)
    loss.backward()
    g = pred.grad
    assert torch.equal(g[:, 0], g[:, 1]) and torch.equal(g[:, 0], g[:, 2])


def test_edge_mean_mode_has_zero_chroma_gradient():
    """mode='gray' (BT.601) measured 49% of its gradient in the chroma
    direction at an achromatic input; mode='mean' must measure exactly zero
    while producing the same loss value on R=G=B frames."""
    from src.losses.losses import EdgeLoss
    torch.manual_seed(0)
    g = torch.rand(1, 1, 32, 32)
    pred = g.repeat(1, 3, 1, 1).clone().requires_grad_(True)
    target = torch.rand(1, 1, 32, 32).repeat(1, 3, 1, 1)

    EdgeLoss(mode='mean')(pred, target).backward()
    gr = pred.grad
    assert torch.equal(gr[:, 0], gr[:, 1]) and torch.equal(gr[:, 0], gr[:, 2])

    same = EdgeLoss(mode='gray')(pred.detach(), target).item()
    mean = EdgeLoss(mode='mean')(pred.detach(), target).item()
    assert abs(same - mean) < 1e-6
