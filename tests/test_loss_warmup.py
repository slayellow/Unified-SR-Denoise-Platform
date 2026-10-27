"""Why the auxiliary losses ramp in instead of switching on at full weight.

A Phase-2 finetune starts from weights that sit at the optimum of Phase-1's
loss. Adding perceptual/SSIM/edge terms in one step moves the objective out
from under the model, and the run's first validation lands WORSE than the
checkpoint it started from -- which happened in five of six v4 Phase-2 runs,
and cost three of them the run: never having beaten their baseline, their
patience counted from epoch 1 and expired while they were still descending.

These tests pin the properties that fix rests on:
  * epoch 1 is fidelity-only, so the first validation reproduces the baseline;
  * the weight is a pure function of the epoch, so resuming mid-warm-up gives
    the same schedule as an uninterrupted run;
  * a config that does not ask for warm-up is untouched.
"""
import pytest
import torch

from src.losses.losses import UnifiedLoss

WARM = {
    "mse": {"enabled": True, "weight": 1.0},
    "ssim": {"enabled": True, "weight": 0.1, "warmup_epochs": 10},
    "edge": {"enabled": True, "weight": 0.05, "warmup_epochs": 10, "mode": "mean"},
}
PLAIN = {
    "mse": {"enabled": True, "weight": 1.0},
    "ssim": {"enabled": True, "weight": 0.1},
    "edge": {"enabled": True, "weight": 0.05, "mode": "mean"},
}


def _pair(seed=0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(2, 3, 32, 32, generator=g)
    target = torch.rand(2, 3, 32, 32, generator=g)
    return pred, target


def test_epoch_one_is_fidelity_only():
    """The whole point: epoch 1 must reproduce the starting checkpoint's loss,
    so the finetune cannot validate worse than its own baseline on the way in."""
    warm, plain = UnifiedLoss(WARM), UnifiedLoss(PLAIN)
    warm.set_epoch(1)
    assert warm.weights["ssim"] == 0.0
    assert warm.weights["edge"] == 0.0
    assert warm.weights["mse"] == 1.0  # fidelity term is never ramped

    pred, target = _pair()
    total, parts = warm(pred, target)
    mse_only = torch.nn.functional.mse_loss(pred, target)
    assert total == pytest.approx(mse_only.item(), rel=1e-6)
    # ...and the disabled terms still occupy their CSV columns
    assert set(parts) == set(plain.weights)
    assert parts["ssim"] == 0.0 and parts["edge"] == 0.0


@pytest.mark.parametrize("epoch,frac", [(1, 0.0), (2, 0.1), (6, 0.5), (11, 1.0), (50, 1.0)])
def test_ramp_reaches_target_at_warmup_plus_one(epoch, frac):
    loss = UnifiedLoss(WARM)
    loss.set_epoch(epoch)
    assert loss.weights["ssim"] == pytest.approx(0.1 * frac)
    assert loss.weights["edge"] == pytest.approx(0.05 * frac)


def test_schedule_is_a_pure_function_of_the_epoch():
    """Resume safety: a run restarted at epoch 5 must weight its losses exactly
    as an uninterrupted run does at epoch 5, whatever it did before."""
    a, b = UnifiedLoss(WARM), UnifiedLoss(WARM)
    for e in range(1, 51):
        a.set_epoch(e)
    a.set_epoch(5)
    b.set_epoch(5)
    assert a.weights == b.weights


def test_config_without_warmup_is_unchanged():
    """Every existing config: set_epoch must be a no-op for them."""
    loss = UnifiedLoss(PLAIN)
    before = dict(loss.weights)
    for e in (1, 2, 30):
        loss.set_epoch(e)
        assert loss.weights == before
        assert loss.active_weights() == {}


def test_active_weights_reports_only_what_is_still_ramping():
    loss = UnifiedLoss(WARM)
    loss.set_epoch(6)
    assert set(loss.active_weights()) == {"ssim", "edge"}
    loss.set_epoch(11)
    assert loss.active_weights() == {}  # nothing to report once at target


def test_warmed_run_ends_at_the_same_loss_as_an_unwarmed_one():
    """Warm-up only changes the path, not the destination."""
    warm, plain = UnifiedLoss(WARM), UnifiedLoss(PLAIN)
    warm.set_epoch(11)
    pred, target = _pair(3)
    assert warm(pred, target)[0] == pytest.approx(plain(pred, target)[0], rel=1e-6)
