"""Why the scheduler is NOT passed through accelerator.prepare.

The trainer steps its scheduler once per epoch. AcceleratedScheduler steps the
inner scheduler num_processes times per call, and freezes any scheduler that
merely HAS a total_steps attribute (accelerate's OneCycle guard) once
_step_count exceeds it -- which KneeLRScheduler trips by name alone.

These tests pin the two facts the unwrapping rests on:
  * for this repo's launch mode (single process, per-epoch stepping) the
    wrapped and bare schedulers are bit-identical, so removing the wrapper
    changed nothing for any run, including a mid-run resume;
  * under a simulated multi-process state the wrapped scheduler compresses
    the knee by num_processes -- the distortion the removal fixes.

Each Accelerator carries process-global state, so the accelerate singletons
are reset between runs.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import accelerate.scheduler as asched
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState

from src.engine.schedulers import KneeLRScheduler

TOTAL_STEPS = 10


def _reset():
    AcceleratorState._reset_state()
    GradientState._reset_state()
    PartialState._reset_state()


def _run(prepare_sched, ga=1, epochs=10, fake_np=None):
    _reset()
    torch.manual_seed(0)
    acc = Accelerator(gradient_accumulation_steps=ga, cpu=True)
    model = nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = KneeLRScheduler(opt, peak_lr=5e-4, warmup_steps=0, explore_steps=0,
                            total_steps=TOTAL_STEPS, min_lr=5e-6)
    loader = DataLoader(TensorDataset(torch.randn(10, 4), torch.randn(10, 4)), batch_size=2)
    if prepare_sched:
        model, opt, sched, loader = acc.prepare(model, opt, sched, loader)
    else:
        model, opt, loader = acc.prepare(model, opt, loader)

    orig_state = asched.AcceleratorState
    if fake_np is not None:
        class _FakeState:
            num_processes = fake_np
        asched.AcceleratorState = _FakeState
    lrs = []
    try:
        for _ in range(epochs):
            for x, y in loader:
                opt.zero_grad()
                with acc.accumulate(model):
                    loss = ((model(x) - y) ** 2).mean()
                    acc.backward(loss)
                    opt.step()
                    opt.zero_grad()
            sched.step()  # once per epoch, exactly like Trainer.fit
            lrs.append(opt.param_groups[0]["lr"])
    finally:
        asched.AcceleratorState = orig_state
    return lrs, sched.state_dict()


def test_single_process_unwrapping_is_bit_identical():
    """np=1, ga=1: the launch mode of every run this repo starts."""
    wrapped, wrapped_sd = _run(True)
    bare, bare_sd = _run(False)
    assert wrapped == bare
    assert wrapped_sd == bare_sd


def test_gradient_accumulation_does_not_distort_per_epoch_stepping():
    """ga=2 with an odd number of accumulate groups per epoch: end-of-dataloader
    forces sync_gradients=True, so the per-epoch step always lands."""
    wrapped, _ = _run(True, ga=2)
    bare, _ = _run(False, ga=2)
    assert wrapped == bare


def test_multi_process_wrapping_is_the_distorted_one():
    """Simulated np=2: the wrapper steps the knee twice per epoch, reaching
    min_lr at the halfway point. The bare scheduler follows the analytic knee."""
    wrapped, _ = _run(True, fake_np=2)
    bare, _ = _run(False, fake_np=2)  # fake state is irrelevant when unwrapped
    assert wrapped != bare
    assert wrapped[TOTAL_STEPS // 2 - 1] == pytest.approx(5e-6)  # min_lr at epoch 5
    expected = [max(5e-6, 5e-4 * (1 - e / TOTAL_STEPS)) for e in range(1, TOTAL_STEPS + 1)]
    assert bare == pytest.approx(expected)
