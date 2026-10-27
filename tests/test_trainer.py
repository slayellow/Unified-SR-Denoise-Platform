"""Trainer behaviour that the shipped recipes depend on."""

import pytest
import torch
import torch.nn as nn

import src.engine.trainer as t


def _trainer(tmp_path, monkeypatch, accum):
    monkeypatch.setattr(t, "create_lpips", lambda *a, **k: None)
    monkeypatch.setattr(t, "create_niqe", lambda *a, **k: None)
    cfg = {
        "task": "sr",
        "model": {"name": "conv", "scale": 1},
        "train": {"lr": 1e-3, "epochs": 1, "gradient_accumulation_steps": accum, "scheduler": {"type": "StepLR"},
                  "ema": {"enabled": True, "decay": 0.5, "start_epoch": 0}},
        "loss": {"l1": {"enabled": True, "weight": 1.0}},
    }
    torch.manual_seed(0)
    trainer = t.Trainer(nn.Conv2d(3, 3, 3, padding=1), cfg, "cpu", checkpoint_dir=str(tmp_path))
    data = [{"lr": torch.rand(3, 8, 8), "hr": torch.rand(3, 8, 8)} for _ in range(8)]
    loader = torch.utils.data.DataLoader(data, batch_size=1)
    trainer.model, trainer.optimizer, loader = trainer.accelerator.prepare(trainer.model, trainer.optimizer, loader)
    return trainer, loader


def _count_calls(monkeypatch, obj, name):
    calls = []
    orig = getattr(obj, name)
    monkeypatch.setattr(obj, name, lambda *a, **k: calls.append(1) or orig(*a, **k))
    return calls


@pytest.mark.parametrize("accum,steps_per_epoch", [(1, 8), (2, 4)])
def test_ema_updates_once_per_optimizer_step(accum, steps_per_epoch, tmp_path, monkeypatch):
    """With gradient accumulation the optimizer steps once per accumulation window; an EMA update
    on every micro-batch folds the unchanged weights in again and halves the averaging horizon."""
    trainer, loader = _trainer(tmp_path, monkeypatch, accum=accum)
    steps = _count_calls(monkeypatch, trainer.optimizer.optimizer, "step")
    emas = _count_calls(monkeypatch, trainer, "_ema_update")
    trainer.train_epoch(loader, epoch=1)
    assert len(steps) == steps_per_epoch and len(emas) == len(steps), (len(steps), len(emas))


def _fold(trainer, n):
    for _ in range(n):
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(0.1)
        trainer._ema_update()


def test_checkpoint_ships_ema_only_from_its_start_epoch(tmp_path, monkeypatch):
    trainer, _ = _trainer(tmp_path, monkeypatch, accum=1)
    trainer.ema_start_epoch = 2
    _fold(trainer, 3)
    trainer.save_checkpoint(1, filename="ep1.pth")
    trainer.save_checkpoint(2, filename="ep2.pth")
    early = torch.load(tmp_path / "ep1.pth", weights_only=False)
    late = torch.load(tmp_path / "ep2.pth", weights_only=False)
    live = trainer.accelerator.unwrap_model(trainer.model).state_dict()
    assert early["ema_used"] is False and early["live_state_dict"] is None
    assert all(torch.equal(early["model_state_dict"][k], v) for k, v in live.items())
    assert late["ema_used"] is True
    assert all(torch.equal(late["live_state_dict"][k], v) for k, v in live.items())
    assert all(torch.equal(late["model_state_dict"][k], trainer.ema_state[k]) for k in live)


def test_resume_restores_live_weights_ema_and_early_stopping_state(tmp_path, monkeypatch):
    trainer, _ = _trainer(tmp_path, monkeypatch, accum=1)
    trainer.ema_start_epoch = 0
    _fold(trainer, 3)
    trainer._best_attempt, trainer.early_stop_counter = 0.25, 3
    trainer.save_checkpoint(4, filename="last.pth")
    live = {k: v.clone() for k, v in trainer.accelerator.unwrap_model(trainer.model).state_dict().items()}
    ema = {k: v.clone() for k, v in trainer.ema_state.items()}

    resumed, _ = _trainer(tmp_path / "again", monkeypatch, accum=1)
    resumed.resume(str(tmp_path / "last.pth"))
    weights = resumed.accelerator.unwrap_model(resumed.model).state_dict()
    assert resumed.start_epoch == 5
    assert all(torch.equal(weights[k], v) for k, v in live.items())
    assert all(torch.equal(resumed.ema_state[k], v) for k, v in ema.items())
    assert resumed._best_attempt == 0.25 and resumed.early_stop_counter == 3


def test_resume_from_missing_checkpoint_raises(tmp_path, monkeypatch):
    trainer, _ = _trainer(tmp_path, monkeypatch, accum=1)
    with pytest.raises(FileNotFoundError):
        trainer.resume(str(tmp_path / "nope" / "last.pth"))


def test_best_checkpoints_carry_the_state_of_their_own_epoch(tmp_path, monkeypatch):
    """best.pth used to be written before the early-stop counter reset and best_attempt.pth before the
    best metric update, so resuming from either restored the previous epoch's state."""
    monkeypatch.setattr(t, "create_lpips", lambda *a, **k: None)
    monkeypatch.setattr(t, "create_niqe", lambda *a, **k: None)
    cfg = {
        "task": "sr",
        "model": {"name": "conv", "scale": 1},
        "train": {"lr": 1e-3, "epochs": 4, "scheduler": {"type": "StepLR"}, "save_interval": 100, "plots": False,
                  "early_stopping": {"patience": 5}},
        "loss": {"l1": {"enabled": True, "weight": 1.0}},
    }
    trainer = t.Trainer(nn.Conv2d(3, 3, 3, padding=1), cfg, "cpu", checkpoint_dir=str(tmp_path))
    losses = iter([1.0, 1.1, 1.2, 0.5])
    monkeypatch.setattr(trainer, "train_epoch", lambda loader, epoch: 0.0)
    monkeypatch.setattr(trainer.validator, "run", lambda loader, epoch: {"loss": next(losses)})
    data = [{"lr": torch.rand(3, 8, 8), "hr": torch.rand(3, 8, 8)}]
    trainer.fit(data, data, epochs=4)
    best = torch.load(tmp_path / "best.pth", weights_only=False)
    attempt = torch.load(tmp_path / "best_attempt.pth", weights_only=False)
    assert best["epoch"] == 4 and best["early_stop_counter"] == 0 and best["best_metric_val"] == 0.5
    assert attempt["epoch"] == 4 and attempt["best_metric_val"] == 0.5 and attempt["best_attempt"] == 0.5


def test_probe_is_drawn_with_the_weights_that_are_judged(tmp_path, monkeypatch):
    """Validation and checkpoints read the EMA weights; the probe image was drawn with the live ones."""
    trainer, loader = _trainer(tmp_path, monkeypatch, accum=1)
    trainer.cfg["train"].update(save_interval=1, plots=True)
    drawn = []
    monkeypatch.setattr(t.visualize, "render_probe",
                        lambda model, *a, **k: drawn.append(model.weight.detach().clone()))
    monkeypatch.setattr(trainer, "train_epoch", lambda loader, epoch: _fold(trainer, 3) or 0.0)
    monkeypatch.setattr(trainer.validator, "run", lambda loader, epoch: {"loss": 1.0})
    trainer.fit(loader, loader, epochs=1)
    live = trainer.accelerator.unwrap_model(trainer.model).weight
    assert len(drawn) == 1 and torch.equal(drawn[0], trainer.ema_state["weight"].to(drawn[0].dtype))
    assert not torch.equal(drawn[0], live.detach())


def test_monitor_loss_with_a_loss_warmup_is_rejected(tmp_path, monkeypatch):
    """A warm-up raises the validation loss epoch by epoch, so 'loss' would keep the first epochs as best."""
    monkeypatch.setattr(t, "create_lpips", lambda *a, **k: None)
    monkeypatch.setattr(t, "create_niqe", lambda *a, **k: None)
    cfg = {"task": "sr", "model": {"name": "conv", "scale": 1},
           "train": {"lr": 1e-3, "epochs": 1, "scheduler": {"type": "StepLR"}},
           "loss": {"l1": {"enabled": True, "weight": 1.0, "warmup_epochs": 2}}}
    with pytest.raises(ValueError, match="warm"):
        t.Trainer(nn.Conv2d(3, 3, 3, padding=1), cfg, "cpu", checkpoint_dir=str(tmp_path))


def test_git_sha_marks_uncommitted_changes(tmp_path):
    """A run trained on uncommitted code used to record the clean HEAD sha."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True,
                       capture_output=True)

    git("init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("commit", "-q", "-m", "init")
    clean = t.git_sha(str(tmp_path))
    assert clean != "unknown" and not clean.endswith("-dirty")
    (tmp_path / "a.py").write_text("x = 2\n")
    assert t.git_sha(str(tmp_path)) == clean + "-dirty"
