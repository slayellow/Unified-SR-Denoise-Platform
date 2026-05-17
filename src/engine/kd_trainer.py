import csv
import os

import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.losses.kd_losses import OnlineMultiTeacherKDLoss


class OnlineKDTrainer:
    def __init__(self, student_model, teacher_models, config, writer=None, checkpoint_dir=None):
        self.cfg = config
        grad_accum_steps = self.cfg["train"].get("gradient_accumulation_steps", 1)
        self.accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)
        self.device = self.accelerator.device

        self.model = student_model
        self.teacher_models = teacher_models
        self._freeze_teachers()

        self.criterion = OnlineMultiTeacherKDLoss(
            supervised_config=self.cfg.get("loss", {}),
            kd_config=self.cfg.get("kd", {}),
        ).to(self.device)

        opt_cfg = self.cfg["train"].get("optimizer", {})
        opt_type = opt_cfg.get("type", "AdamW")
        lr = float(self.cfg["train"]["lr"])
        weight_decay = float(self.cfg["train"].get("weight_decay", 0.0))

        if opt_type == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        elif opt_type == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=opt_cfg.get("betas", (0.9, 0.999)),
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=opt_cfg.get("betas", (0.9, 0.999)),
                weight_decay=weight_decay,
            )

        sched_cfg = self.cfg["train"].get("scheduler", {})
        sched_type = sched_cfg.get("type", "CosineAnnealingLR")
        self.scheduler_type = "step"
        if sched_type == "ReduceLROnPlateau":
            self.scheduler_type = "plateau"
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_cfg.get("mode", "min"),
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 10),
            )
        elif sched_type == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(sched_cfg.get("T_max", self.cfg["train"]["epochs"])),
                eta_min=float(sched_cfg.get("eta_min", 0.0)),
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg["train"].get("step_size", 50),
                gamma=self.cfg["train"].get("gamma", 0.5),
            )

        self.checkpoint_dir = checkpoint_dir or os.path.join(
            "checkpoints",
            f"{self.cfg['model']['name']}_{self.cfg['task']}_kd_x{self.cfg['model']['scale']}",
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = writer
        if self.writer is None and self.accelerator.is_main_process:
            self.writer = SummaryWriter(os.path.join(self.checkpoint_dir, "logs"))

        early_stop_cfg = self.cfg["train"].get("early_stopping", {})
        self.patience = early_stop_cfg.get("patience", 20)
        self.min_delta = early_stop_cfg.get("min_delta", 0.0)
        self.monitor_metric = early_stop_cfg.get("monitor", "val_loss")
        self.monitor_mode = early_stop_cfg.get("mode", "min")
        self.best_metric_val = float("inf") if self.monitor_mode == "min" else -float("inf")
        self.early_stop_counter = 0
        self.start_epoch = 1

    def _freeze_teachers(self):
        for teacher in self.teacher_models.values():
            teacher.to(self.device)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

    def prepare_accelerator(self, train_loader, val_loader=None):
        if val_loader is not None:
            self.model, self.optimizer, self.scheduler, train_loader, val_loader = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
                train_loader,
                val_loader,
            )
        else:
            self.model, self.optimizer, self.scheduler, train_loader = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
                train_loader,
            )
        return train_loader, val_loader

    @staticmethod
    def _extract_model_output(output):
        if isinstance(output, tuple):
            return output[0]
        return output

    def _teacher_forward(self, noisy):
        outputs = {}
        with torch.no_grad():
            for name, teacher in self.teacher_models.items():
                outputs[name] = self._extract_model_output(teacher(noisy)).detach()
        return outputs

    def train_epoch(self, loader, epoch, log_interval=10):
        self.model.train()
        for teacher in self.teacher_models.values():
            teacher.eval()

        running_loss = 0.0
        epoch_loss = 0.0
        num_steps = 0
        pbar = tqdm(loader, desc=f"KD Epoch {epoch}", disable=not self.accelerator.is_local_main_process)

        for step, batch in enumerate(pbar):
            noisy = batch["lr"].to(self.device)
            clean = batch["hr"].to(self.device)

            with self.accelerator.accumulate(self.model):
                teacher_outputs = self._teacher_forward(noisy)
                student_out = self._extract_model_output(self.model(noisy))
                loss, loss_parts = self.criterion(student_out, clean, noisy, teacher_outputs)

                self.accelerator.backward(loss)

                grad_clip = self.cfg["train"].get("grad_clip", 0.0)
                if grad_clip > 0 and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad()

            loss_value = loss.detach().float().item()
            running_loss += loss_value
            epoch_loss += loss_value
            num_steps += 1

            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.writer and self.accelerator.is_main_process:
                    global_step = epoch * len(loader) + step
                    self.writer.add_scalar("Loss/Total", avg_loss, global_step)
                    self.writer.add_scalar("Train/LR", current_lr, global_step)
                    for name, value in loss_parts.items():
                        self.writer.add_scalar(f"Loss/{name}", value.detach().float().item(), global_step)
                pbar.set_postfix({"loss": f"{avg_loss:.6f}", "lr": f"{current_lr:.2e}"})
                running_loss = 0.0

        local = torch.tensor([epoch_loss, num_steps], device=self.device, dtype=torch.float32)
        reduced = self.accelerator.reduce(local, reduction="sum")
        return (reduced[0] / reduced[1].clamp_min(1.0)).item()

    def validate_epoch(self, loader, epoch):
        if loader is None:
            return {}

        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        total_psnr = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        total_count = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"KD Validation {epoch}", disable=not self.accelerator.is_local_main_process):
                noisy = batch["lr"].to(self.device)
                clean = batch["hr"].to(self.device)
                student_out = self._extract_model_output(self.model(noisy))
                supervised_loss, _ = self.criterion.supervised_criterion(student_out, clean, noisy)

                batch_size = noisy.shape[0]
                total_loss += supervised_loss.detach().float() * batch_size

                pred = torch.clamp(student_out, 0.0, 1.0)
                target = torch.clamp(clean, 0.0, 1.0)
                mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
                psnr = -10.0 * torch.log10(mse)
                total_psnr += psnr.sum().float()
                total_count += batch_size

        stats = torch.stack([total_loss, total_psnr, total_count])
        stats = self.accelerator.reduce(stats, reduction="sum")
        count = stats[2].clamp_min(1.0)
        metrics = {
            "val_loss": (stats[0] / count).item(),
            "psnr": (stats[1] / count).item(),
        }

        if self.writer and self.accelerator.is_main_process:
            self.writer.add_scalar("Loss/Val", metrics["val_loss"], epoch)
            self.writer.add_scalar("Metrics/PSNR", metrics["psnr"], epoch)

        self.accelerator.print(
            f"[KD Validation][Epoch {epoch}] Loss: {metrics['val_loss']:.6f} | PSNR: {metrics['psnr']:.4f}"
        )
        return metrics

    def resume(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_metric_val = checkpoint.get("best_metric_val", self.best_metric_val)
        self.accelerator.print(f"[KDTrainer] Resumed from epoch {self.start_epoch}: {checkpoint_path}")

    def save_checkpoint(self, epoch, filename="last.pth", is_best=False):
        if not self.accelerator.is_main_process:
            return

        state = {
            "epoch": epoch,
            "model_state_dict": self.accelerator.get_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric_val": self.best_metric_val,
            "config": self.cfg,
        }

        save_path = os.path.join(self.checkpoint_dir, "best.pth" if is_best else filename)
        self.accelerator.save(state, save_path)
        self.accelerator.print(f"Saved KD checkpoint: {save_path}")

    def fit(self, train_loader, val_loader=None, epochs=100, resume_path=None):
        train_loader, val_loader = self.prepare_accelerator(train_loader, val_loader)
        if resume_path:
            self.resume(resume_path)

        csv_path = os.path.join(self.checkpoint_dir, "training_log.csv")
        file_exists = os.path.isfile(csv_path)

        if self.accelerator.is_main_process:
            csv_file = open(csv_path, mode="a", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_loss", "psnr", "lr"])
            if not file_exists:
                writer.writeheader()
        else:
            csv_file = None
            writer = None

        try:
            self.accelerator.print(f"Start online KD training from epoch {self.start_epoch} to {epochs}...")
            for epoch in range(self.start_epoch, epochs + 1):
                train_loss = self.train_epoch(train_loader, epoch)
                val_metrics = self.validate_epoch(val_loader, epoch) if val_loader else {}
                val_loss = val_metrics.get("val_loss")

                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.scheduler_type == "plateau":
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()

                monitor_value = val_metrics.get(self.monitor_metric, val_loss if val_loss is not None else train_loss)
                is_best = False
                if self.monitor_mode == "min":
                    is_best = monitor_value < (self.best_metric_val - self.min_delta)
                else:
                    is_best = monitor_value > (self.best_metric_val + self.min_delta)

                if is_best:
                    self.best_metric_val = monitor_value
                    self.early_stop_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.early_stop_counter += 1

                if self.accelerator.is_main_process and writer is not None:
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss if val_loss is not None else "",
                            "psnr": val_metrics.get("psnr", ""),
                            "lr": current_lr,
                        }
                    )
                    csv_file.flush()

                if epoch % self.cfg["train"]["save_interval"] == 0:
                    self.save_checkpoint(epoch, filename=f"epoch_{epoch}.pth")
                self.save_checkpoint(epoch, filename="last.pth")

                if self.early_stop_counter >= self.patience:
                    self.accelerator.print("[KDTrainer] Early stopping triggered.")
                    break
        finally:
            if csv_file is not None:
                csv_file.close()
            if self.writer is not None and self.accelerator.is_main_process:
                self.writer.close()
