import os
import csv
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pyiqa
from accelerate import Accelerator
from src.losses.losses import UnifiedLoss

class BaseEngine:
    def __init__(self, device):
        self.device = device

    def to_device(self, data):
        if isinstance(data, dict):
            return {k: v.to(self.device) for k, v in data.items() if isinstance(v, torch.Tensor)}
        elif isinstance(data, (list, tuple)):
            return [v.to(self.device) for v in data]
        return data.to(self.device)

class Validator(BaseEngine):
    def __init__(self, model, criterion, device, writer=None, task='sr', is_guided_model=True, accelerator=None):
        super().__init__(device)
        self.model = model
        self.criterion = criterion
        self.writer = writer
        self.task = task
        self.is_guided_model = is_guided_model
        self.accelerator = accelerator
        
        # Initialize Metrics
        self._print("[Validator] Initializing metrics (PSNR, SSIM, LPIPS, NIQE)...")
        self.metric_psnr = pyiqa.create_metric('psnr', device=device)
        self.metric_ssim = pyiqa.create_metric('ssim', device=device)
        self.metric_lpips = pyiqa.create_metric('lpips', device=device)
        self.metric_niqe = pyiqa.create_metric('niqe', device=device)

    def _print(self, message):
        if self.accelerator is not None:
            self.accelerator.print(message)
        else:
            print(message)

    def run(self, loader, epoch):
        self.model.eval()
        val_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # Metric accumulators
        acc_psnr = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        acc_ssim = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        acc_lpips = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        acc_niqe = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        n_samples = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc=f"Validation Epoch {epoch}", disable=not self.accelerator.is_local_main_process)):
                lr = batch['lr'].to(self.device)
                hr = batch['hr'].to(self.device)
                batch_size = lr.shape[0]
                
                if self.task == 'guide' and 'guide' in batch and self.is_guided_model:
                    guide = batch['guide'].to(self.device)
                    sr = self.model(lr, guide)
                else:
                    sr = self.model(lr)
                if isinstance(sr, tuple): sr = sr[0]
                
                # Loss
                loss, _ = self.criterion(sr, hr, lr) 
                val_loss += loss.detach().float() * batch_size
                
                # Metrics (Clamp 0-1 for safety)
                sr_clamp = torch.clamp(sr, 0.0, 1.0)
                hr_clamp = torch.clamp(hr, 0.0, 1.0)
                
                acc_psnr += self.metric_psnr(sr_clamp, hr_clamp).mean().detach().float() * batch_size
                acc_ssim += self.metric_ssim(sr_clamp, hr_clamp).mean().detach().float() * batch_size
                acc_lpips += self.metric_lpips(sr_clamp, hr_clamp).mean().detach().float() * batch_size
                acc_niqe += self.metric_niqe(sr_clamp).mean().detach().float() * batch_size # NIQE is no-reference
                
                n_samples += batch_size
                
        stats = torch.stack([val_loss, acc_psnr, acc_ssim, acc_lpips, acc_niqe, n_samples])
        if self.accelerator is not None:
            stats = self.accelerator.reduce(stats, reduction="sum")

        count = stats[5].clamp_min(1.0)
        avg_val_loss = (stats[0] / count).item()
        avg_psnr = (stats[1] / count).item()
        avg_ssim = (stats[2] / count).item()
        avg_lpips = (stats[3] / count).item()
        avg_niqe = (stats[4] / count).item()
        
        if self.writer:
            self._print(f"[Validation][Epoch {epoch}] Loss: {avg_val_loss:.6f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | NIQE: {avg_niqe:.4f}")
            self.writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            self.writer.add_scalar("Metrics/PSNR", avg_psnr, epoch)
            self.writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)
            self.writer.add_scalar("Metrics/LPIPS", avg_lpips, epoch)
            self.writer.add_scalar("Metrics/NIQE", avg_niqe, epoch)
            
            self.writer.add_scalar("Metrics/NIQE", avg_niqe, epoch)
            
        return {
            'loss': avg_val_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips,
            'niqe': avg_niqe
        }

class Trainer(BaseEngine):
    def __init__(self, model, config, device, writer=None, checkpoint_dir=None):
        super().__init__(device)
        self.model = model
        self.cfg = config
        self.writer = writer
        
        # Initialize Accelerator
        grad_accum_steps = self.cfg['train'].get('gradient_accumulation_steps', 1)
        self.accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)
        self.device = self.accelerator.device # Override base device
        if not self.accelerator.is_main_process:
            self.writer = None
        
        # Setup Loss
        # Accelerator handles device placement, but criterion is often created before prepare
        self.criterion = UnifiedLoss(self.cfg.get('loss', {})).to(self.device)
        
        # Optimizer
        opt_cfg = self.cfg['train'].get('optimizer', {})
        opt_type = opt_cfg.get('type', 'Adam')
        lr = float(self.cfg['train']['lr'])
        weight_decay = float(self.cfg['train'].get('weight_decay', 0.0))
        
        if opt_type == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=opt_cfg.get('momentum', 0.9), 
                weight_decay=weight_decay
            )
        elif opt_type == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                betas=opt_cfg.get('betas', (0.9, 0.999)),
                weight_decay=weight_decay
            )
        else: # Default Adam
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                betas=opt_cfg.get('betas', (0.9, 0.999))
            )
        
        # Scheduler
        sched_cfg = self.cfg['train'].get('scheduler', {})
        sched_type = sched_cfg.get('type', 'StepLR')
        self.scheduler_type = 'step' # Default assumption
        
        if sched_type == 'ReduceLROnPlateau':
            self.scheduler_type = 'plateau'
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=sched_cfg.get('factor', 0.5), 
                patience=sched_cfg.get('patience', 10),
                verbose=True
            )
        elif sched_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(sched_cfg.get('T_max', self.cfg['train']['epochs'])),
                eta_min=float(sched_cfg.get('eta_min', 0))
            )
        else: # Default StepLR
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.cfg['train'].get('step_size', 50), 
                gamma=self.cfg['train'].get('gamma', 0.5)
            )
        
        # Checkpoint Utils
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            # Fallback (Legacy)
            exp_name = f"{self.cfg['model']['name']}_{self.cfg['task']}_x{self.cfg['model']['scale']}"
            self.checkpoint_dir = os.path.join("checkpoints", exp_name)
            
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.start_epoch = 1
        
        # Validator
        self.task = self.cfg.get('task', 'sr')
        self.is_guided_model = self.cfg.get('model', {}).get('guided', True)
        self.validator = Validator(self.model, self.criterion, self.device, self.writer, task=self.task, is_guided_model=self.is_guided_model, accelerator=self.accelerator)

        # Early Stopping
        early_stop_cfg = self.cfg['train'].get('early_stopping', {})
        self.patience = early_stop_cfg.get('patience', 10)
        self.min_delta = early_stop_cfg.get('min_delta', 0.0)
        self.monitor_metric = early_stop_cfg.get('monitor', 'loss')
        self.monitor_mode = early_stop_cfg.get('mode', 'min')
        self.early_stop_counter = 0
        
        self.best_metric_val = float('inf') if self.monitor_mode == 'min' else -float('inf')

    @staticmethod
    def _strip_state_dict_prefix(state_dict, prefix):
        if not state_dict:
            return state_dict

        keys = list(state_dict.keys())
        if all(key.startswith(prefix) for key in keys):
            return OrderedDict((key[len(prefix):], value) for key, value in state_dict.items())

        return state_dict

    def _normalize_model_state_dict(self, state_dict):
        model_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(state_dict.keys())

        if not model_keys or not checkpoint_keys:
            return state_dict

        model_uses_module_prefix = all(key.startswith("module.") for key in model_keys)
        checkpoint_uses_module_prefix = all(key.startswith("module.") for key in checkpoint_keys)

        if checkpoint_uses_module_prefix and not model_uses_module_prefix:
            return self._strip_state_dict_prefix(state_dict, "module.")

        return state_dict

    def prepare_accelerator(self, train_loader, val_loader=None):
        """
        Prepare model, optimizer, scheduler and loaders with Accelerator.
        Must be called before training loop starts.
        """
        if val_loader:
            self.model, self.optimizer, self.scheduler, train_loader, val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, train_loader, val_loader
            )
        else:
            self.model, self.optimizer, self.scheduler, train_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, train_loader
            )
            
        return train_loader, val_loader

    def resume(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"[Trainer] Checkpoint not found: {checkpoint_path}")
            return
            
        print(f"[Trainer] Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_state_dict = self._normalize_model_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_metric_val = checkpoint.get('best_metric_val', self.best_val_loss if self.monitor_mode == 'min' else -float('inf'))
        # Config is now loaded from file in train.py before Trainer init, so strictly we don't need this override.
        # We disable this to allow users to edit train_config.yaml in the folder and have it apply on resume.
        # if 'config' in checkpoint:
        #     self.cfg = checkpoint['config'] 
        
        print(f"[Trainer] Resumed at epoch {self.start_epoch} (Best Loss: {self.best_val_loss:.6f})")

    def train_epoch(self, loader, epoch, log_interval=10):
        self.model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        
        num_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_local_main_process)
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with self.accelerator.accumulate(self.model):
                if self.task == 'guide' and 'guide' in batch and self.is_guided_model:
                    guide = batch['guide'].to(self.device)
                    sr = self.model(lr, guide)
                else:
                    sr = self.model(lr)
                
                # CoReFusion returns (sr, proj1, proj2) — extract extras for contrastive loss
                extras = None
                if isinstance(sr, tuple):
                    extras = sr[1:]
                    sr = sr[0]
                
                loss, loss_components = self.criterion(sr, hr, lr)
                
                # Contrastive loss (if model returned projection features)
                if extras is not None and len(extras) >= 2:
                    from src.losses.losses import ContrastiveLoss
                    contrastive_w = self.cfg.get('contrastive_weight', 0.0)
                    if contrastive_w > 0:
                        contrastive_fn = ContrastiveLoss()
                        c_loss = contrastive_fn(extras[0], extras[1])
                        loss = loss + contrastive_w * c_loss
                        loss_components['contrastive'] = c_loss.item()
                
                self.accelerator.backward(loss)
                
                # Gradient Clipping
                grad_clip = self.cfg['train'].get('grad_clip', 0.0)
                if grad_clip > 0 and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                    
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_steps += 1
            
            if (i + 1) % log_interval == 0 and self.writer:
                avg_loss = running_loss / log_interval
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.writer.add_scalar("Loss/Total", avg_loss, epoch * len(loader) + i)
                self.writer.add_scalar("Train/LR", current_lr, epoch * len(loader) + i)
                
                # Log components
                for name, val in loss_components.items():
                    self.writer.add_scalar(f"Loss/{name}", val, epoch * len(loader) + i)

                pbar.set_postfix({'loss': f"{avg_loss:.6f}", 'lr': f"{current_lr:.2e}"})
                running_loss = 0.0
                
        stats = torch.tensor([epoch_loss, num_steps], device=self.device, dtype=torch.float32)
        stats = self.accelerator.reduce(stats, reduction="sum")
        return (stats[0] / stats[1].clamp_min(1.0)).item()

    def fit(self, train_loader, val_loader=None, epochs=100):
        # Prepare components with accelerator
        train_loader, val_loader = self.prepare_accelerator(train_loader, val_loader)
        
        self.accelerator.print(f"Start training from epoch {self.start_epoch} to {epochs}...")
        
        # Initialize CSV logging
        csv_path = os.path.join(self.checkpoint_dir, "training_log.csv")
        file_exists = os.path.isfile(csv_path)
        
        if self.accelerator.is_main_process:
            csv_file = open(csv_path, mode='a', newline='')
            fieldnames = ['epoch', 'train_loss', 'val_loss', 'psnr', 'ssim', 'lpips', 'niqe', 'lr']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
        else:
            csv_file = None
            writer = None

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                train_loss = self.train_epoch(train_loader, epoch)
                
                val_metrics = {}
                if val_loader:
                    val_metrics = self.validator.run(val_loader, epoch)
                    val_loss = val_metrics['loss']
                else:
                    val_loss = None
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Write to CSV
                log_row = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss if val_loss is not None else '',
                    'psnr': val_metrics.get('psnr', ''),
                    'ssim': val_metrics.get('ssim', ''),
                    'lpips': val_metrics.get('lpips', ''),
                    'niqe': val_metrics.get('niqe', ''),
                    'lr': current_lr
                }
                if self.accelerator.is_main_process and writer is not None:
                    writer.writerow(log_row)
                    csv_file.flush()
                
                if self.scheduler_type == 'plateau':
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                if val_loss is not None:
                    current_metric = val_metrics.get(self.monitor_metric, val_loss)
                    is_best = False
                    
                    if self.monitor_mode == 'min':
                        if current_metric < (self.best_metric_val - self.min_delta):
                            is_best = True
                    else:
                        if current_metric > (self.best_metric_val + self.min_delta):
                            is_best = True

                    if is_best:
                        self.best_metric_val = current_metric
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, is_best=True)
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        self.accelerator.print(f"[Trainer] EarlyStopping counter: {self.early_stop_counter} out of {self.patience} (Monitor: {self.monitor_metric})")
                        if self.early_stop_counter >= self.patience:
                            self.accelerator.print("[Trainer] Early stopping triggered.")
                            break
                
                if epoch % self.cfg['train']['save_interval'] == 0:
                    self.save_checkpoint(epoch, is_best=False)
                    
                # Always save last
                self.save_checkpoint(epoch, is_best=False, filename="last.pth")
        finally:
            if csv_file is not None:
                csv_file.close()

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        if filename is None:
            filename = f"epoch_{epoch}.pth"
            
        save_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save unified state dict
        state = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metric_val': self.best_metric_val,
            'config': self.cfg
        }
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            if self.accelerator.is_main_process:
                torch.save(state, best_path)
                metric_str = f"{self.monitor_metric}: {self.best_metric_val:.6f}"
                print(f"Saved Best Checkpoint: {best_path} ({metric_str})")
        else:
            if self.accelerator.is_main_process:
                torch.save(state, save_path)
                print(f"Saved checkpoint: {save_path}")

        self.accelerator.wait_for_everyone()

    def export_onnx(self, input_shape=(1, 3, 360, 640)):
        print("Exporting ONNX...")
        dummy_input = torch.randn(input_shape).to(self.device)
        onnx_path = os.path.join(self.checkpoint_dir, "final_model.onnx")
        
        # Handle switch_to_deploy if available
        if hasattr(self.model, 'switch_to_deploy'):
            self.model.switch_to_deploy()
            
        torch.onnx.export(self.model, dummy_input, onnx_path, opset_version=11, 
                          input_names=['input'], output_names=['output'])
        print(f"Exported ONNX to {onnx_path}")
