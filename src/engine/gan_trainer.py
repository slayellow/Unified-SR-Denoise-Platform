"""
GAN Trainer for LapGSR adversarial training.
Inherits from Trainer and overrides train_epoch() for G/D alternating training.

Training loop per batch:
  1. D step: D(hr) → real loss, D(sr.detach()) → fake loss, gradient penalty
  2. G step: D(sr) → adversarial loss + reconstruction loss (MSE)
"""

import torch
import torch.optim as optim
from tqdm import tqdm

from src.engine.trainer import Trainer
from src.losses.losses import GANLoss, UnifiedLoss, gradient_penalty


class GANTrainer(Trainer):
    """
    Extends Trainer with adversarial training support for LapGSR.
    
    Additional config keys:
      - discriminator_model: nn.Module (passed at init)
      - gan_weight: weight for adversarial loss (default 1.0)
      - gp_weight: weight for gradient penalty (default 100.0)
    """
    
    def __init__(self, model, discriminator, config, device, writer=None, checkpoint_dir=None):
        # Initialize parent (sets up generator model, optimizer, scheduler, criterion, etc.)
        super().__init__(model, config, device, writer, checkpoint_dir)
        
        self.discriminator = discriminator.to(self.device)
        self.gan_loss = GANLoss().to(self.device)
        self.gan_weight = config.get('gan_weight', 1.0)
        self.gp_weight = config.get('gp_weight', 100.0)
        
        # Discriminator optimizer
        d_opt_cfg = config['train'].get('d_optimizer', config['train'].get('optimizer', {}))
        d_lr = float(d_opt_cfg.get('lr', config['train']['lr']))
        d_betas = d_opt_cfg.get('betas', (0.9, 0.99))
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=d_lr,
            betas=d_betas,
        )
        
        # Prepare discriminator with accelerator
        self.discriminator, self.d_optimizer = self.accelerator.prepare(
            self.discriminator, self.d_optimizer
        )

    def train_epoch(self, loader, epoch, log_interval=10):
        self.model.train()
        self.discriminator.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} [GAN]")
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            guide = batch['guide'].to(self.device) if 'guide' in batch else lr
            
            # ========== Discriminator Step ==========
            self.d_optimizer.zero_grad()
            
            with torch.no_grad():
                sr = self.model(lr, guide)
                if isinstance(sr, tuple):
                    sr = sr[0]
            
            d_real = self.discriminator(hr)
            d_fake = self.discriminator(sr.detach())
            
            d_loss_real = self.gan_loss(d_real, is_real=True)
            d_loss_fake = self.gan_loss(d_fake, is_real=False)
            
            # Gradient penalty
            gp = gradient_penalty(self.discriminator, hr, sr.detach(), self.device)
            
            d_loss = d_loss_real + d_loss_fake + self.gp_weight * gp
            
            self.accelerator.backward(d_loss)
            self.d_optimizer.step()
            
            # ========== Generator Step ==========
            self.optimizer.zero_grad()
            
            with self.accelerator.accumulate(self.model):
                sr = self.model(lr, guide)
                if isinstance(sr, tuple):
                    sr = sr[0]
                
                # Reconstruction loss (from UnifiedLoss — typically MSE with weight)
                recon_loss, loss_components = self.criterion(sr, hr, lr)
                
                # Adversarial loss
                d_fake_for_g = self.discriminator(sr)
                g_adv_loss = self.gan_loss(d_fake_for_g, is_real=True)
                
                g_loss = recon_loss + self.gan_weight * g_adv_loss
                
                self.accelerator.backward(g_loss)
                
                grad_clip = self.cfg['train'].get('grad_clip', 0.0)
                if grad_clip > 0 and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            if (i + 1) % log_interval == 0 and self.writer:
                avg_g = running_g_loss / log_interval
                avg_d = running_d_loss / log_interval
                step = epoch * len(loader) + i
                
                self.writer.add_scalar("Loss/G_Total", avg_g, step)
                self.writer.add_scalar("Loss/D_Total", avg_d, step)
                self.writer.add_scalar("Loss/Recon", loss_components.get('mse', recon_loss.item()), step)
                self.writer.add_scalar("Loss/G_Adv", g_adv_loss.item(), step)
                self.writer.add_scalar("Loss/GP", gp.item(), step)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'G': f"{avg_g:.4f}", 
                    'D': f"{avg_d:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                running_g_loss = 0.0
                running_d_loss = 0.0
        
        return epoch_g_loss / len(loader)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """Save both generator and discriminator states."""
        import os
        if filename is None:
            filename = f"epoch_{epoch}.pth"
        
        save_path = os.path.join(self.checkpoint_dir, filename)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.cfg,
        }
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(state, best_path)
            print(f"Saved Best Checkpoint: {best_path} (Loss: {self.best_val_loss:.6f})")
        else:
            torch.save(state, save_path)
            print(f"Saved checkpoint: {save_path}")

    def resume(self, checkpoint_path):
        """Resume both generator and discriminator."""
        import os
        if not os.path.exists(checkpoint_path):
            print(f"[GANTrainer] Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"[GANTrainer] Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if 'd_optimizer_state_dict' in checkpoint:
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"[GANTrainer] Resumed at epoch {self.start_epoch} (Best Loss: {self.best_val_loss:.6f})")
