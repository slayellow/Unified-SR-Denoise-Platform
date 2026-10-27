import contextlib
import os
import csv
import subprocess
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pyiqa

from src.metrics import METRIC_SPEC, create_lpips, create_niqe, create_psnr, create_ssim
from src.engine import visualize

# Keys the Validator puts in val_metrics; early_stopping.monitor must name one.
VALID_MONITOR_METRICS = frozenset({'loss', 'psnr', 'ssim', 'lpips', 'niqe'})
from accelerate import Accelerator
from src.losses.losses import UnifiedLoss
from src.engine.schedulers import KneeLRScheduler

def git_sha(repo_root):
    """Short HEAD sha of repo_root, '-dirty' when tracked files differ from it; 'unknown' outside git."""
    # ponytail: an untracked new module does not mark the tree dirty; add --untracked-files if that matters
    try:
        sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, cwd=repo_root, timeout=5).stdout.strip()
        if not sha:
            return 'unknown'
        dirty = subprocess.run(['git', '--no-optional-locks', 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, cwd=repo_root, timeout=5).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


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
    def __init__(self, model, criterion, device, writer=None, task='sr', accelerator=None, scale=1):
        super().__init__(device)
        self.model = model
        self.criterion = criterion
        self.writer = writer
        self.task = task
        self.accelerator = accelerator
        self.scale = scale

        # One definition for the whole repo -- see src/metrics. pyiqa's defaults
        # measured PSNR on RGB and SSIM on YIQ, so the two numbers in a single
        # row of training_log.csv did not even agree with each other.
        self._print(f"[Validator] Metrics: {METRIC_SPEC}, scale={scale} (PSNR, SSIM, LPIPS, NIQE)")
        self.metric_psnr = create_psnr(scale, device)
        self.metric_ssim = create_ssim(scale, device)
        self.metric_lpips = create_lpips(device)
        self.metric_niqe = create_niqe(scale, device)
        self._shape_mismatch_warned = False

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
                
                sr = self.model(lr)

                # Validation pairs can contain non-divisible sizes (e.g. x3 with width 1280).
                # Align to common spatial size to keep loss/metric computation stable.
                if sr.shape[-2:] != hr.shape[-2:]:
                    min_h = min(sr.shape[-2], hr.shape[-2])
                    min_w = min(sr.shape[-1], hr.shape[-1])
                    if min_h > 0 and min_w > 0:
                        if not self._shape_mismatch_warned:
                            self._print(
                                f"[Validator] Shape mismatch detected (SR={tuple(sr.shape[-2:])}, HR={tuple(hr.shape[-2:])}). "
                                f"Cropping both to ({min_h}, {min_w}) for validation."
                            )
                            self._shape_mismatch_warned = True
                        sr = sr[..., :min_h, :min_w]
                        hr = hr[..., :min_h, :min_w]
                
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
        elif opt_type == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_cfg.get('betas', (0.9, 0.999))
            )
        else:
            # Silently defaulting to Adam meant a typo trained a different
            # optimizer than the config claimed, and the run looked normal.
            raise ValueError(
                f"알 수 없는 optimizer type: {opt_type!r}. SGD / AdamW / Adam 중 하나여야 합니다."
            )
        
        # Scheduler
        sched_cfg = self.cfg['train'].get('scheduler', {})
        sched_type = sched_cfg.get('type', 'StepLR')
        
        if sched_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(sched_cfg.get('T_max', self.cfg['train']['epochs'])),
                eta_min=float(sched_cfg.get('eta_min', 0))
            )
        elif sched_type == 'KneeLR':
            self.scheduler = KneeLRScheduler(
                self.optimizer,
                peak_lr=float(sched_cfg.get('peak_lr', self.cfg['train']['lr'])),
                warmup_steps=int(sched_cfg.get('warmup_steps', 0)),
                explore_steps=int(sched_cfg.get('explore_steps', 0)),
                total_steps=int(sched_cfg.get('total_steps', self.cfg['train']['epochs'])),
                min_lr=float(sched_cfg.get('min_lr', 0.0)),
            )
        elif sched_type == 'StepLR':
            # Note the nesting: these two have always been read from
            # cfg['train'], not from cfg['train']['scheduler'], so a config that
            # puts them where every other scheduler key lives is silently
            # ignored and the defaults win.
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get('step_size', self.cfg['train'].get('step_size', 50)),
                gamma=sched_cfg.get('gamma', self.cfg['train'].get('gamma', 0.5))
            )
        else:
            # This used to fall through to StepLR(50, 0.5) without a word, so a
            # typo -- 'SteplR', 'CosineAnnealing' -- trained a completely
            # different LR curve and looked like a modelling result.
            raise ValueError(
                f"알 수 없는 scheduler type: {sched_type!r}. "
                f"KneeLR / CosineAnnealingLR / StepLR 중 하나여야 합니다."
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
        self.scale = int(self.cfg.get('model', {}).get('scale', 1) or 1)
        self.validator = Validator(self.model, self.criterion, self.device, self.writer, task=self.task, accelerator=self.accelerator, scale=self.scale)

        # Deployment softening. The SR input is not what this pipeline used to
        # synthesise: measured 2026-09-08 on the board, the tensor the model
        # actually receives carries 0.0065 of its energy above half-Nyquist,
        # while the training LR carries 0.0227 -- 3.5x sharper than anything
        # deployment produces. A model trained on the sharper distribution has
        # nothing to remove at run time, which is why the shipped x2 output sat
        # 1.8 DN from a plain upscale.
        #
        # The fix is not to pin training to the denoised path, because the same
        # EO weights also serve a camera that has no denoise stage
        # (gst_compare_eo4k.sh). Pinning would repeat v1's mistake with the sign
        # flipped. So each batch is split: part of it goes through the deployed
        # denoise net -- the exact operator, not an approximation of it, which
        # costs a measured 3x in high-frequency energy -- and the rest stays as
        # it is. One model then covers both paths.
        #
        # Noise must already be in the LR when this runs, and it is: the
        # degradation pipeline adds it before returning. Handing the denoise net
        # a clean frame would pass it through nearly unchanged and reproduce
        # none of the softening this exists to reproduce.
        soft_cfg = self.cfg['train'].get('deploy_softening', {})
        self.soften_prob = float(soft_cfg.get('denoise_prob', 0.0))
        # Default matches the deployed denoise net's own training distribution
        # (stage2 common_noise, sigma 2..15), which is the noise level it is
        # calibrated to remove.
        ns = soft_cfg.get('noise_sigma', [2.0, 15.0])
        self.soften_noise_sigma = tuple(float(v) for v in ns) if ns else None
        self.denoise_net = None
        if self.soften_prob > 0:
            self.denoise_net = self._load_denoise_net(soft_cfg)

        # Weight EMA. The last step's weights are one draw from the noise the
        # optimizer is still bouncing through; the running average of the tail
        # is a steadier point in the same basin, and it costs one extra copy of
        # the weights (89 KB here) plus a lerp per step.
        #
        # It only pays if the thing being judged and shipped is the average, so
        # validation and every checkpoint below read the EMA copy, not the live
        # one. Training itself keeps stepping the live weights -- feeding the
        # average back into the optimizer would damp the exploration the
        # average is summarising.
        ema_cfg = self.cfg['train'].get('ema', {})
        self.ema_enabled = bool(ema_cfg.get('enabled', False))
        self.ema_decay = float(ema_cfg.get('decay', 0.999))
        # Below this epoch the average is still dominated by its initialisation,
        # so judging on it would score noise. Default is one epoch of warmup.
        self.ema_start_epoch = int(ema_cfg.get('start_epoch', 2))
        self.ema_state = None

        # Early Stopping
        early_stop_cfg = self.cfg['train'].get('early_stopping', {})
        self.patience = early_stop_cfg.get('patience', 10)
        self.min_delta = early_stop_cfg.get('min_delta', 0.0)
        self.monitor_metric = early_stop_cfg.get('monitor', 'loss')
        self.monitor_mode = early_stop_cfg.get('mode', 'min')
        # A typo here used to fall through to val_loss without a word, so a run
        # configured to track 'lpips' would quietly early-stop on loss instead.
        if self.monitor_metric not in VALID_MONITOR_METRICS:
            raise ValueError(
                f"early_stopping.monitor={self.monitor_metric!r} is not a metric this "
                f"trainer computes. Choose one of {sorted(VALID_MONITOR_METRICS)}."
            )
        if self.monitor_mode not in ('min', 'max'):
            raise ValueError(f"early_stopping.mode={self.monitor_mode!r} must be 'min' or 'max'.")
        # A warm-up raises the weighted validation loss epoch by epoch, so judging
        # on 'loss' would keep the first, least-weighted epochs as best.
        warming = sorted(n for n, e in getattr(self.criterion, 'warmup_epochs', {}).items() if e > 0)
        if self.monitor_metric == 'loss' and warming:
            raise ValueError(
                f"early_stopping.monitor='loss' with a loss warm-up ({', '.join(warming)}): the loss is not "
                f"comparable across epochs. Monitor a metric (psnr, ssim, lpips, niqe) instead."
            )
        self.early_stop_counter = 0
        
        self.best_metric_val = float('inf') if self.monitor_mode == 'min' else -float('inf')

        # The config snapshot says what was asked for; the SHA says which code
        # interpreted it. Same config + different loss/degradation code = a
        # different model, and without this the difference left no trace.
        self.git_sha = git_sha(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # Set by the caller when the weights come from a previous phase. Without a
        # baseline, best_metric_val starts at infinity and epoch 1 is written as
        # "best" whatever it scores -- so a finetune could ship a model worse than
        # the checkpoint it started from, and the log would never say so.
        self.validate_before_fit = False
        self._best_attempt = None

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
        Prepare model, optimizer and loaders with Accelerator.
        Must be called before training loop starts.

        The scheduler is deliberately NOT prepared: it is stepped once per
        epoch, and AcceleratedScheduler would step it num_processes times per
        call -- and freeze KneeLRScheduler outright once _step_count exceeds
        total_steps, via accelerate's OneCycle total_steps guard, which our
        scheduler trips simply by having a total_steps attribute. For the
        single-process runs this repo launches the two are bit-identical
        (tests/test_scheduler_prepare.py proves it); for multi-process the
        wrapped behavior was the distorted one.
        """
        if val_loader:
            self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
                self.model, self.optimizer, train_loader, val_loader
            )
        else:
            self.model, self.optimizer, train_loader = self.accelerator.prepare(
                self.model, self.optimizer, train_loader
            )
            
        return train_loader, val_loader

    def resume(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            # Printing and carrying on trained from random init under a run that
            # claimed to be a resume.
            raise FileNotFoundError(f"[Trainer] resume checkpoint not found: {checkpoint_path}")

        print(f"[Trainer] Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Resume the trajectory, not the summary of it. When EMA is on,
        # model_state_dict holds the averaged weights that were judged and are
        # meant to ship; live_state_dict holds where the optimizer actually
        # was. Restarting from the average would discard the exploration the
        # optimizer state is still describing.
        _resume_from = checkpoint.get('live_state_dict') or checkpoint['model_state_dict']
        model_state_dict = self._normalize_model_state_dict(_resume_from)
        self.model.load_state_dict(model_state_dict)
        if checkpoint.get('ema_used'):
            # Same non-tensor carve-out as _ema_update: a QuantSim checkpoint
            # holds encoding dicts alongside the weights.
            self.ema_state = {
                k: (v.detach().clone().float() if torch.is_tensor(v) else v)
                for k, v in self._normalize_model_state_dict(
                    checkpoint['model_state_dict']).items()}
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_metric_val = checkpoint.get('best_metric_val', self.best_val_loss if self.monitor_mode == 'min' else -float('inf'))
        # Checkpoints from before 2026-10-16 lack these two; they resume as they always did.
        self._best_attempt = checkpoint.get('best_attempt')
        self.early_stop_counter = checkpoint.get('early_stop_counter', 0)
        # Resume keeps the train_config.yaml that tools/train.py read, not checkpoint['config'].

        print(f"[Trainer] Resumed at epoch {self.start_epoch} (Best Loss: {self.best_val_loss:.6f})")

    def _load_denoise_net(self, cfg):
        """Build the frozen denoise net that stands in for the deployed one.

        Its weights decide what "deployment softening" means, so the checkpoint
        path is recorded in every saved config: swap the deployed denoise model
        and this training run has silently gone stale.
        """
        from src.models import build_model
        from src.engine.checkpoint import normalize_state_dict

        path = cfg.get('denoise_checkpoint')
        if not path or not os.path.isfile(path):
            # An unreadable path is a mistake, not a reason to train on the
            # wrong distribution -- the softening would silently not happen and
            # the run would look normal. August cost two days to this class of
            # silent fallback.
            raise FileNotFoundError(
                f"deploy_softening.denoise_prob is {self.soften_prob} but "
                f"denoise_checkpoint is missing: {path!r}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model_cfg = cfg.get('denoise_model') or (ckpt.get('config') or {}).get('model')
        if not model_cfg:
            raise ValueError(
                f"{path} carries no model config; set deploy_softening.denoise_model")
        net = build_model(dict(model_cfg))
        net.load_state_dict(normalize_state_dict(ckpt.get('model_state_dict', ckpt)))
        if hasattr(net, 'switch_to_deploy'):
            net.switch_to_deploy()
        net.eval().to(self.device)
        for p in net.parameters():
            p.requires_grad_(False)
        self.accelerator.print(
            f"[Trainer] Deployment softening: {self.soften_prob:.0%} of each batch "
            f"through {os.path.basename(path)}")
        return net

    def _soften(self, lr):
        """Run part of the batch through the denoise net, in place of the batch.

        Split per sample rather than per batch so both paths appear in every
        gradient step; a whole-batch coin flip would make the loss alternate
        between two distributions and the optimizer would chase the alternation.
        """
        if self.denoise_net is None or self.soften_prob <= 0:
            return lr
        pick = torch.rand(lr.shape[0], device=lr.device) < self.soften_prob
        if not bool(pick.any()):
            return lr
        sel = lr[pick]

        # Sensor noise goes in here, not in the degradation recipe, because in
        # deployment it comes from the sensor and the denoise net is what
        # removes it. The SR recipe supplies gaussian prob 0.1 sigma 1..2 while
        # the deployed denoise was trained at prob 0.5 sigma 2..15, so without
        # this most samples hand it a frame far cleaner than anything it was
        # calibrated for.
        #
        # Honest note on what this bought: measured over 20 validation frames,
        # adding it changed the roll-off drop from 32% to 31% -- nothing. The
        # softening this net applies turns out to be largely independent of how
        # noisy its input is. It is kept because deployment does have sensor
        # noise at this point and matching the operator's input costs nothing,
        # not because a benefit was demonstrated. Set noise_sigma to null to
        # drop it.
        #
        # It does not push the model toward denoising either way: the noise is
        # added and removed inside this function, so what reaches the model is
        # the softening left behind, not the noise.
        if self.soften_noise_sigma:
            lo, hi = self.soften_noise_sigma
            sigma = torch.empty(sel.shape[0], 1, 1, 1, device=sel.device).uniform_(lo, hi) / 255.0
            sel = (sel + torch.randn_like(sel) * sigma).clamp(0.0, 1.0)

        with torch.no_grad():
            out = self.denoise_net(sel)
            if isinstance(out, tuple):
                out = out[0]
        lr = lr.clone()
        lr[pick] = out.clamp(0.0, 1.0).to(lr.dtype)
        return lr

    def _ema_update(self):
        """Fold the current weights into the running average.

        Every floating-point entry is averaged, buffers included: BatchNorm
        running_mean/running_var follow the weights they normalise, so the
        EMA copy carries statistics that match its own weights, and that pair
        is what validation reads and the checkpoint saves. Integer buffers
        (num_batches_tracked) are copied.

        Not every state_dict entry is a tensor. An AIMET QuantSim model stores
        each quantizer's encoding as a nested dict, so QAT reaches here with
        entries that have no dtype at all; averaging is meaningless for them
        anyway -- an encoding is a range, not a weight, and QAT learns it
        through its own parametrisation. Those entries are carried across
        untouched so _ema_weights can still load_state_dict the result.
        """
        if not self.ema_enabled:
            return
        model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            if self.ema_state is None:
                self.ema_state = {
                    k: (v.detach().clone().float() if torch.is_tensor(v) else v)
                    for k, v in model.state_dict().items()}
                return
            d = self.ema_decay
            for k, v in model.state_dict().items():
                if not torch.is_tensor(v):
                    self.ema_state[k] = v
                elif v.dtype.is_floating_point:
                    self.ema_state[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                else:
                    self.ema_state[k] = v.detach().clone()

    @contextlib.contextmanager
    def _ema_weights(self, epoch):
        """Swap the EMA weights in for the duration of the block.

        A context manager rather than a copy of the model: at 22.7k parameters
        a second module is affordable, but swapping keeps validation reading the
        exact object the Validator was constructed around, so there is no way
        for the two to drift apart.
        """
        use = (self.ema_enabled and self.ema_state is not None
               and epoch >= self.ema_start_epoch)
        if not use:
            yield False
            return
        model = self.accelerator.unwrap_model(self.model)
        # Non-tensor entries (QuantSim encodings) pass through both ways: they
        # are not averaged, and load_state_dict wants them back in the shape it
        # handed them over.
        backup = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                  for k, v in model.state_dict().items()}
        try:
            model.load_state_dict({
                k: (v.to(backup[k].dtype) if torch.is_tensor(v) else v)
                for k, v in self.ema_state.items()})
            yield True
        finally:
            model.load_state_dict(backup)

    def train_epoch(self, loader, epoch, log_interval=10):
        self.model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        
        num_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_local_main_process)
        for i, batch in enumerate(pbar):
            lr = self._soften(batch['lr'].to(self.device))
            hr = batch['hr'].to(self.device)

            # Gradients are cleared only inside accumulate(), after the step:
            # clearing them here as well read as if every micro-batch discarded
            # the one before (it did not -- the prepared optimizer skips
            # zero_grad on non-sync steps -- but nothing in the code said so).
            with self.accelerator.accumulate(self.model):
                sr = self.model(lr)

                loss, loss_components = self.criterion(sr, hr, lr)
                
                self.accelerator.backward(loss)
                
                # Gradient Clipping
                grad_clip = self.cfg['train'].get('grad_clip', 0.0)
                if grad_clip > 0 and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                    
                self.optimizer.step()
                self.optimizer.zero_grad()
                # The optimizer only steps when gradients sync; averaging on every
                # micro-batch would fold the unchanged weights in again.
                if self.accelerator.sync_gradients:
                    self._ema_update()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_steps += 1
            
            if (i + 1) % log_interval == 0 and self.writer:
                avg_loss = running_loss / log_interval
                current_lr = self.optimizer.param_groups[0]['lr']
                # Epochs are 1-indexed; the first step of epoch 1 is step 0.
                step = (epoch - 1) * len(loader) + i

                self.writer.add_scalar("Loss/Total", avg_loss, step)
                self.writer.add_scalar("Train/LR", current_lr, step)

                # Log components
                for name, val in loss_components.items():
                    self.writer.add_scalar(f"Loss/{name}", val, step)

                pbar.set_postfix({'loss': f"{avg_loss:.6f}", 'lr': f"{current_lr:.2e}"})
                running_loss = 0.0
                
        stats = torch.tensor([epoch_loss, num_steps], device=self.device, dtype=torch.float32)
        stats = self.accelerator.reduce(stats, reduction="sum")
        return (stats[0] / stats[1].clamp_min(1.0)).item()

    def fit(self, train_loader, val_loader=None, epochs=100):
        # Checked up front: a missing key used to surface as a KeyError only
        # after the first epoch had trained.
        save_interval = self.cfg['train'].get('save_interval')
        if not isinstance(save_interval, int) or isinstance(save_interval, bool) or save_interval < 1:
            raise ValueError(
                f"train.save_interval 은 1 이상의 정수여야 합니다 (지금 값: {save_interval!r}).")

        # Prepare components with accelerator
        train_loader, val_loader = self.prepare_accelerator(train_loader, val_loader)
        
        if self.validate_before_fit and val_loader is not None:
            baseline = self.validator.run(val_loader, self.start_epoch - 1)
            self.best_val_loss = baseline['loss']
            self.best_metric_val = baseline.get(self.monitor_metric, baseline['loss'])
            self.accelerator.print(
                f"[Trainer] Baseline from pretrained weights: "
                f"{self.monitor_metric}={self.best_metric_val:.6f} "
                f"(loss={self.best_val_loss:.6f}). best.pth is only written if this is beaten."
            )

        self.accelerator.print(f"Start training from epoch {self.start_epoch} to {epochs}...")

        # Run artifacts, ultralytics-style (train.plots: false to disable).
        # Rendering must never kill training: every call is guarded, and the
        # start-time renders run under preserved_rng so drawing a sample batch
        # does not shift the training's random stream.
        self.plots_enabled = bool(self.cfg['train'].get('plots', True)) and self.accelerator.is_main_process
        plots_dir = os.path.join(self.checkpoint_dir, "plots")
        if self.plots_enabled:
            os.makedirs(plots_dir, exist_ok=True)
            try:
                with visualize.preserved_rng():
                    ds = getattr(train_loader, 'dataset', None)
                    if ds is not None and hasattr(ds, '__getitem__') and len(ds) > 0:
                        idx = [i % len(ds) for i in range(8)]
                        # One __getitem__ per index: the dataset random-crops on
                        # every call, so indexing twice would pair an LR from one
                        # crop with an HR from another.
                        items = [ds[i] for i in idx]
                        batch = {
                            'lr': torch.stack([it['lr'] for it in items]),
                            'hr': torch.stack([it['hr'] for it in items]),
                        }
                        visualize.render_train_batch(batch, os.path.join(plots_dir, "train_batch.png"))
                    if ds is not None and hasattr(ds, 'image_paths'):
                        visualize.render_dataset_stats(ds.image_paths, os.path.join(plots_dir, "dataset_stats.png"))
            except Exception as exc:
                self.accelerator.print(f"[Plots] start-time rendering skipped: {exc}")

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
                # Any loss configured with warmup_epochs is scaled here, before
                # both the training pass and the validation that judges it, so
                # the two always agree on the objective. Resuming works because
                # the weights are a pure function of the epoch number.
                if hasattr(self.criterion, 'set_epoch'):
                    self.criterion.set_epoch(epoch)
                    ramping = self.criterion.active_weights()
                    if ramping:
                        self.accelerator.print(
                            f"[Trainer] Epoch {epoch} loss warm-up: "
                            + ", ".join(f"{n}={w:.4g}" for n, w in sorted(ramping.items()))
                        )
                train_loss = self.train_epoch(train_loader, epoch)

                val_metrics = {}
                if val_loader:
                    # Judge the average, since the average is what gets saved.
                    with self._ema_weights(epoch) as using_ema:
                        val_metrics = self.validator.run(val_loader, epoch)
                        if using_ema and epoch == self.ema_start_epoch:
                            self.accelerator.print(
                                f"[Trainer] EMA (decay {self.ema_decay}) now judged and saved")
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
                    if self.plots_enabled:
                        try:
                            visualize.render_results(csv_path, os.path.join(plots_dir, "results.png"))
                        except Exception as exc:
                            self.accelerator.print(f"[Plots] results.png skipped: {exc}")
                
                self.scheduler.step()
                
                if val_loss is not None:
                    current_metric = val_metrics.get(self.monitor_metric, val_loss)

                    # best_attempt: this run's own best, regardless of the seeded
                    # baseline. A finetune that never beats its pretrained baseline
                    # writes no best.pth, and its strongest weights used to vanish
                    # entirely (the x3 near-miss at -0.0056 was unrecoverable).
                    better_attempt = (self._best_attempt is None or
                                      (current_metric < self._best_attempt if self.monitor_mode == 'min'
                                       else current_metric > self._best_attempt))
                    if self.monitor_mode == 'min':
                        is_best = current_metric < (self.best_metric_val - self.min_delta)
                    else:
                        is_best = current_metric > (self.best_metric_val + self.min_delta)

                    # Update the state before saving: the checkpoints carry it and a resume restores it.
                    if better_attempt:
                        self._best_attempt = current_metric
                    if is_best:
                        self.best_metric_val = current_metric
                        self.best_val_loss = val_loss
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                    if better_attempt:
                        self.save_checkpoint(epoch, filename='best_attempt.pth')
                    if is_best:
                        self.save_checkpoint(epoch, is_best=True)
                    else:
                        self.accelerator.print(f"[Trainer] EarlyStopping counter: {self.early_stop_counter} out of {self.patience} (Monitor: {self.monitor_metric})")
                        if self.early_stop_counter >= self.patience:
                            self.accelerator.print("[Trainer] Early stopping triggered.")
                            break
                
                if self.plots_enabled and val_loader is not None and epoch % save_interval == 0:
                    try:
                        vds = getattr(val_loader, 'dataset', None)
                        if vds is not None and len(vds) > 0:
                            picks = sorted({0, len(vds) // 2, len(vds) - 1})
                            samples = [vds[i] for i in picks]
                            # Draw with the weights that were judged and saved (EMA once it starts).
                            with self._ema_weights(epoch):
                                visualize.render_probe(
                                    self.accelerator.unwrap_model(self.model), samples,
                                    self.scale, self.device,
                                    os.path.join(plots_dir, f"probe_ep{epoch}.png"))
                    except Exception as exc:
                        self.accelerator.print(f"[Plots] probe skipped: {exc}")

                if epoch % save_interval == 0:
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

        # Ship what was judged. Validation above scores the EMA weights, so a
        # checkpoint holding the live ones would be a different model than the
        # metric in training_log.csv describes -- the same class of mismatch as
        # scoring FP32 while deploying W8A8. The live weights ride along under
        # a separate key so a resumed run continues the trajectory rather than
        # restarting from the average.
        _live = self.accelerator.unwrap_model(self.model).state_dict()
        _use_ema = (self.ema_enabled and self.ema_state is not None
                    and epoch >= self.ema_start_epoch)
        # QuantSim encodings are dicts, not tensors: they ride along as-is.
        _shipped = ({k: (v.to(_live[k].dtype) if torch.is_tensor(v) else v)
                     for k, v in self.ema_state.items()}
                    if _use_ema else _live)

        # Save unified state dict
        state = {
            'epoch': epoch,
            'model_state_dict': _shipped,
            'ema_used': _use_ema,
            'live_state_dict': _live if _use_ema else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metric_spec': METRIC_SPEC,
            'git_sha': self.git_sha,
            'best_metric_val': self.best_metric_val,
            'best_attempt': self._best_attempt,
            'early_stop_counter': self.early_stop_counter,
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

