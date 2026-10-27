import os
import sys
import shutil
import datetime
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random

# NumPy madvises every array >= 4 MB for transparent hugepages by default. On a
# host whose free memory is fragmented (page cache in the 100s of GB) each such
# fault goes into direct compaction and fails, and the data workers end up at
# 97 % kernel time with the GPU idle (2026-09-10: 6 min/epoch -> 3 h/epoch).
# Must be set before numpy is imported; setdefault so an explicit value wins.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Unified Platform Imports
from src.models import build_model
from src.data.datasets import SRDataset, DenoiseDataset, PairedDataset
from src.engine.trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser(description="Unified SR/Denoise Training")
    parser.add_argument("--config", type=str, help="Path to main train config file")
    parser.add_argument("--data_config", type=str, help="Path to data config file (override default)")
    parser.add_argument("--model", type=str, help="Model name (override config)")
    parser.add_argument("--task", type=str, choices=['sr', 'denoise'], help="Task type (override config)")
    parser.add_argument("--scale", type=int, help="Scale factor (override config)")
    parser.add_argument("--work_dir", type=str, help="Directory to save checkpoints and logs")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from (e.g. checkpoints/exp/last.pth)")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: train.seed in config, else 42)")
    args = parser.parse_args()
    if not args.config and not args.resume:
        parser.error("the following arguments are required: --config unless --resume is provided")
    if args.resume:
        # A mistyped path used to start a fresh run from --config inside the typo's directory.
        if not os.path.isfile(args.resume):
            parser.error(f"--resume {args.resume}: no such checkpoint")
        args.resume = os.path.abspath(args.resume)  # the run directory is its dirname, even for a bare file name
        # The run directory's train_config.yaml and data_config.yaml are the resume's source of truth.
        # --lr too: resume() restores the optimizer and scheduler, which set the learning rate.
        ignored = [f for f, v in (("--config", args.config), ("--work_dir", args.work_dir),
                                  ("--data_config", args.data_config), ("--lr", args.lr)) if v]
        if ignored:
            parser.error(f"--resume reads its config from the run directory and writes there; drop {', '.join(ignored)}")
    return args

def infer_default_data_config_path(config: dict) -> str | None:
    """Infer the default data-config path from the task."""
    task = config.get('task')
    if task == 'denoise':
        return "configs/data/bicubicpp/denoise.yaml"
    if task == 'sr':
        return "configs/data/sr_train.yaml"
    return None


def resolve_dataset_kind(config: dict) -> str:
    """The task alone decides the dataset. dataset_type used to override it,
    which meant a denoise config could say task: sr and still train. An
    unknown or missing task raises."""
    task = str(config.get('task', '')).lower()
    if task not in {'sr', 'denoise'}:
        raise ValueError(f"unknown task: {task!r} (expected 'sr' or 'denoise')")
    return task


def apply_arg_overrides(config: dict, args):
    """Apply CLI overrides after loading config from a file or checkpoint."""
    if args.model: config['model']['name'] = args.model
    if args.task: config['task'] = args.task
    if args.scale: config['model']['scale'] = args.scale
    if args.epochs: config['train']['epochs'] = args.epochs
    if args.batch_size: config['train']['batch_size'] = args.batch_size
    if args.lr: config['train']['lr'] = args.lr
    return config


def load_config(args):
    # 1. If Resuming, load config from the checkpoint directory
    if args.resume and os.path.exists(args.resume):
        exp_dir = os.path.dirname(args.resume)
        train_cfg_path = os.path.join(exp_dir, "train_config.yaml")
        data_cfg_path = os.path.join(exp_dir, "data_config.yaml")
        
        if os.path.exists(train_cfg_path):
            print(f"Resuming: Loading train config from {train_cfg_path}")
            with open(train_cfg_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Cannot find train_config.yaml in {exp_dir}")
            
        # Without data_config.yaml the copy embedded in train_config.yaml stays.
        if os.path.exists(data_cfg_path):
            print(f"Resuming: Loading data config from {data_cfg_path}")
            with open(data_cfg_path, "r") as f:
                data_config = yaml.safe_load(f)
            config['data_config'] = data_config

        return apply_arg_overrides(config, args)

    # 2. New Training
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Load Data Config
    data_cfg_path = args.data_config or config.get('data_config_path') or infer_default_data_config_path(config)
        
    if data_cfg_path and not os.path.exists(data_cfg_path):
        # A named-but-missing recipe used to fall through to {}, and an empty
        # degradation config is not "no degradation": most effects default to
        # enabled: true, so the run silently trains on the full Real-ESRGAN
        # stack instead of the resize-only Phase-1 recipe. A typo in
        # --data_config or in data_config_path would cost the whole pretrain
        # and look like a modelling result.
        raise FileNotFoundError(
            f"data_config 를 찾을 수 없습니다: {data_cfg_path}\n"
            f"  (빈 config 는 '열화 없음'이 아니라 '기본값 전부 켜짐'을 뜻하므로 "
            f"조용히 넘어가지 않습니다)"
        )
    if data_cfg_path:
        with open(data_cfg_path, 'r') as f:
            data_config = yaml.safe_load(f)
        config['data_config'] = data_config
        config['data_config_path'] = data_cfg_path # Store original path for copying
    else:
        config['data_config'] = {}
        config['data_config_path'] = None
    
    return apply_arg_overrides(config, args)

def set_seed(seed):
    """Seed python, numpy and torch. DataLoader workers are re-seeded per worker
    by PyTorch, so augmentation stays diverse across them (verified)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = get_args()
    config = load_config(args)

    # cudnn.deterministic is deliberately left alone: it costs throughput and the
    # remaining nondeterminism is in kernel reduction order, not in the data.
    seed = args.seed if args.seed is not None else config['train'].get('seed', 42)
    set_seed(seed)
    print(f"Seed: {seed}")
    
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device.isdigit():
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device}")
        else:
            print("Warning: CUDA not available, falling back to CPU.")
            device = torch.device('cpu')
    else:
        # User passed 'cuda' or 'cuda:0'
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU.")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
    print(f"Using device: {device}")
    
    # load_config() already read train_config.yaml out of the checkpoint's own
    # directory when resuming, and the config embedded in the .pth is not
    # reloaded, so editing train_config.yaml applies on resume.
    # Note the learning rate is not among the things an edit can change: it is
    # restored from optimizer_state_dict, not from the config.

    # 1. Dataset & Dataloader
    dataset_kind = resolve_dataset_kind(config)
    dataset_type = config.get('data_config', {}).get('dataset_type', 'unset')
    print(f"Loading datasets for task={config['task']} dataset_type={dataset_type} resolved={dataset_kind}...")
    if dataset_kind == 'sr':
        train_ds = SRDataset(
            dataset_root=config['data']['train_root'],
            scale_factor=config['model']['scale'],
            patch_size=config['train']['patch_size'],
            is_train=True,
            config=config.get('data_config', {})
        )
    elif dataset_kind == 'denoise':
        if config['model'].get('scale', 1) != 1:
            raise ValueError(
                f"task=denoise expects model.scale=1, got {config['model'].get('scale')}: "
                f"the dataset yields same-resolution pairs, so a scaling model can only "
                f"shape-mismatch or silently broadcast."
            )
        train_ds = DenoiseDataset(
            dataset_root=config['data']['train_root'],
            scale_factor=1,
            patch_size=config['train']['patch_size'],
            is_train=True,
            config=config.get('data_config', {})
        )
    else:
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")
        
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        # The HR images are full-size LSDIR frames on a network mount and every
        # __getitem__ decodes one to take a 256px crop, so the loader, not the
        # GPU, sets the pace: at 4 workers the cards sat at 0-7% utilisation.
        # Keeping the workers alive across epochs avoids re-forking them every
        # 6,182 steps, and a deeper prefetch covers the mount's latency spikes.
        persistent_workers=config['train']['num_workers'] > 0,
        prefetch_factor=config['train'].get('prefetch_factor', 4)
        if config['train']['num_workers'] > 0 else None
    )
    
    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    if val_hr and val_lr:
        val_ds = PairedDataset(val_hr, val_lr)
        val_loader = DataLoader(val_ds,
                                batch_size=config['train'].get('val_batch_size', 1),
                                shuffle=False,
                                num_workers=config['train'].get('val_num_workers', 0))
    elif val_hr or val_lr:
        raise ValueError(
            f"val_hr_root and val_lr_root must be set together "
            f"(val_hr_root={val_hr!r}, val_lr_root={val_lr!r}); "
            f"a half-configured pair is almost always a typo."
        )
    elif config['train'].get('allow_no_validation', False):
        val_loader = None
        print("[WARNING] Training WITHOUT validation (train.allow_no_validation).")
        print("[WARNING]   - best.pth will never be written (no metric to beat)")
        print("[WARNING]   - early stopping will never fire")
    else:
        raise ValueError(
            "No validation set configured (data.val_hr_root / data.val_lr_root). "
            "Set both paths, or set train.allow_no_validation: true to accept "
            "training without best-checkpoint selection and early stopping."
        )

    # 2. Model
    print(f"Building model: {config['model']['name']}...")
    model = build_model(config['model']).to(device)
    
    # 2.1 Load Pretrained Weights (Finetuning)
    loaded_pretrained = False
    if not args.resume and config['train'].get('pretrained_path'):
        pretrained_path = config['train']['pretrained_path']
        if os.path.exists(pretrained_path):
            print(f"Finetuning: Loading pretrained weights from {pretrained_path}...")
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            # strict=False allows partial weights, but it also returns quietly when
            # nothing matched at all -- a wrong path or another model's checkpoint
            # printed "Loading pretrained weights" and then trained from scratch.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            matched = len(model.state_dict()) - len(missing)
            print(f"[Finetune] matched={matched} missing={len(missing)} unexpected={len(unexpected)}")
            if matched == 0:
                raise RuntimeError(
                    f"No weights matched from {pretrained_path}. "
                    f"The checkpoint holds {len(state_dict)} tensors, none of which name a "
                    f"parameter of {config['model']['name']}."
                )
            loaded_pretrained = True
        else:
            # A named-but-missing checkpoint used to print a warning and train
            # from random init. A finetune config always means to start from
            # something specific, so a path that does not resolve is a mistake,
            # not a fallback -- and the failure is invisible unless someone
            # reads the first lines of a log that then runs for days. This bit:
            # a bulk path rewrite prefixed the Phase-2 configs with an archive
            # directory, and all three would have silently discarded a two-day
            # pretrain.
            raise FileNotFoundError(
                f"pretrained_path 를 찾을 수 없습니다: {pretrained_path}\n"
                f"  (파인튜닝은 출발점이 반드시 있어야 하므로 랜덤 초기화로 "
                f"조용히 넘어가지 않습니다. 처음부터 학습하려면 config 에서 "
                f"pretrained_path 를 지우세요)"
            )
    
    # 2.5 Setup Work Directory & Save Configs
    if args.resume:
        # The run's own directory: train_config.yaml/data_config.yaml live there and
        # are the resume source of truth (get_args rejects --work_dir on a resume).
        work_dir = os.path.dirname(args.resume)
    else:
        if args.work_dir:
            work_dir = args.work_dir
        elif config['train'].get('save_name'):
             # Use custom save name if provided in config
             work_dir = os.path.join("checkpoints", config['train']['save_name'])
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{config['model']['name']}_{config['task']}_x{config['model']['scale']}_{timestamp}"
            work_dir = os.path.join("checkpoints", exp_name)
    
    os.makedirs(work_dir, exist_ok=True)
    
    # Snapshot the effective config (CLI overrides applied). New runs write
    # train_config.yaml / data_config.yaml -- exactly the filenames
    # load_config() reads back on resume, so keep these names and keep the
    # content safe_load-compatible. Resumed runs must NOT overwrite them
    # (they are the resume source of truth); a per-resume history file is
    # written after trainer.resume() below instead.
    if not args.resume:
        # Verbatim copies preserve comments and key order for humans; the
        # dumps below are the machine-readable truth with overrides applied.
        shutil.copy(args.config, os.path.join(work_dir, "train_config.orig.yaml"))
        if config.get('data_config_path'):
            shutil.copy(config['data_config_path'], os.path.join(work_dir, "data_config.orig.yaml"))
        with open(os.path.join(work_dir, "train_config.yaml"), 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        if config.get('data_config'):
            with open(os.path.join(work_dir, "data_config.yaml"), 'w', encoding='utf-8') as f:
                yaml.safe_dump(config['data_config'], f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # 3. Trainer Setup
    print(f"Work Directory: {work_dir}")
    log_dir = os.path.join(work_dir, "logs")
    writer = SummaryWriter(log_dir)
    
    trainer = Trainer(model, config, device, writer, checkpoint_dir=work_dir)
    # Finetuning starts from a model that already has a score; make best.pth beat it.
    trainer.validate_before_fit = loaded_pretrained
    
    # Resume if requested
    if args.resume:
        trainer.resume(args.resume)
        # Record the config this resume was REQUESTED with as a history file.
        # Note the learning rate in it may not be what actually runs: resume()
        # restores optimizer/scheduler state, so --lr on a resume is
        # overwritten by the checkpoint's schedule. train_config.yaml itself is
        # left untouched: load_config() reads it as the resume source of truth,
        # and overwriting it here would make a one-off override sticky.
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        resume_snap = os.path.join(
            work_dir, f"train_config.resume_ep{trainer.start_epoch}_{stamp}.yaml")
        with open(resume_snap, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    # 4. Fit
    try:
        trainer.fit(train_loader, val_loader, epochs=config['train']['epochs'])
    finally:
        writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
