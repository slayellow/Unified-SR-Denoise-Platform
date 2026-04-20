import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import datetime
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Unified Platform Imports
from src.models import build_model
from src.data.datasets import SRDataset, DenoiseDataset, GuidedSRDataset, PairedDataset
from src.engine.trainer import Trainer
from src.engine.gan_trainer import GANTrainer

def get_args():
    parser = argparse.ArgumentParser(description="Unified SR/Denoise Training")
    parser.add_argument("--config", type=str, required=True, help="Path to main train config file")
    parser.add_argument("--data_config", type=str, help="Path to data config file (override default)")
    parser.add_argument("--model", type=str, help="Model name (override config)")
    parser.add_argument("--task", type=str, choices=['sr', 'denoise', 'guide'], help="Task type (override config)")
    parser.add_argument("--scale", type=int, help="Scale factor (override config)")
    parser.add_argument("--work_dir", type=str, help="Directory to save checkpoints and logs")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from (e.g. checkpoints/exp/last.pth)")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    return parser.parse_args()

def infer_default_data_config_path(config: dict, train_config_path: str) -> str | None:
    """Infer the default data-config path from task and train-config location."""
    task = config.get('task')
    normalized_path = os.path.normpath(train_config_path).lower()

    if task == 'guide':
        return "configs/data/gsr_train.yaml"
    if task == 'denoise':
        return "configs/data/denoise.yaml"
    if task == 'sr' and f"{os.sep}denoise{os.sep}" in normalized_path:
        return "configs/data/denoise.yaml"
    if task == 'sr':
        return "configs/data/sr_train.yaml"
    return None


def resolve_dataset_kind(config: dict) -> str:
    """Resolve which dataset class to instantiate from task and data_config."""
    task = str(config.get('task', '')).lower()
    dataset_type = str(config.get('data_config', {}).get('dataset_type', '')).lower()

    if task == 'guide':
        return 'guide'
    if dataset_type in {'denoise', 'dn'}:
        return 'denoise'
    if dataset_type in {'guide', 'guided_sr'}:
        return 'guide'
    if dataset_type in {'sr', 'super_resolution'}:
        return 'sr'
    if task in {'denoise', 'guide', 'sr'}:
        return task
    return 'sr'


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
            
        if os.path.exists(data_cfg_path):
            print(f"Resuming: Loading data config from {data_cfg_path}")
            with open(data_cfg_path, "r") as f:
                data_config = yaml.safe_load(f)
            config['data_config'] = data_config
        else:
            config['data_config'] = {}
            
        return config

    # 2. New Training
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Load Data Config
    if args.data_config:
        data_cfg_path = args.data_config
    else:
        data_cfg_path = infer_default_data_config_path(config, args.config)
        
    if data_cfg_path and os.path.exists(data_cfg_path):
        with open(data_cfg_path, 'r') as f:
            data_config = yaml.safe_load(f)
        config['data_config'] = data_config
        config['data_config_path'] = data_cfg_path # Store original path for copying
    else:
        config['data_config'] = {}
        config['data_config_path'] = None
    
    # Args override
    if args.model: config['model']['name'] = args.model
    if args.task: config['task'] = args.task
    if args.scale: config['model']['scale'] = args.scale
    if args.epochs: config['train']['epochs'] = args.epochs
    if args.batch_size: config['train']['batch_size'] = args.batch_size
    if args.lr: config['train']['lr'] = args.lr
    
    return config

def main():
    args = get_args()
    config = load_config(args)
    
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
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
    print(f"Using device: {device}")
    
    # Check resume to override config if needed (User wanted to use config from checkpoint)
    # However, to initialize dataset we need config first...
    # Strategy: Initialize with current config, then Trainer.resume() will update Trainer.cfg 
    # But Datasets need to be consistent. 
    # Better: If resume, pre-load config needed for Datasets?
    # User said: "read config files inside it".
    # So we should try to load checkpoint config first if possible or rely on Trainer resume.
    # Actually, Dataset path/type is structural.
    # Let's trust that the resume file corresponds to the same Task.
    
    if args.resume and os.path.exists(args.resume):
        print(f"Pre-loading config from {args.resume} for Dataset initialization...")
        ckpt = torch.load(args.resume, map_location='cpu') # Load to cpu just for config
        if 'config' in ckpt:
            config = ckpt['config']
            print("Loaded config from checkpoint.")
        else:
            print("Warning: No config found in checkpoint. Using command line/default config.")
    
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
            print(f"Warning: dataset_type=denoise but model.scale={config['model'].get('scale')} (expected 1).")
        train_ds = DenoiseDataset(
            dataset_root=config['data']['train_root'],
            scale_factor=1,
            patch_size=config['train']['patch_size'],
            is_train=True,
            config=config.get('data_config', {})
        )
    elif dataset_kind == 'guide':
        train_ds = GuidedSRDataset(
            hr_dirs=config['data']['train_hr_root'],
            lr_dirs=config['data'].get('train_lr_root'),
            guide_dirs=config['data']['train_guide_root'],
            patch_size=config['train']['patch_size'],
            scale_factor=config['model']['scale'],
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
        pin_memory=True
    )
    
    if config['task'] == 'guide' and config['data'].get('val_hr_root'):
        val_ds = GuidedSRDataset(
            hr_dirs=config['data']['val_hr_root'],
            lr_dirs=config['data']['val_lr_root'],
            guide_dirs=config['data']['val_guide_root'],
            patch_size=config['train']['patch_size'],
            scale_factor=config['model']['scale'],
            is_train=False,
            config=config.get('data_config', {})
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    elif config['data'].get('val_hr_root') and config['data'].get('val_lr_root'):
        val_ds = PairedDataset(config['data']['val_hr_root'], config['data']['val_lr_root'])
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    else:
        val_loader = None
        print("Validation set not configured.")

    # 2. Model
    print(f"Building model: {config['model']['name']}...")
    model = build_model(config['model']).to(device)
    
    # 2.1 Load Pretrained Weights (Finetuning)
    if not args.resume and config['train'].get('pretrained_path'):
        pretrained_path = config['train']['pretrained_path']
        if os.path.exists(pretrained_path):
            print(f"Finetuning: Loading pretrained weights from {pretrained_path}...")
            checkpoint = torch.load(pretrained_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            # Use strict=False to allow for flexible loading (e.g. partial weights)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Pretrained path {pretrained_path} not found. Starting from scratch.")
    
    # 2.5 Setup Work Directory & Save Configs
    if args.resume:
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
    
    # Copy configs only if starting new (or ensure they exist)
    if not args.resume:
        import shutil
        # Save explicit train config
        with open(os.path.join(work_dir, "train_config.yaml"), 'w') as f:
            yaml.dump(config, f)
        
        # Save data config separately for clarity (or just saving merged is fine, but separating is cleaner)
        if config.get('data_config'):
             with open(os.path.join(work_dir, "data_config.yaml"), 'w') as f:
                yaml.dump(config['data_config'], f)

    # 3. Trainer Setup
    print(f"Work Directory: {work_dir}")
    log_dir = os.path.join(work_dir, "logs")
    writer = SummaryWriter(log_dir)
    
    # Use GANTrainer if discriminator config is present (e.g. LapGSR)
    if config.get('discriminator'):
        print(f"Building discriminator: {config['discriminator']['name']}...")
        discriminator = build_model(config['discriminator']).to(device)
        trainer = GANTrainer(model, discriminator, config, device, writer, checkpoint_dir=work_dir)
    else:
        trainer = Trainer(model, config, device, writer, checkpoint_dir=work_dir)
    
    # Resume if requested
    if args.resume:
        trainer.resume(args.resume)
    
    # 4. Fit
    trainer.fit(train_loader, val_loader, epochs=config['train']['epochs'])

    # 5. Export
    if config.get('export_onnx', False):
        trainer.export_onnx()

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
