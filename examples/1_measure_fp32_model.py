
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.data.datasets import PairedDataset
from src.aimet.utils import evaluate_model

# python3 examples/1_measure_fp32_model.py --config configs/train/quicksrnet.yaml 
# --checkpoint checkpoints/train_quicksrnet_large_sr_x2_dim64_epoch800_bs_32_lr_1e-4/best.pth 
# --device 0 --data_config configs/data/sr.yaml

def get_args():
    parser = argparse.ArgumentParser(description="Measure FP32 Model Performance")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file")
    parser.add_argument("--data_config", type=str, default="configs/data/sr.yaml", help="Data config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Device Setup
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device.isdigit():
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device}")
        else:
            print("Warning: CUDA not available, falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # 3. Build Model
    print(f"Loading model: {config['model']['name']}...")
    model = build_model(config['model']).to(device)
    
    # 4. Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if hasattr(model, 'switch_to_deploy'):
        print("Switching model to deploy mode...")
        model.switch_to_deploy()

    # 5. Prepare Validation Loader
    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        # 6. Evaluate
        evaluate_model(model, val_loader, device, title="FP32 Baseline")
    else:
        print("Warning: validation paths not found in config ('val_hr_root', 'val_lr_root'). Cannot evaluate.")

    # 7. Print Module Names (for Ignore Layers)
    print("\n" + "="*60)
    print(f"{'Module Name':<40} | {'Type':<20}")
    print("="*60)
    for name, module in model.named_modules():
        # Print leaf modules (no children)
        if len(list(module.children())) == 0:
            print(f"{name:<40} | {module.__class__.__name__}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
