import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyiqa

from src.metrics import METRIC_SPEC, create_lpips, create_niqe, create_psnr, create_ssim

from src.engine.checkpoint import load_model
from src.data.datasets import PairedDataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Unified Model Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Config file used for training (to load model params)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--hr_dir", type=str, default=None, help="Path to HR images (Optional, defaults to config val_hr_root)")
    parser.add_argument("--lr_dir", type=str, default=None, help="Path to LR images (Optional, defaults to config val_lr_root)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, required=True, help="Base directory for metrics.csv and output images (<save_dir>/<checkpoint name>/)")
    parser.add_argument("--save_images", action="store_true", help="If set, save the reconstructed images")
    return parser.parse_args()

def calculate_metrics(img1, img2, metrics, device):
    """
    img1, img2: Numpy (H, W, C) in [0, 1] RGB. img1 is HR, img2 is SR.

    PSNR and SSIM come from src.metrics so this table can be compared with the
    trainer's validation log and with published numbers. The previous local
    definition read the luminance channel through cv2's full-range YCrCb, which
    on the x2 validation set sat 1.3 dB below the SR convention.
    """
    t_hr = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
    t_sr = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)

    psnr_val = metrics['psnr'](t_sr, t_hr).item()
    ssim_val = metrics['ssim'](t_sr, t_hr).item()
    lpips_val = metrics['lpips'](t_sr, t_hr).item()
    # NIQE is no-reference, so it only sees the reconstruction.
    niqe_val = metrics['niqe'](t_sr).item()

    return psnr_val, ssim_val, lpips_val, niqe_val

def main():
    args = get_args()
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
    
    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Resolve Data Paths (Prioritize CLI -> Config)
    hr_dir = args.hr_dir if args.hr_dir else config['data'].get('val_hr_root')
    lr_dir = args.lr_dir if args.lr_dir else config['data'].get('val_lr_root')

    if not hr_dir or not lr_dir:
        raise ValueError("Dataset paths must be provided via CLI arguments or Config file.")
    

    # 2. Resolve Save Directory
    # Create a subfolder based on checkpoint name for better organization
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_path = os.path.join(args.save_dir, ckpt_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Evaluation Settings:")
    print(f" - Model: {config['model']['name']}")
    print(f" - Checkpoint: {args.checkpoint}")
    print(f" - HR Dir: {hr_dir}")
    print(f" - LR Dir: {lr_dir}")
    print(f" - Save Dir: {save_path}")
    print(f" - Save Images: {args.save_images}")

    # Model in deploy mode, the graph that ships
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, config['model'], device)

    # Dataset
    dataset = PairedDataset(hr_dir, lr_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Metrics -- one definition, shared with the trainer (see src/metrics).
    scale = int(config.get('model', {}).get('scale', 1) or 1)
    print(f"Loading metrics: {METRIC_SPEC}, scale={scale}")
    metrics = {
        'psnr': create_psnr(scale, device),
        'ssim': create_ssim(scale, device),
        'lpips': create_lpips(device),
        'niqe': create_niqe(scale, device),
    }
    
    # Results
    results = []
    
    print(f"Index | PSNR | SSIM | LPIPS | NIQE | Filename")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            path = batch['path'][0]
            filename = os.path.basename(path)
            
            # Inference
            sr = model(lr)
            if isinstance(sr, tuple): sr = sr[0]

            # x3 of a 1280-wide frame is 1278 wide. Crop both to the common size, as the
            # trainer's Validator does, so the table matches the validation log.
            h, w = min(sr.shape[-2], hr.shape[-2]), min(sr.shape[-1], hr.shape[-1])
            sr, hr = sr[..., :h, :w], hr[..., :h, :w]
            
            # Post-process
            sr = torch.clamp(sr, 0.0, 1.0)
            
            # Convert to Numpy (H, W, C) RGB
            sr_np = sr.cpu().squeeze(0).permute(1, 2, 0).numpy()
            hr_np = hr.cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            # Metrics
            p, s, l, n = calculate_metrics(hr_np, sr_np, metrics, device)
            
            results.append({
                'filename': filename,
                'PSNR': p,
                'SSIM': s,
                'LPIPS': l,
                'NIQE': n
            })
            
            # Save Image (Optional)
            if args.save_images:
                out_path = os.path.join(save_path, filename)
                # RGB -> BGR for opencv
                sr_bgr = cv2.cvtColor((sr_np * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, sr_bgr)

    df = pd.DataFrame(results)
    print("\n=== Evaluation Summary ===")
    print(df.mean(numeric_only=True))
    
    csv_path = os.path.join(save_path, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

if __name__ == "__main__":
    main()
