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
import lpips
import pyiqa
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from src.models import build_model
from src.data.datasets import PairedDataset, GuidedPairedDataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Unified Model Evaluation")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file used for training (to load model params)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--hr_dir", type=str, default=None, help="Path to HR images (Optional, defaults to config val_hr_root)")
    parser.add_argument("--lr_dir", type=str, default=None, help="Path to LR images (Optional, defaults to config val_lr_root)")
    parser.add_argument("--guide_dir", type=str, default=None, help="Path to Guide images (Optional, defaults to config val_guide_root)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results", help="Base directory to save output images")
    parser.add_argument("--save_images", action="store_true", help="If set, save the reconstructed images")
    return parser.parse_args()

def calculate_metrics(img1, img2, lpips_fn, niqe_fn, device):
    """
    img1, img2: Numpy (H, W, C) in [0, 1] RGB
    """

    # 1. [0, 1] 범위의 RGB를 [0, 255] 범위의 BGR로 변환하여 OpenCV와 호환되게 맞춤
    img1_bgr = cv2.cvtColor((img1 * 255).astype(np.float32), cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor((img2 * 255).astype(np.float32), cv2.COLOR_RGB2BGR)
    
    # 2. BGR을 YCrCb로 변환하고 Y 채널만 추출
    hr_y = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    sr_y = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    
    # 3. Y 채널에 대해 PSNR / SSIM 계산 (data_range는 255)
    # 흑백(2D) 이미지이므로 channel_axis를 쓰지 않습니다.
    psnr_val = psnr(hr_y, sr_y, data_range=255.0)
    ssim_val = ssim(hr_y, sr_y, data_range=255.0)

    # # PSNR / SSIM
    # # skimage expects [0, 1] or [0, 255] with data_range specified
    # psnr_val = psnr(img1, img2, data_range=1.0)
    # ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=2)
    
    # Prepare Tensors for LPIPS/NIQE
    # img1 = HR (Reference), img2 = SR (Distorted)
    
    # LPIPS: Expects Float Tensor (N, 3, H, W) in [-1, 1]
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    lpips_val = lpips_fn(t1, t2).item()

    # NIQE: Expects Float Tensor (N, 3, H, W) in [0, 1]
    # NIQE is No-Reference, so we check the SR image (t2_01)
    t2_01 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)
    niqe_val = niqe_fn(t2_01).item()
    
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
    guide_dir = args.guide_dir if args.guide_dir else config['data'].get('val_guide_root')

    if not hr_dir or not lr_dir:
        raise ValueError("Dataset paths must be provided via CLI arguments or Config file.")
    
    task = config.get('task', 'sr')
    if task == 'guide' and not guide_dir:
        print("[Info] No guide directory provided. Running without guide (self-guide mode).")

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
    if task == 'guide':
        print(f" - Guide Dir: {guide_dir}")
    print(f" - Save Dir: {save_path}")
    print(f" - Save Images: {args.save_images}")

    # Build Model
    print(f"Loading model...")
    model = build_model(config['model']).to(device)
    
    # Load Weights
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, torch.nn.Module):
        print("[Info] Loaded full model object.")
        model = checkpoint.to(device)
    else:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)

    model.eval()
    
    # Switch to deploy if possible (for faster inference/realistic evaluation)
    if hasattr(model, 'switch_to_deploy'):
        print("Switching model to deploy mode...")
        model.switch_to_deploy()

    # Dataset
    if task == 'guide' and guide_dir:
        dataset = GuidedPairedDataset(hr_dir, lr_dir, guide_dir)
    else:
        dataset = PairedDataset(hr_dir, lr_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Metrics
    print("Loading metrics...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    niqe_fn = pyiqa.create_metric('niqe', device=device)
    
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
            if task == 'guide' and 'guide' in batch:
                guide = batch['guide'].to(device)
                sr = model(lr, guide)
            elif task == 'guide':
                # guide task이지만 guide 없음 → self-guide mode
                sr = model(lr, None)
            else:
                sr = model(lr)
            if isinstance(sr, tuple): sr = sr[0]
            
            # Post-process
            sr = torch.clamp(sr, 0.0, 1.0)
            
            # Convert to Numpy (H, W, C) RGB
            sr_np = sr.cpu().squeeze(0).permute(1, 2, 0).numpy()
            hr_np = hr.cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            # Metrics
            p, s, l, n = calculate_metrics(hr_np, sr_np, lpips_fn, niqe_fn, device)
            
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
                sr_bgr = cv2.cvtColor((sr_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, sr_bgr)
            
            # print(f"{filename}: PSNR={p:.2f} SSIM={s:.4f} LPIPS={l:.4f} NIQE={n:.4f}")

    df = pd.DataFrame(results)
    print("\n=== Evaluation Summary ===")
    print(df.mean(numeric_only=True))
    
    csv_path = os.path.join(save_path, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

if __name__ == "__main__":
    main()
