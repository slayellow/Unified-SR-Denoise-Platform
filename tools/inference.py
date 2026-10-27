import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import argparse
import glob
import time
import yaml
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.checkpoint import load_model

def get_args():
    parser = argparse.ArgumentParser(description="Unified SR/Denoise Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="results/inference", help="Path to save results")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    return parser.parse_args()

def load_image(path):
    # Read image using OpenCV (BGR)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # HWC -> CHW
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def save_image(tensor, path):
    # CHW -> HWC
    img = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Clamp and Denormalize
    img = np.clip(img, 0, 1)
    img = (img * 255.0).round().astype(np.uint8)
    
    # RGB -> BGR for OpenCV saving
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

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
    
    # Model in deploy mode, the graph that ships
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, config['model'], device)

    if args.fp16 and device.type == 'cuda':
        print("Using FP16 precision")
        model.half()

    # Input Files
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*")))
        # Filter images
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    else:
        files = [args.input]

    print(f"Found {len(files)} files to process.")
    
    # Inference Loop
    os.makedirs(args.output, exist_ok=True)
    
    total_time = 0
    img_count = 0
    with torch.no_grad():
        for fpath in tqdm(files, desc="Total Files"):
            # Load
            img_in = load_image(fpath).to(device)
            if args.fp16 and device.type == 'cuda':
                img_in = img_in.half()

            # Inference
            start = time.time()
            output = model(img_in)
            if isinstance(output, tuple):
                output = output[0]

            # End Time
            # Synchronize for accurate timing if cuda
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)
            img_count += 1

            # Save
            filename = os.path.basename(fpath)
            save_path = os.path.join(args.output, filename)
            save_image(output, save_path)
            
    avg_time = (total_time / img_count) * 1000 if img_count > 0 else 0
    print(f"\nInference finished.")
    if img_count > 0:
        print(f"Average Inference Time (Images): {avg_time:.2f} ms / image")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
