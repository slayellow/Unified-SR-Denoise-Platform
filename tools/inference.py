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

from src.models import build_model

def get_args():
    parser = argparse.ArgumentParser(description="Unified SR/Denoise Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="results/inference", help="Path to save results")
    parser.add_argument("--guide", type=str, default=None, help="Path to guide image or directory (Optional)")
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

def process_video(fpath, model, device, args, scale):
    cap = cv2.VideoCapture(fpath)
    if not cap.isOpened():
        print(f"Failed to open video: {fpath}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 30.0 # Default fallback if reading fails
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if scale == 2:
        target_w, target_h = 640, 360
    elif scale == 4:
        target_w, target_h = 320, 180
    else:
        target_w, target_h = width, height
        
    start_y = max(0, (height - target_h) // 2)
    start_x = max(0, (width - target_w) // 2)
    
    # Fallback to an MP4 backend explicitly
    out_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(fpath))[0]}_output.mp4")
    out_w, out_h = target_w * scale, target_h * scale
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_path = out_path.replace('.mp4', '.avi')
        
    out = cv2.VideoWriter(out_path, cv2.CAP_FFMPEG, fourcc, fps, (int(out_w), int(out_h)), True)
    
    if not out.isOpened():
        print(f"Warning: Failed to initialize VideoWriter with FFMPEG backing. Trying default.")
        out = cv2.VideoWriter(out_path, fourcc, fps, (int(out_w), int(out_h)), True)
    
    print(f"Processing video: {fpath}")
    print(f"Crop: {target_w}x{target_h} -> Output: {out_w}x{out_h} (Expected)")
    
    pbar = tqdm(total=total_frames, desc="Video Frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cropped = frame[start_y:start_y+target_h, start_x:start_x+target_w]
        
        img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        if args.fp16 and device.type == 'cuda':
            img = img.half()
            
        output = model(img)
        if isinstance(output, tuple):
            output = output[0]
            
        out_img = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        out_img = np.clip(out_img, 0, 1)
        out_img = (out_img * 255.0).round().astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        
        # Ensure the output frame size matches exactly (OpenCV is very strict)
        if out_img.shape[1] != int(out_w) or out_img.shape[0] != int(out_h):
            out_img = cv2.resize(out_img, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)
            
        success = out.write(out_img)
        if not success and pbar.n < 5:
             # Just debug on first few frames
             pass
        pbar.update(1)
        
    pbar.close()
    cap.release()
    out.release()

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
    
    # Build Model
    print(f"Loading model: {config['model']['name']}...")
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

    if hasattr(model, 'switch_to_deploy'):
        print("Switching model to deploy mode (merging branches)...")
        model.switch_to_deploy()
    
    model.eval()

    if args.fp16 and device.type == 'cuda':
        print("Using FP16 precision")
        model.half()

    # Input Files
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*")))
        # Filter images and videos
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.mp4', '.avi', '.mkv', '.mov', '.ts'))]
    else:
        files = [args.input]
        
    task = config.get('task', 'sr')
    scale = config['model']['scale']
    if task == 'guide' and not args.guide:
        print("[Info] No --guide provided for 'guide' task. Running without guide (self-guide mode).")
        
    print(f"Found {len(files)} files to process.")
    
    # Inference Loop
    os.makedirs(args.output, exist_ok=True)
    
    total_time = 0
    img_count = 0
    with torch.no_grad():
        for fpath in tqdm(files, desc="Total Files"):
            ext = os.path.splitext(fpath)[-1].lower()
            if ext in ['.mp4', '.avi', '.mkv', '.mov', '.ts']:
                process_video(fpath, model, device, args, scale)
            else:
                # Load
                img_in = load_image(fpath).to(device)
                if args.fp16 and device.type == 'cuda':
                    img_in = img_in.half()
                
                # Load Guide if needed
                guide_in = None
                if task == 'guide':
                    filename = os.path.basename(fpath)
                    if os.path.isdir(args.guide):
                        guide_path = os.path.join(args.guide, filename)
                    else:
                        guide_path = args.guide # Single file case
                    
                    if not os.path.exists(guide_path):
                        print(f"[Warning] Guide image not found for {filename}: {guide_path}. Skipping.")
                        continue
                        
                    guide_in = load_image(guide_path).to(device)
                    if args.fp16 and device.type == 'cuda':
                        guide_in = guide_in.half()
                
                # Inference
                start = time.time()
                if task == 'guide' and guide_in is not None:
                    output = model(img_in, guide_in)
                elif task == 'guide':
                    # guide task이지만 guide 없음 → self-guide mode
                    output = model(img_in, None)
                else:
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
