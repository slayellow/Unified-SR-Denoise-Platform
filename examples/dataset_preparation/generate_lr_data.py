import os
import sys
import argparse
import glob
import cv2
import yaml
import numpy as np
import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.datasets import SRDataset

def get_args():
    parser = argparse.ArgumentParser(description="Generate Validation Dataset (HR, LRx2, LRx4)")
    parser.add_argument("--input", type=str, required=True, help="Path to input HR images")
    parser.add_argument("--output", type=str, required=True, help="Output root directory")
    parser.add_argument("--config", type=str, default="configs/data/sr.yaml", help="Path to degradation config")
    parser.add_argument("--width", type=int, default=1280, help="Target HR width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Target HR height (default: 720)")
    return parser.parse_args()

def center_crop(img, size):
    """
    Center crop image to target size (width, height).
    If image is smaller than target, resize it so it covers target size.
    """
    target_w, target_h = size
    h, w = img.shape[:2]

    # If smaller, resize preserving aspect ratio to cover target
    if w < target_w or h < target_h:
        scale = max(target_w / w, target_h / h)
        # Use bicubic for upscaling
        new_w, new_h = int(np.ceil(w * scale)), int(np.ceil(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    # Center Crop
    start_x = (w - target_w) // 2
    start_y = (h - target_h) // 2
    return img[start_y:start_y+target_h, start_x:start_x+target_w]

def main():
    args = get_args()
    
    # 1. Load Config
    print(f"Loading config: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Initialize Datasets (Dummy root, just need pipeline)
    print("Initializing Degradation Pipelines...")
    ds_x2 = SRDataset(dataset_root=[], scale_factor=2, config=config)
    ds_x4 = SRDataset(dataset_root=[], scale_factor=4, config=config)
    ds_x8 = SRDataset(dataset_root=[], scale_factor=8, config=config)
    
    # 3. Prepare Output Directories
    out_hr = os.path.join(args.output, "HR")
    out_lrx2 = os.path.join(args.output, "LRx2")
    out_lrx4 = os.path.join(args.output, "LRx4")
    out_lrx8 = os.path.join(args.output, "LRx8")
    
    os.makedirs(out_hr, exist_ok=True)
    os.makedirs(out_lrx2, exist_ok=True)
    os.makedirs(out_lrx4, exist_ok=True)
    os.makedirs(out_lrx8, exist_ok=True)
    
    # 4. Process Images
    # Supports png, jpg, jpeg
    exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    image_paths = sorted(list(set(image_paths)))
    
    print(f"Found {len(image_paths)} images in {args.input}")
    
    target_size = (args.width, args.height) # (W, H)
    
    valid_idx = 0
    for path in tqdm.tqdm(image_paths):
        # Read HR
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping corrupt image: {path}")
            continue
            
        out_name = f"{valid_idx:04d}.png" # Save as PNG for lossless
        valid_idx += 1
            
        # Center Crop to Target HR Size
        hr_img = center_crop(img, target_size)
        
        # Save HR
        cv2.imwrite(os.path.join(out_hr, out_name), hr_img)
        
        lr_x2 = ds_x2.degradation_pipeline(hr_img)
        cv2.imwrite(os.path.join(out_lrx2, out_name), lr_x2)
        
        # Generate LRx4
        lr_x4 = ds_x4.degradation_pipeline(hr_img)
        cv2.imwrite(os.path.join(out_lrx4, out_name), lr_x4)

        lr_x8 = ds_x8.degradation_pipeline(hr_img)
        cv2.imwrite(os.path.join(out_lrx8, out_name), lr_x8)
        
    print("Done!")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
