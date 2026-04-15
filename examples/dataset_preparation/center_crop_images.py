import os
import glob
import cv2
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Center Crop Images in a Directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory to save cropped images")
    parser.add_argument("--width", type=int, required=True, help="Target crop width")
    parser.add_argument("--height", type=int, required=True, help="Target crop height")
    return parser.parse_args()

def center_crop(img, crop_h, crop_w):
    h, w = img.shape[:2]
    
    # Check if image is smaller than target crop size
    if h < crop_h or w < crop_w:
        print(f"Warning: Image size ({w}x{h}) is smaller than target crop size ({crop_w}x{crop_h}). Resizing...")
        # Scale image to match the minimum side needed while preserving aspect ratio
        scale = max(crop_h / h, crop_w / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find images
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper()))) # Handle .PNG, .JPG, etc.

    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    for path in tqdm(image_paths, desc="Cropping"):
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to read {path}. Skipping.")
            continue

        cropped_img = center_crop(img, args.height, args.width)

        filename = os.path.basename(path)
        out_path = os.path.join(args.output_dir, filename)
        
        # Save keeping original extension
        cv2.imwrite(out_path, cropped_img)

    print(f"\nDone! Cropped images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
