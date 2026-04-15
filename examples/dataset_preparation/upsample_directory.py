import os
import argparse
import cv2
import glob
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Batch Bicubic Upsampling Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save upsampled images")
    parser.add_argument("--scale", type=int, default=2, help="Upsampling scale factor")
    return parser.parse_args()

def main():
    args = get_args()

    # Input Check
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return

    # Image Extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)])
    
    if not files:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(files)} images to process.")
    print(f"Upsampling Mode: Bicubic, Scale: x{args.scale}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Processing
    for fpath in tqdm(files):
        try:
            # Read Image
            img = cv2.imread(fpath)
            if img is None:
                print(f"Warning: Could not read {fpath}")
                continue
            
            # Dimensions
            h, w = img.shape[:2]
            new_dim = (w * args.scale, h * args.scale)
            
            # Upsample
            upsampled = cv2.resize(img, new_dim, interpolation=cv2.INTER_CUBIC)
            
            # Save
            filename = os.path.basename(fpath)
            save_path = os.path.join(args.output_dir, filename)
            cv2.imwrite(save_path, upsampled)
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    print(f"Done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
