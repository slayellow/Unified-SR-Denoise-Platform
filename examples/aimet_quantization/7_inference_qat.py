import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import cv2
import glob
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import build_model
from src.aimet.utils import apply_mmp_from_json

# AIMET Imports
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_common.defs import QuantScheme
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model

def get_args():
    parser = argparse.ArgumentParser(description="QAT Model Inference Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (e.g. configs/finetune/quicksrnet_4x_ir.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to QAT best checkpoint (e.g. results/qat/qat_checkpoints/best_model.pth)")
    parser.add_argument("--input", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument("--output_dir", type=str, default="results/qat_inference", help="Directory to save output images")
    
    # Quantization Params
    parser.add_argument("--param_bw", type=int, default=8, help="Default Parameter Bitwidth")
    parser.add_argument("--output_bw", type=int, default=8, help="Default Output Bitwidth")
    parser.add_argument("--encodings", type=str, default=None, help="Path to pre-computed .encodings file (Optional)")
    parser.add_argument("--mmp_config", type=str, default=None, help="Path to MMP JSON config (Optional)")
    
    parser.add_argument("--base_model", type=str, default=None, help="Path to base model object (.pth), e.g., compressed_model.pth if pruned")
    parser.add_argument("--width", type=int, default=640, help="Width of dummy input (must match training patch/size ratio if static needed)")
    parser.add_argument("--height", type=int, default=360, help="Height of dummy input")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def apply_default_mmp(sim):
    print("[Info] Applying default hardcoded MMP logic...")
    high_mse_indices = [5, 6, 7, 11]
    sensitive_indices = [10]
    target_modules = []

    target_modules.append(("head", sim.model.head))

    for idx in high_mse_indices:
        prev_idx = idx - 1
        if prev_idx >= 0:
            try:
                prev_block = getattr(sim.model.body, str(prev_idx))
                target_modules.append((f"body.{prev_idx}.act", prev_block.act))
            except AttributeError:
                pass

    for idx in sensitive_indices:
        try:
            curr_block = getattr(sim.model.body, str(idx))
            target_modules.append((f"body.{idx}.act", curr_block.act))
        except AttributeError:
            pass

    if hasattr(sim.model, 'upsample'):
        if isinstance(sim.model.upsample, torch.nn.Conv2d) or \
           (hasattr(sim.model.upsample, 'output_quantizers') and sim.model.upsample.output_quantizers):
             target_modules.append(("upsample", sim.model.upsample))
        else:
            try:
                upsample_0 = getattr(sim.model.upsample, "0")
                target_modules.append(("upsample.0", upsample_0))
            except AttributeError:
                try:
                    upsample_0 = sim.model.upsample[0]
                    target_modules.append(("upsample.0", upsample_0))
                except:
                    pass
        
    if hasattr(sim.model, 'final_act'): 
        target_modules.append(("final_act", sim.model.final_act))

    if hasattr(sim.model, 'module_add'): 
        target_modules.append(("module_add", sim.model.module_add))

    unique_modules = {name: mod for name, mod in target_modules}
    for name, module in unique_modules.items():
        if hasattr(module, 'output_quantizers') and module.output_quantizers:
            if module.output_quantizers[0] is not None:
                module.output_quantizers[0].bitwidth = 8
                
        if name == "head":
            if hasattr(module, 'input_quantizers') and module.input_quantizers:
                if module.input_quantizers[0] is not None:
                    module.input_quantizers[0].bitwidth = 8

def process_video(fpath, model, device, args, output_dir, scale):
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
    
    # As the script uses constant dummy shapes, assuming args.width / args.height for crops usually
    target_w, target_h = args.width, args.height
    
    start_y = max(0, (height - target_h) // 2)
    start_x = max(0, (width - target_w) // 2)
    
    # Fallback to an MP4 backend explicitly
    out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(fpath))[0]}_output.mp4")
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
        
        # Preprocess (BGR -> RGB)
        img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # To Tensor [1, C, H, W]
        input_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
        
        # Inference
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
            
        output = torch.clamp(output, 0.0, 1.0)
        
        # Postprocess (RGB -> BGR, uint8)
        out_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_img = (out_img * 255.0).astype(np.uint8)
        out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        
        # Ensure the output frame size matches exactly
        if out_img_bgr.shape[1] != int(out_w) or out_img_bgr.shape[0] != int(out_h):
            out_img_bgr = cv2.resize(out_img_bgr, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)
            
        success = out.write(out_img_bgr)
        if not success and pbar.n < 5:
             pass
        pbar.update(1)
        
    pbar.close()
    cap.release()
    out.release()

def main():
    args = get_args()

    # 1. Device & Config
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device.isdigit():
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device}")
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Build or Load Base Model
    if args.base_model and os.path.exists(args.base_model):
        print(f">> Loading base model object from: {args.base_model}")
        checkpoint = torch.load(args.base_model, map_location=device, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint.to(device)
            print("[Success] Loaded base model object with pruned/compressed architecture.")
        else:
            print("[Warning] base_model is not a full model object. Building from config...")
            model = build_model(config['model']).to(device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
    else:
        print(f">> Building model: {config['model']['name']}...")
        model = build_model(config['model']).to(device)

    if hasattr(model, 'switch_to_deploy'):
        print(">> Switching model to deploy mode (reparameterization)...")
        model.switch_to_deploy()

    # 3. Prepare Model (Must match QAT script preparation exactly)
    print(">> Preparing Model (BN Folding, CLE)...")
    model = prepare_model(model)
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
    
    _ = fold_all_batch_norms(model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
    _ = equalize_model(model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
    model.eval()

    # 4. Initialize QuantSim
    quant_scheme = getattr(QuantScheme, "training_range_learning_with_tf_init") # Default for QAT
    print(f"\n>> Initializing QuantizationSimModel (Base: W{args.param_bw}A{args.output_bw})...")
    sim = QuantizationSimModel(model=model,
                               dummy_input=dummy_input,
                               quant_scheme=quant_scheme,
                               default_output_bw=args.output_bw,
                               default_param_bw=args.param_bw,
                               config_file="htp_v73")

    # 5. Apply MMP
    if args.mmp_config:
        apply_mmp_from_json(sim, args.mmp_config)
    else:
        apply_default_mmp(sim)

    # 6. Load Encodings
    if args.encodings and os.path.exists(args.encodings):
        print(f">> Loading encodings from {args.encodings}...")
        try:
            # New API
            sim.load_encodings(args.encodings, strict=False)
        except AttributeError:
            # Legacy fallback
            load_encodings_to_sim(sim, args.encodings)
    else:
        print("[Warning] No encodings provided. Model will use default initialization stats.")

    # 7. Load Checkpoint
    print(f">> Loading QAT checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

    try:
        sim.model.load_state_dict(state_dict)
        print("[Success] Loaded QAT model state dictionary.")
    except Exception as e:
        print(f"[Error] Failed to load checkpoint. {e}")
        return

    sim.model.eval()

    # 8. Discover Inputs
    input_paths = []
    if os.path.isfile(args.input):
        input_paths.append(args.input)
    elif os.path.isdir(args.input):
        exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.mp4', '*.avi', '*.mkv', '*.mov', '*.ts']
        for ext in exts:
            input_paths.extend(glob.glob(os.path.join(args.input, ext)))
            # Additionally grab case-insensitive matches usually handled by Python 3 glob directly
        input_paths.sort()
    else:
        print(f"Error: Input {args.input} not found.")
        return

    if not input_paths:
        print("No valid input media files found.")
        return
    scale = config['model']['scale']
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n>> Found {len(input_paths)} files. Starting inference...")

    # 9. Inference Loop
    with torch.no_grad():
        for path in tqdm(input_paths, desc="Total Files Inference"):
            ext = os.path.splitext(path)[-1].lower()
            if ext in ['.mp4', '.avi', '.mkv', '.mov', '.ts']:
                process_video(path, sim.model, device, args, args.output_dir, scale)
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Warning: Could not read image {path}")
                    continue
    
                # Preprocess (BGR -> RGB)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # To Tensor [1, C, H, W]
                input_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
    
                # Inference
                output = sim.model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                    
                output = torch.clamp(output, 0.0, 1.0)
    
                # Postprocess (RGB -> BGR, uint8)
                out_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                out_img = (out_img * 255.0).astype(np.uint8)
                out_img_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    
                # Save
                filename = os.path.basename(path)
                out_path = os.path.join(args.output_dir, filename)
                cv2.imwrite(out_path, out_img_bgr)

    print(f"\n>> Inference Complete. Results saved in '{args.output_dir}'")

if __name__ == "__main__":
    main()
