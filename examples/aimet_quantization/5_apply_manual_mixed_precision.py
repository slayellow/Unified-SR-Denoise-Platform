import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader


# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import build_model
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset
from src.aimet.utils import evaluate_model, create_sampled_data_loader, AutoQuantDatasetWrapper, apply_mmp_from_json

# AIMET Imports
from aimet_common.defs import QuantScheme

from src.aimet.quant_sim import DEFAULT_QUANTSIM_CONFIG, prepare_deploy_model, build_quantsim

def get_args():
    parser = argparse.ArgumentParser(description="Manual Mixed Precision Application")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data_config", type=str, required=True, help="Data config of the recipe; calibration draws from it")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/mmp", help="Output directory")
    
    parser.add_argument("--param_bw", type=int, default=8, help="Default Parameter Bitwidth")
    parser.add_argument("--output_bw", type=int, default=4, help="Default Output Bitwidth (Target: 4)")
    parser.add_argument("--scheme", type=str, default="post_training_tf_enhanced", 
                        choices=["post_training_tf", "post_training_tf_enhanced", "post_training_percentile"],
                        help="Quantization Scheme")
    
    parser.add_argument("--width", type=int, default=640, help="Width of dummy input")
    parser.add_argument("--height", type=int, default=360, help="Height of dummy input")
    parser.add_argument("--calib_batches", type=int, default=500, help="Number of samples for calibration")
    parser.add_argument("--mmp_config", type=str, required=True, help="Path to MMP JSON config: the layers whose bitwidth to override")
    parser.add_argument("--quantsim_config", type=str, default=DEFAULT_QUANTSIM_CONFIG,
                        help="AIMET quantsim config: builtin name (e.g. htp_v73) or path to a JSON file. "
                             "Default: repo config without the Conv+LeakyRelu supergroup, so LeakyReLU "
                             "models keep Conv output encodings during DLC conversion.")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def pass_calibration_data(model, loader):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                inputs = batch['lr'].to(device)
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            model(inputs)

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
        
    # A missing recipe must stop the run: an empty data config would calibrate on every
    # degradation at its default instead of on the recipe the model was trained with.
    with open(args.data_config, 'r') as f:
        config['data_config'] = yaml.safe_load(f)

    # 2. Load Model & Checkpoint
    print(f"Loading model: {config['model']['name']}...")
    model = build_model(config['model']).to(device)

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

    # 3. switch_to_deploy -> prepare_model -> eval (single definition in src/aimet/quant_sim.py)
    model = prepare_deploy_model(model)

    # 4. Data Loaders
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
    
    # Validation Loader Setup
    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    val_loader = None
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Calibration Loader Setup
    if config['task'] == 'sr':
        train_dataset = SRDataset(dataset_root=config['data']['train_root'],
            scale_factor=config['model']['scale'],
            patch_size=config['train']['patch_size'],
            is_train=True, config=config.get('data_config', {}))
    elif config['task'] == 'denoise':
        train_dataset = DenoiseDataset(dataset_root=config['data']['train_root'],
            scale_factor=1,
            patch_size=config['train']['patch_size'],
            is_train=True, config=config.get('data_config', {}))
    
    print(f"Creating Calibration Loader ({args.calib_batches} samples)...")
    unlabeled_train_dataset = AutoQuantDatasetWrapper(train_dataset)
    calib_loader = create_sampled_data_loader(unlabeled_train_dataset, args.calib_batches, batch_size=1)

    # =========================================================================
    # [Evaluation 1] FP32 Baseline Performance
    # =========================================================================
    print("\n" + "="*50)
    print(" [Step 1] Evaluating FP32 Model...")
    print("="*50)
    acc_fp32 = 0.0
    if val_loader:
        acc_fp32 = evaluate_model(model, val_loader, device)
        print(f">> FP32 Accuracy (PSNR): {acc_fp32:.4f} dB")
    else:
        print(">> Validation loader not found. Skipping Eval.")

    # 5. Initialize QuantizationSimModel (Base W8A4)
    print(f"\n>> Initializing QuantizationSimModel (Base: W{args.param_bw}A{args.output_bw})...")
    quant_scheme = getattr(QuantScheme, args.scheme)
    
    # fold+CLE run inside build_quantsim, after the FP32 eval as before.
    sim = build_quantsim(model, dummy_input,
                         quant_scheme=quant_scheme,
                         default_output_bw=args.output_bw,
                         default_param_bw=args.param_bw,
                         config_file=args.quantsim_config,
                         prepare=False)

    

    # =========================================================================
    # [Evaluation 2] W8A4 Baseline Performance (Pre-MMP)
    # =========================================================================
    print("\n" + "="*50)
    print(f" [Step 2] Evaluating Baseline Quantized Model (W{args.param_bw}A{args.output_bw})...")
    print("="*50)
    
    # MMP 적용 전, 기본 설정으로 Encodings 계산 (1차 Calibration)
    print(">> Computing Encodings (Base)...")
    sim.compute_encodings(forward_pass_callback=pass_calibration_data, forward_pass_callback_args=calib_loader)
    
    acc_base = 0.0
    if val_loader:
        acc_base = evaluate_model(sim.model, val_loader, device)
        print(f">> Baseline W{args.param_bw}A{args.output_bw} PSNR: {acc_base:.4f} dB")
        print(f">> Drop from FP32: {acc_base - acc_fp32:.4f} dB")

    # =========================================================================
    # [Step 3] Apply Manual Mixed Precision (MMP)
    # =========================================================================
    print("\n" + "="*50)
    print(" [Step 3] Applying Manual Mixed Precision (MMP)...")
    print("="*50)
    
    apply_mmp_from_json(sim, args.mmp_config)

    # Re-compute Encodings for changed layers (2차 Calibration)
    print("\n>> Re-Computing Encodings (MMP Applied)...")
    sim.compute_encodings(forward_pass_callback=pass_calibration_data, forward_pass_callback_args=calib_loader)

    # =========================================================================
    # [Evaluation 3] W8A4 MMP Performance (Post-MMP)
    # =========================================================================
    print("\n" + "="*50)
    print(" [Step 4] Evaluating MMP Quantized Model...")
    print("="*50)
    
    acc_mmp = 0.0
    if val_loader:
        acc_mmp = evaluate_model(sim.model, val_loader, device)
        print(f">> MMP W{args.param_bw}A{args.output_bw} PSNR: {acc_mmp:.4f} dB")
        print(f">> Improvement from Base: +{acc_mmp - acc_base:.4f} dB")
        print(f">> Drop from FP32: {acc_mmp - acc_fp32:.4f} dB")

    # 6. Export
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n>> Exporting to {args.output_dir}...")
    sim.export(path=args.output_dir, filename_prefix="mmp_model", dummy_input=dummy_input.cpu())

    # Final Summary
    print("\n" + "#"*60)
    print("                  FINAL SUMMARY                   ")
    print("#"*60)
    print(f" 1. FP32 Model PSNR       : {acc_fp32:.4f} dB")
    print(f" 2. Base W{args.param_bw}A{args.output_bw} PSNR       : {acc_base:.4f} dB")
    print(f" 3. MMP  W{args.param_bw}A{args.output_bw} PSNR       : {acc_mmp:.4f} dB")
    print("-" * 60)
    print(f" Recovery via MMP         : +{acc_mmp - acc_base:.4f} dB")
    print("#"*60)

if __name__ == "__main__":
    main()