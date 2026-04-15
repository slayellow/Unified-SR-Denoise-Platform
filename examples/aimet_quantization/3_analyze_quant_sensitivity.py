
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import build_model
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset
from src.aimet.utils import evaluate_model, save_active_results_to_csv, create_sampled_data_loader

# AIMET Imports
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.common.defs import QuantScheme, QuantizationDataType
from aimet_torch import QuantizationSimModel

# python3 examples/3_analyze_quant_sensitivity.py --config configs/train/quicksrnet.yaml 
# --checkpoint checkpoints/train_quicksrnet_large_sr_x2_dim64_epoch800_bs_32_lr_1e-4/best.pth 
# --device 0 --calib_batches 10 --data_config configs/data/sr.yaml --use_bn_folding --use_cle 
# --output_dir results/bnf_cle_noada

def get_args():
    parser = argparse.ArgumentParser(description="Analyze Quantization Sensitivity")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file")
    parser.add_argument("--data_config", type=str, default="configs/data/sr.yaml", help="Data config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/sensitivity", help="Output directory for plots/results")
    parser.add_argument("--calib_batches", type=int, default=500, help="Number of batches for calibration (Default: 500)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bn_folding", action="store_true", help="Use Batch Normalization Folding")
    parser.add_argument("--use_cle", action="store_true", help="Use Cross Layer Equalization")
    parser.add_argument("--width", type=int, default=640, help="Width of dummy input")
    parser.add_argument("--height", type=int, default=360, help="Height of dummy input")
    return parser.parse_args()

def visualize_results(results, output_dir):
    """
    Generate plots for PSNR vs Bitwidth
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter only successful results
    valid_results = [r for r in results if isinstance(r['psnr'], (float, int))]
    
    # Group by Scheme/DataType
    groups = {}
    for r in valid_results:
        key = f"{r['scheme']}_{r['data_type']}"
        if key not in groups:
            groups[key] = {'bw': [], 'psnr': []}
        # Use Output BW as x-axis
        groups[key]['bw'].append(r['output_bw'])
        groups[key]['psnr'].append(r['psnr'])
        
    # Plot PSNR
    plt.figure(figsize=(10, 6))
    for key, data in groups.items():
        # Sort by bw
        sorted_pairs = sorted(zip(data['bw'], data['psnr']))
        bw = [p[0] for p in sorted_pairs]
        val = [p[1] for p in sorted_pairs]
        plt.plot(bw, val, marker='o', label=key)
        
    plt.title("Quantization Sensitivity: PSNR vs Output Bitwidth")
    plt.xlabel("Output Bitwidth")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sensitivity_psnr.png"))
    plt.close()

    print(f"[Visualization] Plots saved to {output_dir}")

def main():
    args = get_args()

    # 1. Device
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
    
    # 2. Config & Model
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load Data Config
    if args.data_config and os.path.exists(args.data_config):
        with open(args.data_config, 'r') as f:
            data_config = yaml.safe_load(f)
        config['data_config'] = data_config
    else:
        # Fallback logic to defaults if not provided, similar to quant_test
        if config['task'] == 'sr':
            default_path = "configs/data/sr.yaml"
        elif config['task'] == 'denoise':
            default_path = "configs/data/denoise.yaml"
        else:
            default_path = None
            
        if default_path and os.path.exists(default_path):
             with open(default_path, 'r') as f:
                config['data_config'] = yaml.safe_load(f)
        else:
            config['data_config'] = {}
        
    print(f"Loading model: {config['model']['name']}...")
    model = build_model(config['model']).to(device)
    
    # 3. Checkpoint
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
    if hasattr(model, 'switch_to_deploy'):
        print("Switching model to deploy mode...")
        model.switch_to_deploy()

    # 4. Data Loaders
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device) # Customizable dummy shape
    
    if config['task'] == 'sr':
        dataset = SRDataset(dataset_root=config['data']['train_root'],
            scale_factor=config['model']['scale'],
            patch_size=config['train']['patch_size'],
            is_train=True,
            config=config.get('data_config', {}))
    elif config['task'] == 'denoise':
        dataset = DenoiseDataset(dataset_root=config['data']['train_root'],
            scale_factor=1,
            patch_size=config['train']['patch_size'],
            is_train=True,
            config=config.get('data_config', {}))
    else:
        raise ValueError("Unknown task")
        
    # train_loader = DataLoader(dataset, batch_size=config['train'].get('batch_size', 16), shuffle=True, num_workers=4, drop_last=True)
    
    # Validation Loader (for Evaluation Only)

    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        print(f"[Info] Validation loader created ({len(val_dataset)} samples)")
    else:
        print("Warning: validation paths not found in config. Skipping evaluation.")
        val_loader = None
        
    # Calibration Loader (for Quantization Calibration - Subset of Training Data)
    print(f"Creating Calibration Loader ({args.calib_batches} samples from Training Data)...")
    calib_loader = create_sampled_data_loader(dataset, args.calib_batches, batch_size=1)
    print(f"[Info] Calibration loader created ({len(calib_loader)} samples)")

    # 5. Baseline Evaluation
    print(">> Baseline FP32 Evaluation...")
    fp32_psnr = evaluate_model(model, val_loader, device)
    
    # 6. Prepare Model
    print(">> Preparing Model (BN Folding, etc.)...")
    prepared_model = prepare_model(model)

    if args.use_bn_folding:
        _ = fold_all_batch_norms(prepared_model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
        print(">> Batch Normalization Folding Applied")
    if args.use_cle:
        _ = equalize_model(prepared_model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
        print(">> Cross Layer Equalization Applied")

    # 7. Sensitivity Analysis Loop
    print(">> Starting Sensitivity Analysis...")
    scheme = [QuantScheme.post_training_tf, QuantScheme.post_training_tf_enhanced, QuantScheme.post_training_percentile]
    output_bw = [4, 8, 16]
    param_bw = [4, 8, 16]
    data_type = [QuantizationDataType.int, QuantizationDataType.float]
    
    analysis_results = []
    
    for s in scheme:
        for obw in output_bw:
            for idx, pbw in enumerate(param_bw):
                for dt in data_type:
                    # Validity Checks
                    if dt == QuantizationDataType.float:
                        if obw not in [16, 32] or pbw not in [16, 32]:
                            continue
                    elif dt == QuantizationDataType.int:
                        if obw not in [4, 8, 16] or pbw not in [4, 8, 16]:
                            continue

                    try:
                        current_model = prepared_model
                        sim = QuantizationSimModel(model=current_model, 
                                                   dummy_input=dummy_input,
                                                   quant_scheme=s, 
                                                   default_output_bw=obw,
                                                   default_param_bw=pbw,
                                                   in_place=False,
                                                   config_file="htp_v73",
                                                   default_data_type=dt)
                        
                        # Calibration
                        def pass_data(model, loader):
                            with torch.no_grad():
                                for i, batch in enumerate(loader):
                                    if i >= args.calib_batches: break
                                    if isinstance(batch, dict): inputs = batch['lr'].to(device)
                                    else: inputs = batch[0].to(device)
                                    model(inputs)
                                    
                        sim.compute_encodings(forward_pass_callback=pass_data, forward_pass_callback_args=calib_loader)
                        
                        # Evaluation
                        p = evaluate_model(sim.model, val_loader, device)
                        
                        analysis_results.append({
                            'scheme': str(s).replace('QuantScheme.', ''),
                            'output_bw': obw,
                            'param_bw': pbw,
                            'data_type': str(dt).replace('QuantizationDataType.', ''),
                            'psnr': p
                        })
                        print(f"[{s}|{dt}|W{pbw}A{obw}] PSNR: {p:.4f}")
                        
                    except Exception as e:
                        print(f"[Error] Failed config: {s}, W{pbw}A{obw}, {dt}. Error: {e}")
                        analysis_results.append({
                            'scheme': str(s).replace('QuantScheme.', ''),
                            'output_bw': obw,
                            'param_bw': pbw,
                            'data_type': str(dt).replace('QuantizationDataType.', ''),
                            'psnr': 'Failed'
                        })

    # 8. Sort and Print Results
    # Sort by Output BW (descending), then Param BW
    valid_results = [r for r in analysis_results if isinstance(r['psnr'], (float, int))]
    failed_results = [r for r in analysis_results if not isinstance(r['psnr'], (float, int))]
    
    valid_results.sort(key=lambda x: (x['output_bw'], x['param_bw']), reverse=True)
    
    print("\n" + "="*110)
    print("   [Sensitivity Analysis Summary] (Sorted by Bitwidth)")
    print("="*110)
    print(f"{'Data Type':<15} | {'Scheme':<35} | {'Output BW':<10} | {'Param BW':<10} | {'PSNR':<10}")
    print("-" * 110)
    print(f"{'Float':<15} | {'None':<35} | {'32':<10} | {'32':<10} | {fp32_psnr:<10.4f}")
    
    for res in valid_results:
        print(f"{res['data_type']:<15} | {res['scheme']:<35} | {res['output_bw']:<10} | {res['param_bw']:<10} | {res['psnr']:<10.4f}")
        
    if failed_results:
        print("-" * 110)
        print("Failed Configurations:")
        for res in failed_results:
            print(f"{res['data_type']:<15} | {res['scheme']:<35} | {res['output_bw']:<10} | {res['param_bw']:<10} | {res['psnr']:<10}")
    print("="*110 + "\n")
    
    # 9. Visualize & Save CSV
    save_active_results_to_csv(analysis_results, args.output_dir, filename="sensitivity_results.csv")
    visualize_results(analysis_results, args.output_dir)

if __name__ == "__main__":
    main()
