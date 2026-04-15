
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset
from src.aimet.utils import evaluate_model, save_active_results_to_csv, AdaRoundDataLoader

# AIMET Imports
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.common.defs import QuantScheme, QuantizationDataType
from aimet_torch import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters


# python3 examples/3_apply_adaround.py --config configs/train/quicksrnet.yaml 
# --checkpoint checkpoints/train_quicksrnet_large_sr_x2_dim64_epoch800_bs_32_lr_1e-4/best.pth 
# --output_dir results/bnf_cle_ada --scheme post_training_tf_enhanced --param_bw 8 --output_bw 8 
# --ada_batches 100 --ada_iter 10000 --width 640 --height 360 --calib_batches 10 --device 0

def get_args():
    parser = argparse.ArgumentParser(description="Targeted AdaRound Analysis")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file")
    parser.add_argument("--data_config", type=str, default="configs/data/sr.yaml", help="Data config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/adaround_analysis", help="Output directory")
    
    # Quantization Params
    parser.add_argument("--scheme", type=str, default="post_training_tf_enhanced", 
                        choices=["post_training_tf", "post_training_tf_enhanced", "post_training_percentile"],
                        help="Quantization Scheme")
    parser.add_argument("--param_bw", type=int, default=8, help="Parameter Bitwidth")
    parser.add_argument("--output_bw", type=int, default=8, help="Output/Activation Bitwidth")
    parser.add_argument("--skip", action="store_true", help="Skip AdaRound")
    
    # AdaRound Params
    parser.add_argument("--ada_batches", type=int, default=2000, help="Number of batches for AdaRound")
    parser.add_argument("--ada_iter", type=int, default=10000, help="Number of iterations for AdaRound")
    
    parser.add_argument("--width", type=int, default=640, help="Width of dummy input")
    parser.add_argument("--height", type=int, default=360, help="Height of dummy input")
    
    parser.add_argument("--calib_batches", type=int, default=100, help="Number of batches for calibration")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

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
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 4. Data Loaders
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
    
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
        
    train_loader = DataLoader(dataset, batch_size=config['train'].get('batch_size', 16), shuffle=True, num_workers=4, drop_last=True)

    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        print("Warning: validation paths not found. Cannot evaluate.")
        return

    # 5. Baseline Evaluation
    print(">> Baseline FP32 Evaluation...")
    fp32_psnr = evaluate_model(model, val_loader, device, title="FP32 Baseline")

    # 6. Prepare Model (BN Folding & CLE)
    print(">> Preparing Model (BN Folding & CLE)...")
    prepared_model = prepare_model(model)
    _ = fold_all_batch_norms(prepared_model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
    _ = equalize_model(prepared_model, input_shapes=dummy_input.shape, dummy_input=dummy_input)

    # 7. Apply AdaRound
    quant_scheme = getattr(QuantScheme, args.scheme)
    param_bw = args.param_bw
    output_bw = args.output_bw
    if not args.skip:
        print(f">> Running AdaRound (Scheme: {args.scheme}, Param BW: {param_bw})...")
    
        ada_loader = AdaRoundDataLoader(val_loader, device=device)
        
        ada_params = AdaroundParameters(data_loader=ada_loader, 
                                        num_batches=args.ada_batches, 
                                        default_num_iterations=args.ada_iter)
        
        encoding_path = os.path.join(args.output_dir, f'adaround_{args.scheme}_{param_bw}bit')
        os.makedirs(args.output_dir, exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True to a scalar")
            ada_model = Adaround.apply_adaround(prepared_model, dummy_input, ada_params,
                                                path=args.output_dir,
                                                filename_prefix=f'adaround_{args.scheme}_{param_bw}bit',
                                                default_param_bw=param_bw,
                                                default_quant_scheme=quant_scheme)
    else:
        ada_model = prepared_model
        encoding_path = os.path.join(args.output_dir, f'adaround_{args.scheme}_{param_bw}bit.encodings')

    # 8. Create QuantSim & Apply Adaround Encodings
    print(f">> Creating QuantSim (Output BW: {output_bw})...")
    sim = QuantizationSimModel(model=ada_model,
                               dummy_input=dummy_input,
                               quant_scheme=quant_scheme,
                               default_output_bw=output_bw,
                               default_param_bw=param_bw,
                               in_place=False,
                               config_file="htp_v73",
                               default_data_type=QuantizationDataType.int)
    
    sim.set_and_freeze_param_encodings(encoding_path=encoding_path)
    
    # 9. Calibrate Activations
    print(">> Calibrating Activations...")
    def pass_data(model, loader):
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.calib_batches: break
                if isinstance(batch, dict): inputs = batch['lr'].to(device)
                else: inputs = batch[0].to(device)
                model(inputs)
    
    sim.compute_encodings(forward_pass_callback=pass_data, forward_pass_callback_args=train_loader)
    
    # 10. Evaluate Quantized Model
    print(">> Evaluating Quantized Model...")
    quant_psnr = evaluate_model(sim.model, val_loader, device, title="Quantized Model")
    
    # 11. Results
    results = [
        {
            'data_type': 'Float', 
            'scheme': 'None', 
            'output_bw': 32, 
            'param_bw': 32, 
            'psnr': fp32_psnr, 
        },
        {
            'data_type': 'int', 
            'scheme': args.scheme, 
            'output_bw': output_bw, 
            'param_bw': param_bw, 
            'psnr': quant_psnr, 
        }
    ]
    
    print("\n" + "="*80)
    print("   [Targeted AdaRound Analysis Results]")
    print("="*80)
    print(f"{'Type':<10} | {'Scheme':<30} | {'W/A':<10} | {'PSNR':<10}")
    print("-" * 80)
    print(f"{'Float':<10} | {'None':<30} | {'32/32':<10} | {fp32_psnr:<10.4f}")
    print(f"{'Int':<10} | {args.scheme:<30} | {param_bw}/{output_bw:<10} | {quant_psnr:<10.4f}")
    print("="*80 + "\n")
    
    save_active_results_to_csv(results, args.output_dir, filename=f"result_{args.scheme}_W{param_bw}A{output_bw}.csv")

if __name__ == "__main__":
    main()
