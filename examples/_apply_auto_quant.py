
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset
from src.aimet.utils import evaluate_model, save_active_results_to_csv, AutoQuantDatasetWrapper, create_sampled_data_loader

# AIMET Imports
from aimet_torch.model_preparer import prepare_model
from aimet_torch.auto_quant import AutoQuant
from aimet_torch.adaround.adaround_weight import AdaroundParameters
from aimet_torch.utils import in_eval_mode

def get_args():
    parser = argparse.ArgumentParser(description="AutoQuant Analysis")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file")
    parser.add_argument("--data_config", type=str, default="configs/data/sr.yaml", help="Data config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/auto_quant", help="Output directory")
    parser.add_argument("--allowed_drop", type=float, default=0.5, help="Allowed accuracy (PSNR) drop")
    
    parser.add_argument("--width", type=int, default=640, help="Width of dummy input")
    parser.add_argument("--height", type=int, default=360, help="Height of dummy input")
    
    parser.add_argument("--calib_batches", type=int, default=10, help="Number of batches for calibration (dataset size = batches * batch_size)")
    parser.add_argument("--ada_batches", type=int, default=10, help="Number of batches for Adaround")
    parser.add_argument("--ada_iter", type=int, default=2000, help="Number of iterations for AdaRound")
    
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

    # 4. Data Sets
    # Train Dataset (for Calibration/AdaRound) - Unlabeled
    if config['task'] == 'sr':
        train_dataset = SRDataset(dataset_root=config['data']['train_root'],
            scale_factor=config['model']['scale'],
            patch_size=config['train']['patch_size'],
            is_train=True,
            config=config.get('data_config', {}))
    elif config['task'] == 'denoise':
        train_dataset = DenoiseDataset(dataset_root=config['data']['train_root'],
            scale_factor=1,
            patch_size=config['train']['patch_size'],
            is_train=True,
            config=config.get('data_config', {}))
    else:
        raise ValueError("Unknown task")

    # Validation Dataset (for Evaluation)
    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
    else:
        print("Warning: validation paths not found. Cannot evaluate.")
        return

    # Wrap train dataset to return only inputs
    unlabeled_train_dataset = AutoQuantDatasetWrapper(val_dataset)

    # 5. Define Eval Callback
    def eval_callback(model_to_eval, num_samples=None):
        if num_samples is None:
            loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        else:
            # Create sampled loader from validation set
            loader = create_sampled_data_loader(val_dataset, num_samples, batch_size=1)
            
        return evaluate_model(model_to_eval, loader, device, title=None)

    # 6. AutoQuant Setup
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
    
    # Calibration Loader (Sampled from Train)
    # Calib batches * batch size (assume 16 from create_sampled_data_loader default, or pass it)
    calib_samples = args.calib_batches * 16 
    calib_loader = create_sampled_data_loader(unlabeled_train_dataset, calib_samples, batch_size=16)
    
    print(f">> Initializing AutoQuant (Calib Samples: {calib_samples})...")
    model = prepare_model(model)
    auto_quant = AutoQuant(model,
                           dummy_input=dummy_input,
                           data_loader=calib_loader,
                           eval_callback=eval_callback)

    # 7. Run Inference (Baseline)
    print(">> Running Baseline Inference...")
    sim, initial_psnr = auto_quant.run_inference()
    print(f"- Initial PSNR (FP32/Baseline): {initial_psnr:.4f}")
    
    # 8. Set AdaRound Params
    ada_samples = args.ada_batches * 16
    print(f">> Setting AdaRound Params (Samples: {ada_samples}, Iters: {args.ada_iter})...")
    
    # AdaRound Loader (Sampled from Train)
    ada_loader = create_sampled_data_loader(unlabeled_train_dataset, ada_samples, batch_size=16)
    ada_params = AdaroundParameters(data_loader=ada_loader, 
                                    num_batches=len(ada_loader), 
                                    default_num_iterations=args.ada_iter)
    auto_quant.set_adaround_params(ada_params)
    
    # 9. Optimize
    print(f">> Starting Optimization (Allowed Drop: {args.allowed_drop})...")
    
    trained_model, optimized_psnr, encoding_path = auto_quant.optimize(allowed_accuracy_drop=args.allowed_drop) 
    print(f"- Quantized PSNR (after optimization):  {optimized_psnr:.4f}")
    
    if encoding_path:
        print(f"- Encodings saved to: {encoding_path}")
        
    # 10. Save Result CSV
    results = [
        {
            'method': 'AutoQuant',
            'initial_psnr': initial_psnr,
            'optimized_psnr': optimized_psnr,
            'encoding_path': encoding_path
        }
    ]
    save_active_results_to_csv(results, args.output_dir, filename="auto_quant_results.csv")

    dummy_input = dummy_input.cpu()
    sim.export(path=args.output_dir, filename_prefix='auto_quant', dummy_input=dummy_input)

if __name__ == "__main__":
    main()
