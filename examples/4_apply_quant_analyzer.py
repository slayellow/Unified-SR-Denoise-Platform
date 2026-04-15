
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import copy
import json
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset
from src.aimet.utils import evaluate_model, create_sampled_data_loader, AutoQuantDatasetWrapper

# AIMET Imports
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quant_analyzer import QuantAnalyzer
from aimet_torch.common.utils import CallbackFunc
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

# python3 examples/4_apply_quant_analyzer.py --config configs/train/quicksrnet.yaml --data_config configs/data/sr.yaml 
# --checkpoint checkpoints/train_quicksrnet_large_sr_x2_dim64_epoch800_bs_32_lr_1e-4/best.pth 
# --output_dir results/quicksrnet_larger_quant_analyzer_ob4_pb4_tfe --scheme post_training_tf_enhanced 
# --param_bw 4 --output_bw 4 --width 640 --height 360 --calib_batches 100 --device 1

def get_args():
    parser = argparse.ArgumentParser(description="QuantAnalyzer Application")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file")
    parser.add_argument("--data_config", type=str, default="configs/data/sr.yaml", help="Data config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/quant_analyzer", help="Output directory")
    
    # Quantization Params
    parser.add_argument("--scheme", type=str, default="post_training_tf_enhanced", 
                        choices=["post_training_tf", "post_training_tf_enhanced", "post_training_percentile"],
                        help="Quantization Scheme")
    parser.add_argument("--param_bw", type=int, default=8, help="Parameter Bitwidth")
    parser.add_argument("--output_bw", type=int, default=8, help="Output/Activation Bitwidth")
    
    parser.add_argument("--width", type=int, default=640, help="Width of dummy input")
    parser.add_argument("--height", type=int, default=360, help="Height of dummy input")
    
    parser.add_argument("--calib_batches", type=int, default=500, help="Number of samples for calibration analysis (Default: 500)")
    
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def pass_calibration_data(model, loader):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Handle dictionary or tuple/list based on dataset wrapper
            if isinstance(batch, dict):
                inputs = batch['lr'].to(device)
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            
            model(inputs)

def eval_model_callback(model, loader):
    device = next(model.parameters()).device
    return evaluate_model(model, loader, device, title=None)

def export_quant_config(sim, output_dir):
    config_list = []
    txt_lines = []

    for name, module in sim.model.named_modules():
        # Skip top level module if it just wraps everything
        if name == "":
            continue
            
        layer_info = {"name": name, "module_type": type(module).__name__}
        has_quantizer = False
        
        # Check param_quantizers
        if hasattr(module, "param_quantizers") and module.param_quantizers:
            param_q = {}
            for k, v in module.param_quantizers.items():
                param_q[k] = str(v) if v is not None else "None"
            layer_info["param_quantizers"] = param_q
            has_quantizer = True
        else:
            layer_info["param_quantizers"] = "None"

        # Check input_quantizers
        if hasattr(module, "input_quantizers") and module.input_quantizers:
            input_q = {}
            for i, v in enumerate(module.input_quantizers):
                input_q[str(i)] = str(v) if v is not None else "None"
            layer_info["input_quantizers"] = input_q
            has_quantizer = True
        else:
            layer_info["input_quantizers"] = "None"
            
        # Check output_quantizers
        if hasattr(module, "output_quantizers") and module.output_quantizers:
            output_q = {}
            for i, v in enumerate(module.output_quantizers):
                output_q[str(i)] = str(v) if v is not None else "None"
            layer_info["output_quantizers"] = output_q
            has_quantizer = True
        else:
            layer_info["output_quantizers"] = "None"

        # Only record if it looks like a quantized module (has quantizers) or is a Quantized module class
        if has_quantizer or "Quantized" in type(module).__name__:
            config_list.append(layer_info)
            
            # Format for text file
            txt_lines.append(f"Layer: {name} ({type(module).__name__})")
            if isinstance(layer_info["param_quantizers"], dict):
                txt_lines.append("  (param_quantizers): ModuleDict(")
                for k, v in layer_info["param_quantizers"].items():
                    txt_lines.append(f"    ({k}): {v}")
                txt_lines.append("  )")
            else:
                txt_lines.append(f"  (param_quantizers): {layer_info['param_quantizers']}")

            if isinstance(layer_info["input_quantizers"], dict):
                txt_lines.append("  (input_quantizers): ModuleList(")
                for k, v in layer_info["input_quantizers"].items():
                     txt_lines.append(f"    ({k}): {v}")
                txt_lines.append("  )")
            else:
                 txt_lines.append(f"  (input_quantizers): {layer_info['input_quantizers']}")

            if isinstance(layer_info["output_quantizers"], dict):
                txt_lines.append("  (output_quantizers): ModuleList(")
                for k, v in layer_info["output_quantizers"].items():
                     txt_lines.append(f"    ({k}): {v}")
                txt_lines.append("  )")
            else:
                 txt_lines.append(f"  (output_quantizers): {layer_info['output_quantizers']}")
            
            txt_lines.append("-" * 30)

    # Save JSON
    json_path = os.path.join(output_dir, "quant_layer_config.json")
    with open(json_path, "w") as f:
        json.dump(config_list, f, indent=2)
    
    # Save Text
    txt_path = os.path.join(output_dir, "quant_layer_config.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))
        
    print(f"[Info] Quantization layer config exported to {output_dir}")

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

    # Prepare model
    print(">> Preparing Model...")
    model = prepare_model(model)
    model.eval()

    # 4. Data Loaders
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
    
    # Calibration Loader (Unlabeled Train)
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
        
    unlabeled_train_dataset = AutoQuantDatasetWrapper(train_dataset)
    
    print(f"Creating Calibration Loader ({args.calib_batches} samples from Training Data)...")
    # Use batch_size=1 to have exact control over number of samples
    calib_loader = create_sampled_data_loader(unlabeled_train_dataset, args.calib_batches, batch_size=1)
    print(f"[Info] Calibration loader created ({len(calib_loader)} samples)")

    # 4.5 Export Default Quantization Config
    print(">> Exporting Default Quantization Config...")
    # Create output dir if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    sim_model = copy.deepcopy(model)
    dummy_input_export = torch.randn(1, 3, args.height, args.width).to(device)
    quant_scheme = getattr(QuantScheme, args.scheme)
    
    # Initialize QuantizationSimModel to see default quantizer placement
    # We use the same config_file "htp_v73" to match what QuantAnalyzer likely uses or standard HTP
    sim = QuantizationSimModel(sim_model, dummy_input=dummy_input_export, 
                               quant_scheme=quant_scheme,
                               default_param_bw=args.param_bw,
                               default_output_bw=args.output_bw,
                               config_file="htp_v73")
                               
    export_quant_config(sim, args.output_dir)
    del sim, sim_model
    torch.cuda.empty_cache()

    # Validation Loader
    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        print("Warning: validation paths not found. Cannot evaluate.")
        return

    # 5. Define Callbacks
    forward_pass_cb = CallbackFunc(pass_calibration_data, calib_loader)
    eval_cb = CallbackFunc(eval_model_callback, val_loader)

    # 6. QuantAnalyzer Setup
    print(">> Initializing QuantAnalyzer...")
    analyzer = QuantAnalyzer(model, dummy_input, forward_pass_callback=forward_pass_cb, eval_callback=eval_cb)
    
    # Enable per-layer MSE loss
    print(f">> Enabling Per-Layer MSE Loss Analysis (Batches: {len(calib_loader)})...")
    # Note: data_loader passed here is used for MSE calculation
    analyzer.enable_per_layer_mse_loss(calib_loader, num_batches=len(calib_loader))
    
    # 7. Analyze
    quant_scheme = getattr(QuantScheme, args.scheme)
    param_bw = args.param_bw
    output_bw = args.output_bw
    
    print(f">> Running Analysis (Scheme: {args.scheme}, Param BW: {param_bw}, Output BW: {output_bw})...")
    
    # Create output dir if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyzer.analyze(quant_scheme=quant_scheme,
                     default_param_bw=param_bw,
                     default_output_bw=output_bw,
                     config_file="htp_v73",
                     results_dir=args.output_dir)
                     
    print(f"\n[QuantAnalyzer] Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
