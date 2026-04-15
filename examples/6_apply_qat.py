
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset
from src.aimet.utils import evaluate_model, create_sampled_data_loader, AutoQuantDatasetWrapper, apply_mmp_from_json
from src.engine.trainer import Trainer

# AIMET Imports
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_common.defs import QuantScheme
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model

# python3 examples/6_apply_qat.py --config configs/finetune/svsrnet.yaml --data_config configs/data/sr_finetune_eo.yaml 
# --checkpoint results/svsrnet_pruning_0.5/compressed_model.pth --output_dir results/20260212_svsrnet_qat 
# --scheme training_range_learning_with_tf_init --param_bw 8 --output_bw 8 --qat_epochs 20 --qat_lr 1e-5 
# --train_data_ratio 0.2 --encodings results/20260211_svsrnet_mmp/mmp_model_torch.encodings --width 640 --height 360 --device 4

def get_args():
    parser = argparse.ArgumentParser(description="QAT Application")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file")
    parser.add_argument("--data_config", type=str, default="configs/data/sr.yaml", help="Data config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/qat", help="Output directory")
    
    # Quantization Params
    parser.add_argument("--scheme", type=str, default="training_range_learning_with_tf_init", 
                        choices=["post_training_tf", "post_training_tf_enhanced", "training_range_learning_with_tf_init"],
                        help="Quantization Scheme (Default: training_range_learning_with_tf_init for QAT)")
    parser.add_argument("--param_bw", type=int, default=8, help="Default Parameter Bitwidth")
    parser.add_argument("--output_bw", type=int, default=8, help="Default Output Bitwidth")
    
    # QAT Params
    parser.add_argument("--qat_epochs", type=int, default=30, help="Number of QAT epochs")
    parser.add_argument("--qat_lr", type=float, default=5e-5, help="Learning rate for QAT")
    parser.add_argument("--train_data_ratio", type=float, default=0.3, help="Ratio of training data to use for QAT (0.1 ~ 0.3 recommended)")
    parser.add_argument("--encodings", type=str, default=None, help="Path to pre-computed .encodings file (Optional)")
    parser.add_argument("--mmp_config", type=str, default=None, help="Path to MMP JSON config (Optional)")

    parser.add_argument("--width", type=int, default=640, help="Width of dummy input")
    parser.add_argument("--height", type=int, default=360, help="Height of dummy input")
    parser.add_argument("--calib_batches", type=int, default=500, help="Number of samples for calibration (if encodings not provided)")
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
        
    if args.data_config and os.path.exists(args.data_config):
        with open(args.data_config, 'r') as f:
            data_config = yaml.safe_load(f)
        config['data_config'] = data_config
    else:
        config['data_config'] = {}

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

    if hasattr(model, 'switch_to_deploy'):
        print(">> Switching model to deploy mode (reparameterization)...")
        model.switch_to_deploy()
    
    # 3. Prepare Model (BN Folding)
    print(">> Preparing Model (BN Folding)...")
    model = prepare_model(model)
    model.eval()

    # 4. Data Loaders
    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)
    
    # Validation Loader Setup
    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    val_loader = None
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Training Loader Setup (for QAT and Calibration)
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
    
    # Subsample Dataset for QAT
    total_len = len(train_dataset)
    subset_len = int(total_len * args.train_data_ratio)
    print(f">> Training Data: Using {subset_len} samples out of {total_len} ({args.train_data_ratio*100}%)")
    indices = torch.randperm(total_len)[:subset_len]
    train_subset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_subset, 
                              batch_size=config['train']['batch_size'], 
                              shuffle=True, num_workers=config['train']['num_workers'], drop_last=True)

    # Calibration Loader (from subset or full?) -> Typically from same distribution. 
    # We can use create_sampled_data_loader on the subset or the original.
    # Let's use the subset to be safe/consistent, or just a small slice.
    print(f"Creating Calibration Loader ({args.calib_batches} samples)...")
    unlabeled_train_dataset = AutoQuantDatasetWrapper(train_dataset) # Use full dataset wrapper for calib to allow diversity if needed
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

    _ = fold_all_batch_norms(model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
    print(">> Batch Normalization Folding Applied")
    _ = equalize_model(model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
    print(">> Cross Layer Equalization Applied")

    # 5. Initialize QuantizationSimModel
    print(f"\n>> Initializing QuantizationSimModel (Base: W{args.param_bw}A{args.output_bw})...")
    quant_scheme = getattr(QuantScheme, args.scheme)
    
    sim = QuantizationSimModel(model=model,
                               dummy_input=dummy_input,
                               quant_scheme=quant_scheme,
                               default_output_bw=args.output_bw,
                               default_param_bw=args.param_bw,
                               config_file="htp_v73") # Ensure config file exists

    # =========================================================================
    # [Step 2] Apply Manual Mixed Precision (MMP) --> 이 부분은 수동으로 모델의 상태에 따라 계속 변경될 수 있음
    # =========================================================================
    print("\n" + "="*50)
    print(" [Step 2] Applying Manual Mixed Precision (MMP)...")
    print("="*50)
    
    if args.mmp_config:
        apply_mmp_from_json(sim, args.mmp_config)
    else:
        print("[Info] No MMP config provided. Applying default hardcoded MMP logic (SVSRNet Example)...")
        # 1. High MSE Layers (5, 6, 7, 11) -> Need INT8 Input
        high_mse_indices = [5, 6, 7, 11]
        # 2. Sensitive Layer (10) -> Needs INT8 Output
        sensitive_indices = [10]

        target_modules = []

        # A. Head Protection
        target_modules.append(("head", sim.model.head))

        # B. Fix High MSE Inputs (Target Previous Act)
        for idx in high_mse_indices:
            prev_idx = idx - 1
            if prev_idx >= 0:
                try:
                    prev_block = getattr(sim.model.body, str(prev_idx))
                    target_modules.append((f"body.{prev_idx}.act", prev_block.act))
                except AttributeError:
                    pass

        # C. Fix Sensitive Outputs (Target Current Act)
        for idx in sensitive_indices:
            try:
                curr_block = getattr(sim.model.body, str(idx))
                target_modules.append((f"body.{idx}.act", curr_block.act))
            except AttributeError:
                pass

        # D. Tail Protection
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

        # Apply Configuration
        unique_modules = {name: mod for name, mod in target_modules}
        for name, module in unique_modules.items():
            print(f"   [MMP] Setting Output of '{name}' to INT8")
            if hasattr(module, 'output_quantizers') and module.output_quantizers:
                if module.output_quantizers[0] is not None:
                    module.output_quantizers[0].bitwidth = 8
                    
            if name == "head":
                print(f"   [MMP] Setting Input of '{name}' to INT8")
                if hasattr(module, 'input_quantizers') and module.input_quantizers:
                    if module.input_quantizers[0] is not None:
                        module.input_quantizers[0].bitwidth = 8

    # =========================================================================
    # [Step 3] Encodings (Load or Compute)
    # =========================================================================
    print("\n" + "="*50)
    print(" [Step 3] Setting up Encodings...")
    print("="*50)
    
    if args.encodings and os.path.exists(args.encodings):
        print(f">> Loading encodings from {args.encodings}...")
        load_encodings_to_sim(sim, args.encodings)
    else:
        print(">> Computing Encodings (Calibration)...")
        sim.compute_encodings(forward_pass_callback=pass_calibration_data, forward_pass_callback_args=calib_loader)
        
    # =========================================================================
    # [Evaluation 2] Pre-QAT Performance
    # =========================================================================
    acc_pre_qat = 0.0
    if val_loader:
        acc_pre_qat = evaluate_model(sim.model, val_loader, device)
        print(f">> Pre-QAT PSNR: {acc_pre_qat:.4f} dB")
        print(f">> Drop from FP32: {acc_pre_qat - acc_fp32:.4f} dB")

    # =========================================================================
    # [Step 4] Run QAT
    # =========================================================================
    print("\n" + "="*50)
    print(f" [Step 4] Running QAT ({args.qat_epochs} epochs)...")
    print(f"          LR: {args.qat_lr}, Train Samples: {subset_len}")
    print("="*50)

    # Setup Trainer for QAT
    qat_config = config.copy()
    qat_config['train']['lr'] = args.qat_lr
    qat_config['train']['epochs'] = args.qat_epochs
    
    # Configure scheduler to match epochs
    if 'scheduler' in qat_config['train']:
        if qat_config['train']['scheduler']['type'] == 'CosineAnnealingLR':
             qat_config['train']['scheduler']['T_max'] = args.qat_epochs
    
    qat_output_dir = os.path.join(args.output_dir, "qat_checkpoints")
    os.makedirs(qat_output_dir, exist_ok=True)
    
    # Create Trainer
    writer = SummaryWriter(log_dir=os.path.join(qat_output_dir, "logs"))
    trainer = Trainer(sim.model, qat_config, device, writer=writer, checkpoint_dir=qat_output_dir)
    
    # Run Training
    trainer.fit(train_loader, val_loader, epochs=args.qat_epochs)
    writer.close()
    
    # =========================================================================
    # [Evaluation 3] Post-QAT Performance
    # =========================================================================
    print("\n" + "="*50)
    print(" [Step 5] Evaluating Post-QAT Model...")
    print("="*50)
    
    acc_qat = 0.0
    if val_loader:
        acc_qat = evaluate_model(sim.model, val_loader, device)
        print(f">> Post-QAT PSNR: {acc_qat:.4f} dB")
        print(f">> Improvement from Pre-QAT: {acc_qat - acc_pre_qat:.4f} dB")
        print(f">> Drop from FP32: {acc_qat - acc_fp32:.4f} dB")

    # =========================================================================
    # [Step 6] Export
    # =========================================================================
    print(f"\n>> Exporting to {args.output_dir}...")
    sim.export(path=args.output_dir, filename_prefix="qat_model", dummy_input=dummy_input.cpu())

    # Final Summary
    print("\n" + "#"*60)
    print("                  FINAL SUMMARY                   ")
    print("#"*60)
    print(f" 1. FP32 Model PSNR       : {acc_fp32:.4f} dB")
    print(f" 2. Pre-QAT PSNR          : {acc_pre_qat:.4f} dB")
    print(f" 3. Post-QAT PSNR         : {acc_qat:.4f} dB")
    print("-" * 60)
    print(f" QAT Gain                 : +{acc_qat - acc_pre_qat:.4f} dB")
    print("#"*60)

if __name__ == "__main__":
    main()
