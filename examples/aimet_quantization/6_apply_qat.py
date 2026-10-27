
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.engine.checkpoint import check_model_config
from src.models import build_model
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset
from src.aimet.utils import evaluate_model, create_sampled_data_loader, AutoQuantDatasetWrapper, apply_mmp_from_json
from src.engine.trainer import Trainer

# AIMET Imports
from aimet_torch.quantsim import load_encodings_to_sim
from aimet_common.defs import QuantScheme

from src.aimet.quant_sim import DEFAULT_QUANTSIM_CONFIG, prepare_deploy_model, build_quantsim

# Shipped W16A16 PTQ export (same flags as results/denoise_release_v1/export.sh):
# NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=. CUDA_VISIBLE_DEVICES= python3 examples/aimet_quantization/6_apply_qat.py \
#     --config results/denoise_release_v1/checkpoint/train_config.yaml \
#     --data_config results/denoise_release_v1/checkpoint/data_config.yaml \
#     --checkpoint results/denoise_release_v1/checkpoint/best.pth --output_dir <out_dir>/qat \
#     --param_bw 16 --output_bw 16 --mmp_config configs/aimet/mmp_noprune_w16a16.json \
#     --qat_epochs 0 --calib_batches 500 --width 1280 --height 720 --device cpu
# IR SR: results/release_v3/eval/export.sh. Then examples/deployment/gen_convert_all.py.

def get_args():
    parser = argparse.ArgumentParser(description="QAT Application")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data_config", type=str, required=True, help="Data config of the recipe; calibration draws from it")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/qat", help="Output directory")
    
    # Quantization Params
    parser.add_argument("--scheme", type=str, default="training_range_learning_with_tf_init", 
                        choices=["post_training_tf", "post_training_tf_enhanced", "training_range_learning_with_tf_init"],
                        help="Quantization Scheme (Default: training_range_learning_with_tf_init for QAT)")
    parser.add_argument("--param_bw", type=int, default=8, help="Default Parameter Bitwidth")
    parser.add_argument("--output_bw", type=int, default=8, help="Default Output Bitwidth")
    parser.add_argument("--quantsim_config", type=str, default=DEFAULT_QUANTSIM_CONFIG,
                        help="AIMET quantsim config: builtin name (e.g. htp_v73) or path to a JSON file. "
                             "Use a config without the Conv+LeakyRelu supergroup when the target converter "
                             "does not fuse that pair, otherwise the Conv output tensors get no encodings "
                             "and fall back to float during DLC conversion.")
    
    # QAT Params
    # 0 means calibrate and export, no training: the checkpoint's weights ship
    # as they are. That is the right default at W16A16, where Pre-QAT already
    # matches FP32 to 1e-4 (measured on every v3 run) and there is no gap for
    # training to close. Training only moves the model -- on 2026-09-10 the
    # 30-epoch default, restarting at qat_lr from a Phase-2 that had decayed
    # to 1e-7 and re-running the loss warm-up from epoch 0, spent its first
    # epoch on pure MSE and shipped a model that scored higher on the
    # synthetic val set and worse on real IR frames (x3: 0.798 -> 0.737 of the
    # anchor's sharpness, ringing 1.86% -> 3.47%). Set this above 0 for W8A8,
    # where the quantization gap is real and QAT is what recovers it.
    parser.add_argument("--qat_epochs", type=int, default=0,
                        help="QAT epochs. 0 = calibrate + export only (default; W16A16). >0 for W8A8.")
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

def _check_achromatic_fold(model, cfg):
    """Stop the export if an achromatic (IR) model still averages its channels in forward.

    BicubicPlusPlus.switch_to_deploy folds the mean into the last layer. Left in, the average
    exports as six ops over the full output that fall back to float on HTP (x3: 2.746 ms
    without them, 10.401 ms with, at W8A8), so this checks the graph that is about to ship.
    """
    if not cfg.get('model', {}).get('achromatic_output', False):
        return
    if getattr(model, 'achromatic_output', False):
        raise RuntimeError("achromatic_output is still on after switch_to_deploy; the export would keep the channel mean")
    dev = next(model.parameters()).device
    x = torch.rand(1, 1, 64, 64, device=dev).repeat(1, 3, 1, 1)
    with torch.no_grad():
        y = model(x)
        y = y[0] if isinstance(y, tuple) else y
    drift = (y - y.mean(1, keepdim=True)).abs().max().item() * 255
    if drift > 1e-3:
        raise RuntimeError(f"achromatic fold failed: {drift:.6f} DN of chroma remains")
    print(f"[Info] Achromatic output folded into the final layer "
          f"(residual chroma {drift:.6f} DN); the averaging op stays out of the graph.")

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

    # AIMET allocates some quantizer state on the current CUDA device rather than the
    # module's device, so exporting from a non-zero GPU fails with a device mismatch.
    # Pin the current device to keep every newly created tensor on the same GPU.
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # A missing recipe must stop the export: an empty data config would calibrate on every
    # degradation at its default instead of on the recipe the model was trained with.
    with open(args.data_config, 'r') as f:
        config['data_config'] = yaml.safe_load(f)

    # 2. Load Model & Checkpoint
    print(f"Loading model: {config['model']['name']}...")
    model = build_model(config['model']).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    check_model_config(checkpoint, config['model'], args.checkpoint)
    
    if isinstance(checkpoint, torch.nn.Module):
        print("[Info] Loaded full model object.")
        model = checkpoint.to(device)
    else:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)

    # switch_to_deploy folds the ECB branches and then, for an IR model, the channel
    # mean into the last layer (BicubicPlusPlus._fold_achromatic). It is idempotent,
    # so the call prepare_deploy_model makes below is a no-op.
    # Order that results: ECB fold -> achromatic fold -> prepare_model -> QuantSim.
    if hasattr(model, 'switch_to_deploy'):
        model.switch_to_deploy()
    _check_achromatic_fold(model, config)

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

    # 5. Initialize QuantizationSimModel
    print(f"\n>> Initializing QuantizationSimModel (Base: W{args.param_bw}A{args.output_bw})...")
    print(f">> Quantsim config: {args.quantsim_config}")
    quant_scheme = getattr(QuantScheme, args.scheme)
    
    # fold+CLE run inside build_quantsim, at this same point in the flow --
    # after the FP32 eval, which ran on the prepared-but-unfolded model as before.
    sim = build_quantsim(model, dummy_input,
                         quant_scheme=quant_scheme,
                         default_output_bw=args.output_bw,
                         default_param_bw=args.param_bw,
                         config_file=args.quantsim_config,
                         prepare=False)

    # =========================================================================
    # [Step 2] Apply Manual Mixed Precision (MMP) from --mmp_config
    # =========================================================================
    print("\n" + "="*50)
    print(" [Step 2] Applying Manual Mixed Precision (MMP)...")
    print("="*50)
    
    if args.mmp_config:
        apply_mmp_from_json(sim, args.mmp_config)
    else:
        print(f"[Info] No MMP config: every layer stays at W{args.param_bw}A{args.output_bw}.")

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

    # QAT 0 epochs ships the calibrated model as it is, so no Trainer: building one
    # loaded the softening net for EO configs and restored any best.pth left in a
    # reused qat_checkpoints/ over the model about to be exported.
    if args.qat_epochs > 0:
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

        best_checkpoint_path = os.path.join(qat_output_dir, "best.pth")
        if os.path.isfile(best_checkpoint_path):
            print(f">> Restoring best QAT checkpoint: {best_checkpoint_path}")
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            sim.model.load_state_dict(best_checkpoint["model_state_dict"])
        else:
            print(">> Best QAT checkpoint not found. Using the final epoch state.")
    
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
    else:
        print(">> --qat_epochs 0: no QAT; the calibrated model is exported as it is.")
        acc_qat = acc_pre_qat

    # =========================================================================
    # [Step 6] Export
    # =========================================================================
    print(f"\n>> Exporting to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
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
