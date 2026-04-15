import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from src.models import build_model
from src.engine.trainer import Trainer
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset
from src.aimet.quant_sim import prepare_model_for_quantization, create_quantsim, calibrate_quantsim, export_quantsim

def get_args():
    parser = argparse.ArgumentParser(description="Unified AIMET Quantization Simulation")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="aimet_export", help="Output directory")
    parser.add_argument("--calib_batches", type=int, default=100, help="Number of batches for calibration")
    parser.add_argument("--qat_epochs", type=int, default=0, help="Number of QAT epochs (0 to disable)")
    parser.add_argument("--qat_lr", type=float, default=1e-5, help="Learning rate for QAT")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def evaluate_model(model, loader, device, title="Model"):
    print(f"\n--- Evaluating {title} ---")
    model.eval()
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            
            # Inference
            sr = model(lr)
            if isinstance(sr, tuple): sr = sr[0]
            sr = torch.clamp(sr, 0.0, 1.0)
            
            # Convert to Numpy
            sr_np = sr.cpu().squeeze(0).permute(1, 2, 0).numpy()
            hr_np = hr.cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            # Calculate Metrics
            p = psnr(hr_np, sr_np, data_range=1.0)
            s = ssim(hr_np, sr_np, data_range=1.0, channel_axis=2)
            
            psnr_list.append(p)
            ssim_list.append(s)
            
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"[{title}] PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

def main():
    args = get_args()

    # ------------------------------- CPU or GPU -------------------------------
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
    
    # ------------------------------- Load Config -------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # ------------------------------- Build Model -------------------------------
    print(f"Loading model: {config['model']['name']}...")
    model = build_model(config['model']).to(device)
    
    # ------------------------------- Load Checkpoint -------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ------------------------------- Define Input Shape -------------------------------
    dummy_input = torch.randn(1, 3, 360, 640).to(device)
    
    # ------------------------------- Prepare Validation Loader for Evaluation -------------------------------
    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        print("Warning: validation paths not found in config. Skipping evaluation.")
        val_loader = None

    # ------------------------------- Baseline Evaluation -------------------------------
    fp32_psnr = fp32_ssim = 0.0
    if val_loader:
        print("Running Baseline Evaluation (Float32)...")
        fp32_psnr, fp32_ssim = evaluate_model(model, val_loader, device, title="Float32 Baseline")
    
    # ------------------------------- Prepare Model (BN Folding etc.) -------------------------------
    model = prepare_model_for_quantization(model, dummy_input.shape, device=device)
    
    # ------------------------------- Create QuantSim -------------------------------
    sim = create_quantsim(model, dummy_input, quant_scheme='tf_enhanced', default_output_bw=8, default_param_bw=8)
    
    # ------------------------------- Prepare Calibration/Training Data -------------------------------
    if config['task'] == 'sr':
        dataset = SRDataset(config['data']['train_root'], scale_factor=config['model']['scale'], patch_size=256, is_train=True)
    elif config['task'] == 'denoise':
        dataset = DenoiseDataset(config['data']['train_root'], patch_size=256, is_train=True)
    else:
        raise ValueError("Unknown task")
        
    train_loader = DataLoader(dataset, batch_size=config['train'].get('batch_size', 16), shuffle=True, num_workers=4, drop_last=True)
    
    # ------------------------------- Calibrate (PTQ) -------------------------------
    print("Starting Calibration...")
    sim = calibrate_quantsim(sim, model, train_loader, num_batches=args.calib_batches, device=device)
    
    # ------------------------------- PTQ Evaluation ------------------------------
    ptq_psnr = ptq_ssim = 0.0
    if val_loader:
        print("Running Quantized Evaluation (PTQ - Int8 Sim)...")
        ptq_psnr, ptq_ssim = evaluate_model(sim.model, val_loader, device, title="PTQ (Int8 Sim)")

    # ------------------------------- QAT (Optional) ------------------------------
    qat_psnr = qat_ssim = 0.0
    export_prefix = f"quant_{config['model']['name']}"
    
    if args.qat_epochs > 0:
        print(f"\n--- Starting QAT Finetuning ({args.qat_epochs} epochs) ---")
        
        qat_config = copy.deepcopy(config)
        qat_config['train']['lr'] = args.qat_lr
        
        # Override scheduler T_max if CosineAnnealingLR is used
        if qat_config['train'].get('scheduler', {}).get('type') == 'CosineAnnealingLR':
            print(f"[QAT] Overriding T_max to {args.qat_epochs} for CosineAnnealingLR")
            qat_config['train']['scheduler']['T_max'] = args.qat_epochs
        
        qat_checkpoint_dir = os.path.join(args.output_dir, "qat_checkpoints")
        
        import shutil
        with open(os.path.join(qat_checkpoint_dir, "train_config.yaml"), 'w') as f:
            yaml.dump(config, f)
        
        if config.get('data_config'):
             with open(os.path.join(qat_checkpoint_dir, "data_config.yaml"), 'w') as f:
                yaml.dump(config['data_config'], f)

        writer = SummaryWriter(log_dir=os.path.join(qat_checkpoint_dir, "logs"))

        trainer = Trainer(sim.model, qat_config, device, writer=writer, checkpoint_dir=qat_checkpoint_dir)
        trainer.fit(train_loader, val_loader, epochs=args.qat_epochs)

        writer.close()

        if val_loader:
            print("Running Quantized Evaluation (QAT - Int8 Sim)...")
            qat_psnr, qat_ssim = evaluate_model(sim.model, val_loader, device, title="QAT (Int8 Sim)")
            
        export_prefix = f"quant_qat_{config['model']['name']}"

    # ------------------------------- Summary ------------------------------
    print("\n=== Quantization Results ===")
    print(f"Float32: PSNR={fp32_psnr:.4f}, SSIM={fp32_ssim:.4f}")
    print(f"PTQ    : PSNR={ptq_psnr:.4f}, SSIM={ptq_ssim:.4f} (Drop: {fp32_psnr-ptq_psnr:.4f})")
    if args.qat_epochs > 0:
        print(f"QAT    : PSNR={qat_psnr:.4f}, SSIM={qat_ssim:.4f} (Drop: {fp32_psnr-qat_psnr:.4f})")

    # ------------------------------- Export ------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    export_quantsim(sim, args.output_dir, filename_prefix=export_prefix, dummy_input=dummy_input)
    
    print("AIMET Simulation Complete.")

if __name__ == "__main__":
    main()
