import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


import os
import sys
import argparse
import yaml
import torch
import numpy as np
import math
import decimal
import copy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.aimet.utils import evaluate_model, create_sampled_data_loader, AutoQuantDatasetWrapper
from src.models import build_model
from src.engine.trainer import Trainer
from src.data.datasets import PairedDataset, SRDataset, DenoiseDataset


# AIMET Imports
from aimet_torch.compress import ModelCompressor
from aimet_torch.common.defs import CompressionScheme, CostMetric
from aimet_torch.defs import GreedySelectionParameters, SpatialSvdParameters, ChannelPruningParameters, ModuleCompRatioPair


### Only Channel Pruning
# python3 examples/2_compress_svd_pruning.py --config configs/finetune/quicksrnet_ir.yaml --data_config configs/data/sr_finetune_ir.yaml --checkpoint checkpoints/finetune_quicksrnet_large_ir_sr_x2_dim64_epoch100_bs_32_ga_2_lr_1e-4/best.pth --output_dir results/20260220_quicksrnet-large_ir_prune50 --width 640 --height 360 --use_pruning --target_ratio 0.5 --num_comp_ratio_candidates 10 --calib_batches 500 --ft_epochs 30 --ft_lr 5e-5 --dataset_ratio 0.5 --device 0 --ignore_layers cnn.0 clip_output add_op anchor.net pixel_shuffle


def get_args():
    parser = argparse.ArgumentParser(description="Spatial SVD + Channel Pruning Compression")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file")
    parser.add_argument("--data_config", type=str, default="configs/data/sr.yaml", help="Data config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to float32 checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/compression", help="Output directory")
    
    parser.add_argument("--width", type=int, default=640, help="Width of dummy input")
    parser.add_argument("--height", type=int, default=360, help="Height of dummy input")
    
    # Use Compression Mode
    parser.add_argument("--use_svd", action="store_true", help="Use Spatial SVD compression (사용하지 않음 -> 추론속도 지연 확인)")
    
    parser.add_argument("--use_pruning", action="store_true", help="Use Channel Pruning compression")
    parser.add_argument("--ignore_layers", nargs='+', default=[], help="List of layer names (substrings) to ignore during pruning")

    # Compression Params
    parser.add_argument("--target_ratio", type=float, default=0.1, help="Target compression ratio (e.g., 0.5 for 50%% MACs)")
    parser.add_argument("--num_comp_ratio_candidates", type=int, default=10, help="Number of compression ratio candidates for greedy selection")
    parser.add_argument("--calib_batches", type=int, default=500, help="Number of batches for evaluation/calibration (Default: 500)")
    
    # Fine-tuning Params
    parser.add_argument("--ft_epochs", type=int, default=20, help="Number of fine-tuning epochs per stage")
    parser.add_argument("--ft_lr", type=float, default=1e-5, help="Fine-tuning learning rate")
    parser.add_argument("--dataset_ratio", type=float, default=0.4, help="Ratio of training dataset to use for fine-tuning (0.0 ~ 1.0)")
    
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def generate_generalized_crouton_ratios(model, eval_callback, ignore_layers, target_ratio=0.5):
    """
    모든 모델(YOLO 등)에 범용적으로 적용 가능한 HTP(Crouton) 친화적 비율 탐색기.
    각 레이어의 채널을 32의 배수 단위로만 줄여가며 최적의 압축 비율 리스트를 반환합니다.
    """
    print("\n[Info] --- Generalized Crouton-Aware Greedy Search 시작 ---")
    
    # 1. Pruneable 레이어 탐색 (out_channels가 32보다 큰 Conv2d만 대상)
    layer_info = {}
    total_channels = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if any(ig in name for ig in ignore_layers):
                continue
            if module.out_channels > 32:
                # 초기 상태: 깎인 채널 수 0, 원래 채널 수 저장
                layer_info[name] = {
                    'module': module,
                    'orig_channels': module.out_channels,
                    'current_channels': module.out_channels
                }
                total_channels += module.out_channels

    print(f"[Info] 대상 레이어 수: {len(layer_info)}개")
    baseline_score = eval_callback(model)
    print(f"[Info] Baseline PSNR: {baseline_score:.4f}")

    # 목표로 하는 전체 남은 채널 수 (대략적인 압축 목표치)
    target_total_channels = int(total_channels * target_ratio)
    
    # 2. Greedy Search: 목표치에 도달할 때까지 32채널씩 깎아내리기
    while sum(info['current_channels'] for info in layer_info.values()) > target_total_channels:
        best_layer_to_prune = None
        min_psnr_drop = float('inf')
        
        # 각 레이어마다 32채널을 한 번 더 깎아보고(시뮬레이션) 민감도 측정
        for name, info in layer_info.items():
            if info['current_channels'] <= 32: # 32채널 이하면 더 이상 깎지 않음
                continue
                
            module = info['module']
            orig_weight = module.weight.data.clone()
            
            # 현재 상태에서 가장 덜 중요한 32개 필터 마스킹 (L1 Norm 기준)
            l1_norms = torch.norm(orig_weight.view(orig_weight.size(0), -1), p=1, dim=1)
            # 이미 깎인 채널을 제외하고 추가로 32개를 더 깎았을 때의 민감도
            num_to_mask = info['orig_channels'] - info['current_channels'] + 32
            _, threshold_indices = torch.topk(l1_norms, num_to_mask, largest=False)
            
            temp_weight = orig_weight.clone()
            temp_weight[threshold_indices] = 0.0
            module.weight.data = temp_weight
            
            # 평가
            score = eval_callback(model)
            drop = baseline_score - score
            
            # 복구
            module.weight.data = orig_weight
            
            # 가장 화질 하락이 적은(안전한) 레이어 탐색
            if drop < min_psnr_drop:
                min_psnr_drop = drop
                best_layer_to_prune = name
                
        # 깎을 레이어가 더 이상 없으면 종료
        if best_layer_to_prune is None:
            break
            
        # 선택된 레이어의 채널을 32만큼 영구히 삭감(업데이트)
        layer_info[best_layer_to_prune]['current_channels'] -= 32
        print(f"  -> [Step] {best_layer_to_prune} 레이어 32채널 삭감 (현재: {layer_info[best_layer_to_prune]['current_channels']} / {layer_info[best_layer_to_prune]['orig_channels']}) | 예상 PSNR 하락: {min_psnr_drop:.4f}")

    # 3. AIMET ManualMode용 비율 리스트 생성
    print("\n[Info] --- 최종 도출된 Layer별 압축 비율 ---")
    layer_comp_ratio_list = []
    
    # 전체 네트워크의 모든 모듈 순회 (AIMET은 대상이 아닌 레이어도 리스트에 포함을 요구할 수 있음)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and not any(ig in name for ig in ignore_layers):
            if name in layer_info:
                info = layer_info[name]
                # 원래 채널 대비 현재 결정된 채널의 비율 계산
                ratio = info['current_channels'] / info['orig_channels']
            else:
                ratio = 1.0 # 대상이 아니거나 너무 작은 레이어는 1.0 보존
                
            # layer_comp_ratio_list.append((name, float(ratio)))
            layer_comp_ratio_list.append(ModuleCompRatioPair(module, float(ratio)))
            print(f"  {name}: {ratio:.4f} ({int(module.out_channels * ratio)} channels)")
    print('='*60)
    return layer_comp_ratio_list

def finetune(model, config, train_loader, val_loader, epochs, lr, output_dir, device):
    if epochs <= 0:
        return
        
    print(f"\n--- Starting Fine-tuning ({epochs} epochs, LR={lr}) ---")
    
    ft_config = copy.deepcopy(config)
    ft_config['train']['lr'] = lr
    ft_config['train']['epochs'] = epochs
    
    # Override scheduler T_max if CosineAnnealingLR is used
    if ft_config['train'].get('scheduler', {}).get('type') == 'CosineAnnealingLR':
        ft_config['train']['scheduler']['T_max'] = epochs
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dump configs for reference
    with open(os.path.join(output_dir, "train_config.yaml"), 'w') as f:
        yaml.dump(ft_config, f)
    
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    trainer = Trainer(model, ft_config, device, writer=writer, checkpoint_dir=output_dir)
    trainer.fit(train_loader, val_loader, epochs=epochs)
    
    writer.close()
    
class DeviceDataLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        
    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, dict):
                # AIMET requires tuple/list (inputs, targets)
                # We yield (lr, hr)
                lr = batch['lr'].to(self.device)
                hr = batch['hr'].to(self.device)
                yield (lr, hr)
            elif isinstance(batch, (list, tuple)):
                yield [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in batch]
            else:
                # AIMET requires list or tuple
                yield [batch.to(self.device)]
                
    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
         # Proxy attribute access to the underlying loader
         # e.g., batch_size, dataset, etc.
         return getattr(self.loader, name)

def get_train_loader(config, args):
    print(f"Loading training dataset (Task: {config['task']})...")
    if config['task'] == 'sr':
        full_ds = SRDataset(dataset_root=config['data']['train_root'],
                        scale_factor=config['model']['scale'],
                        patch_size=config['train']['patch_size'],
                        is_train=True,
                        config=config.get('data_config', {}))
    elif config['task'] == 'denoise':
        full_ds = DenoiseDataset(dataset_root=config['data']['train_root'],
                            patch_size=config['train']['patch_size'],
                            is_train=True,
                            config=config.get('data_config', {}))
    else:
        raise ValueError(f"Unknown task: {config['task']}")

    # Subsampling logic
    if args.dataset_ratio < 1.0:
        total_len = len(full_ds)
        subset_len = int(total_len * args.dataset_ratio)
        unused_len = total_len - subset_len
        print(f"[Info] Subsampling dataset: {args.dataset_ratio*100}% ({subset_len} samples out of {total_len})")
        
        subset_ds, _ = torch.utils.data.random_split(full_ds, [subset_len, unused_len], 
                                                     generator=torch.Generator().manual_seed(42))
        train_ds = subset_ds
    else:
        print(f"[Info] Using full dataset ({len(full_ds)} samples)")
        train_ds = full_ds
        
    loader = DataLoader(train_ds, batch_size=config['train'].get('batch_size', 16), 
                        shuffle=True, num_workers=4, drop_last=True)
    return loader

def main():
    args = get_args()

    # 1. Device
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device.isdigit():
        if torch.cuda.is_available():
            # Set default device to ensure operations default to this
            torch.cuda.set_device(int(args.device))
            device = torch.device(f"cuda:{args.device}")
        else:
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
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    # Switch to deploy if possible (for faster inference/realistic evaluation)
    model.eval()
    if hasattr(model, 'switch_to_deploy'):
        print("Switching model to deploy mode...")
        model.switch_to_deploy()
    
    # Validation Loader (for Reporting PSNR)
    val_hr = config['data'].get('val_hr_root')
    val_lr = config['data'].get('val_lr_root')
    if val_hr and val_lr:
        val_dataset = PairedDataset(val_hr, val_lr)
        # Full validation set for reporting
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        print(f"[Info] Validation loader created ({len(val_dataset)} samples)")
    else:
        print("Warning: validation paths not found. Cannot evaluate PSNR.")
        val_loader = None

    # Calibration Loader (for Greedy Selection - Subset of Training Data)
    print(f"Creating Calibration Loader ({args.calib_batches} samples from Training Data)...")
    if config['task'] == 'sr':
        calib_ds = SRDataset(dataset_root=config['data']['train_root'],
                        scale_factor=config['model']['scale'],
                        patch_size=config['train']['patch_size'],
                        is_train=True,
                        config=config.get('data_config', {}))
    elif config['task'] == 'denoise':
        calib_ds = DenoiseDataset(dataset_root=config['data']['train_root'],
                            patch_size=config['train']['patch_size'],
                            is_train=True,
                            config=config.get('data_config', {}))
    
    calib_loader = create_sampled_data_loader(calib_ds, args.calib_batches, batch_size=1)
    print(f"[Info] Calibration loader created ({len(calib_loader)} samples)")

    # Define Callback for AIMET (uses calib_loader)
    def eval_callback(model, iterations=None, use_cuda=True):
        device = next(model.parameters()).device
        # Use calibration loader (Training Subset) for greedy selection scoring!
        avg_score = evaluate_model(model, val_loader, device, title=None)
        return avg_score

    dummy_input = torch.randn(1, 3, args.height, args.width).to(device)

    psnr_fp32 = evaluate_model(model, val_loader, device, title="FP32 Baseline")

    # 4. Prepare Model
    stage_target = args.target_ratio
    print(f">> Target Compression Ratio: {1 -args.target_ratio:.4f}")
    
    if args.use_svd:
        print(f">> Stage 1 (SVD) Target: {1 -stage_target:.4f}")
    if args.use_pruning:
        print(f">> Stage 2 (CP) Target:  {1 - stage_target:.4f}")
    
    # ---------------------------------------------------------
    # Stage 1: Spatial SVD
    # ---------------------------------------------------------
    if args.use_svd:
        print(f"\n[Stage 1] Spatial SVD Compression...")
    
        svd_greedy_params = GreedySelectionParameters(target_comp_ratio=decimal.Decimal(1 - stage_target),
                                                  num_comp_ratio_candidates=args.num_comp_ratio_candidates)
    
        # Use auto mode
        svd_auto_params = SpatialSvdParameters.AutoModeParams(svd_greedy_params, modules_to_ignore=[])
        svd_params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto, params=svd_auto_params)
        eval_iterations = 10
        compress_scheme = CompressionScheme.spatial_svd
        cost_metric = CostMetric.mac
    
        compressed_model_svd, stats_svd = ModelCompressor.compress_model(
            model=model,
            eval_callback=eval_callback,
            eval_iterations=eval_iterations, # eval_callback handles iteration control or usage
            input_shape=(1, 3, args.height, args.width),
            compress_scheme=compress_scheme,
            cost_metric=cost_metric,
            parameters=svd_params
        )
        
        print(f"   >> SVD Stats: {stats_svd}")
        print("   >> Evaluating SVD Model...")
        psnr_svd = evaluate_model(compressed_model_svd, val_loader, device, title="SVD Compressed")
        
        # Fine-tune SVD Model
        if args.ft_epochs > 0:
            if config['task'] == 'sr':
                ds = SRDataset(dataset_root=config['data']['train_root'],
                                scale_factor=config['model']['scale'],
                                patch_size=config['train']['patch_size'],
                                is_train=True,
                                config=config.get('data_config', {}))
            elif config['task'] == 'denoise':
                ds = DenoiseDataset(dataset_root=config['data']['train_root'],
                                    patch_size=config['train']['patch_size'],
                                    is_train=True,
                                    config=config.get('data_config', {}))
            
            # ft_train_loader = DataLoader(ds, batch_size=config['train'].get('batch_size', 16), 
            #                             shuffle=True, num_workers=4, drop_last=True)
            ft_train_loader = get_train_loader(config, args)
                                        
            ft_output_dir = os.path.join(args.output_dir, "finetune_svd")
            finetune(compressed_model_svd, config, ft_train_loader, val_loader, 
                    args.ft_epochs, args.ft_lr, ft_output_dir, device)
                    
            # Re-evaluate after FT
            psnr_svd_ft = evaluate_model(compressed_model_svd, val_loader, device, title="SVD Compressed + FT")

    if args.use_svd:
        final_model = compressed_model_svd
    else:
        final_model = model

    # ---------------------------------------------------------
    # Stage 2: Channel Pruning
    # ---------------------------------------------------------
    if args.use_pruning:
        print(f"\n[Stage 2] Channel Pruning...")
    
        # Validation Loader (for Greedy Selection Eval)
        if not val_loader:
             print("Warning: validation paths not found. Cannot evaluate.")
             return
        
        # CP requires a DeviceDataLoader that yields inputs.
        # But we are using AutoMode with eval_callback.
        # ChannelPruningParameters expects `data_loader` argument for reconstruction/estimation?
        # Actually, `data_loader` in `ChannelPruningParameters` is used for "Input-Output Reconstruction".
        # This MUST be the calibration loader (Training Data subset).
        
        cp_loader = DeviceDataLoader(calib_loader, device)
        
        # [Support] Manual ignoring of layers
        modules_to_ignore = []
        if args.ignore_layers:
            print(f"[Info] Ignoring layers matching: {args.ignore_layers}")
            for name, module in final_model.named_modules():
                for ignore_pattern in args.ignore_layers:
                    if ignore_pattern in name and isinstance(module, (nn.Conv2d, nn.Linear, nn.PixelShuffle, nn.Hardtanh)):
                        modules_to_ignore.append(module)
                        print(f"   - Ignored: {name}")

        manual_layer_ratios = generate_generalized_crouton_ratios(
            model=final_model,
            eval_callback=eval_callback,
            ignore_layers=args.ignore_layers,
            target_ratio=stage_target  # 예: 0.5 또는 0.75
        )

        cp_manual_params = ChannelPruningParameters.ManualModeParams(
            list_of_module_comp_ratio_pairs=manual_layer_ratios
        )
        
        cp_params = ChannelPruningParameters(
            data_loader=cp_loader,
            num_reconstruction_samples=args.calib_batches,
            allow_custom_downsample_ops=False,
            mode=ChannelPruningParameters.Mode.manual,  # <== Manual 모드로 변경
            params=cp_manual_params                     # <== 생성한 리스트 주입
        )

        # cp_greedy_params = GreedySelectionParameters(target_comp_ratio=decimal.Decimal(1 - stage_target),
        #                                             num_comp_ratio_candidates=args.num_comp_ratio_candidates)
                                                    
        # cp_auto_params = ChannelPruningParameters.AutoModeParams(cp_greedy_params, modules_to_ignore=modules_to_ignore)
        
        # cp_params = ChannelPruningParameters(data_loader=cp_loader,
        #                                     num_reconstruction_samples=args.calib_batches, # Match loader size
        #                                     allow_custom_downsample_ops=False,
        #                                     mode=ChannelPruningParameters.Mode.auto,
        #                                     params=cp_auto_params)
        eval_iterations = 10
        compress_scheme = CompressionScheme.channel_pruning
        cost_metric = CostMetric.mac
                                            
        compressed_model_final, stats_cp = ModelCompressor.compress_model(
            model=final_model,
            eval_callback=eval_callback,
            eval_iterations=eval_iterations,
            input_shape=(1, 3, args.height, args.width),
            compress_scheme=compress_scheme,
            cost_metric=cost_metric,
            parameters=cp_params
        )
        
        print(f"   >> CP Stats: {stats_cp}")
        print("   >> Evaluating Final Model...")
        psnr_cp = evaluate_model(compressed_model_final, val_loader, device, title="SVD+CP Compressed")
        
        # Fine-tune Final Model
        if args.ft_epochs > 0:
            # Reuse ft_train_loader if created, else create
            # But variable scope? Let's recreate/ensure it exists.
            if 'ft_train_loader' not in locals():
                ft_train_loader = get_train_loader(config, args)
            
            ft_output_dir = os.path.join(args.output_dir, "finetune_final")
            finetune(compressed_model_final, config, ft_train_loader, val_loader, 
                    args.ft_epochs, args.ft_lr, ft_output_dir, device)
                    
            # Re-evaluate
            psnr_cp_ft = evaluate_model(compressed_model_final, val_loader, device, title="SVD+CP Compressed + FT")
    
    if args.use_pruning:
        final_model = compressed_model_final

    # Save Model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "compressed_model.pth")
    torch.save(final_model, save_path)

    print(f"\n[Success] Compressed model saved to {save_path}")
    print(f"Initial PSNR: (N/A - Run baseline script)")
    print(f"FP32 PSNR:    {psnr_fp32:.4f}")
    if args.use_svd:
        print(f"SVD PSNR:     {psnr_svd:.4f}")
        print(f"SVD+FT PSNR:  {psnr_svd_ft:.4f}")
    if args.use_pruning:
        print(f"CP PSNR:      {psnr_cp:.4f}")
        print(f"CP+FT PSNR:   {psnr_cp_ft:.4f}")

if __name__ == "__main__":
    main()
