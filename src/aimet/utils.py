
import os
import random
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_model(model, loader, device, title="Model"):
    """
    Evaluate model performance (PSNR, SSIM) on a given loader.
    """
    if title:
        print(f"\n--- Evaluating {title} ---")
        
    model.eval()
    psnr_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            # Unified Dataset returns dict {'lr': ..., 'hr': ...}
            if isinstance(batch, dict):
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
            else:
                # Fallback for tuple
                lr, hr = batch
                lr = lr.to(device)
                hr = hr.to(device)
            
            # Inference
            sr = model(lr)
            if isinstance(sr, tuple): sr = sr[0]
            sr = torch.clamp(sr, 0.0, 1.0)
            
            # Iterate over batch to calculate metrics
            for i in range(sr.shape[0]):
                sr_img = sr[i].cpu().permute(1, 2, 0).numpy()
                hr_img = hr[i].cpu().permute(1, 2, 0).numpy()
                
                p = psnr(hr_img, sr_img, data_range=1.0)
                psnr_list.append(p)
            
    avg_psnr = np.mean(psnr_list)
    
    if title:
        print(f"[{title}] PSNR: {avg_psnr:.4f}")
        
    return avg_psnr

def save_active_results_to_csv(results, output_dir, filename="quant_results.csv"):
    """
    Save quantization analysis results to CSV.
    """
    if not results:
        print("[Warning] No results to save.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability if possible
    preferred_order = ['data_type', 'scheme', 'output_bw', 'param_bw', 'psnr', 'ssim']
    # Add any extra columns that might exist
    cols = preferred_order + [c for c in df.columns if c not in preferred_order]
    # Filter only existing columns
    cols = [c for c in cols if c in df.columns]
    
    df = df[cols]
    
    df.to_csv(csv_path, index=False)
    print(f"[CSV] Results saved to {csv_path}")

class AdaRoundDataLoader:
    """
    Wrapper for DataLoader to yield (input,) tuples for Adaround.
    Adaround expects the data loader to yield the inputs to the model.
    Since our dataset yields {'lr': ..., 'hr': ...}, we need to extract 'lr'.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, dict):
                # Yield ((input, ), dummy_target) 
                # AIMET default_forward_fn unpacks as: inputs, target = batch
                # Then calls: model(*inputs)
                yield (batch['lr'].to(self.device),), 0
            elif isinstance(batch, (list, tuple)):
                yield (batch[0].to(self.device),), 0
            else:
                yield (batch.to(self.device),), 0

    def __len__(self):
        return len(self.loader)

class AutoQuantDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        batch = self.dataset[index]
        return batch['lr']

    def __len__(self):
        return len(self.dataset)

def create_sampled_data_loader(dataset, num_samples, batch_size=32):
    """
    Create a DataLoader for a random subset of the dataset.
    """
    if num_samples > len(dataset):
        print(f"[Warning] Requested samples {num_samples} > dataset size {len(dataset)}. Using full dataset.")
        indices = range(len(dataset))
    else:
        indices = random.sample(range(len(dataset)), num_samples)
        
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

def apply_mmp_from_json(sim, config_path):
    """
    Apply Manual Mixed Precision (MMP) settings from a JSON file.
    
    JSON Format:
    {
        "layers": {
            "layer_name": {"input_bw": 8, "output_bw": 8},
            "body.0.act": {"output_bw": 8}
        }
    }
    """
    if not config_path or not os.path.exists(config_path):
        print(f"[MMP] Config file not found: {config_path}")
        return

    print(f"\n[MMP] Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        mmp_config = json.load(f)
        
    layers_config = mmp_config.get("layers", {})
    
    for layer_name, settings in layers_config.items():
        module = None
        # Attempt to find the module in the sim model
        try:
            # Handle nested modules (e.g., body.0.act)
            parts = layer_name.split('.')
            module = sim.model
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            print(f"   [Warning] Could not find module '{layer_name}' in model.")
            continue
            
        if module:
            print(f"   [MMP] Applying settings to '{layer_name}': {settings}")
            
            # Apply Output Bitwidth
            if "output_bw" in settings:
                bw = settings["output_bw"]
                if hasattr(module, 'output_quantizers') and module.output_quantizers:
                    if module.output_quantizers[0] is not None:
                        module.output_quantizers[0].bitwidth = bw
                        print(f"     -> Set Output BW: {bw}")
                        
            # Apply Input Bitwidth
            if "input_bw" in settings:
                bw = settings["input_bw"]
                if hasattr(module, 'input_quantizers') and module.input_quantizers:
                    if module.input_quantizers[0] is not None:
                        module.input_quantizers[0].bitwidth = bw
                        print(f"     -> Set Input BW: {bw}")
                        
            # Apply Param Bitwidth (Weight)
            if "param_bw" in settings:
                bw = settings["param_bw"]
                if hasattr(module, 'param_quantizers') and module.param_quantizers:
                    # 'weight' or 'bias'
                    if 'weight' in module.param_quantizers and module.param_quantizers['weight'] is not None:
                        module.param_quantizers['weight'].bitwidth = bw
                        print(f"     -> Set Param (weight) BW: {bw}")
                    if 'bias' in module.param_quantizers and module.param_quantizers['bias'] is not None:
                        module.param_quantizers['bias'].bitwidth = bw
                        print(f"     -> Set Param (bias) BW: {bw}")
    print("[MMP] Application Complete.\n")


# =============================================================================
# Monkey Patch for AIMET Channel Pruning (Winnowing)
# =============================================================================
try:
    # 1. AIMET Winnow Mask 모듈 가져오기
    import aimet_torch.common.winnow.mask as aimet_mask

    # 2. 원본 함수 백업 (중복 적용 방지)
    if not hasattr(aimet_mask.Mask, '_original_set_default_input_output_masks'):
        aimet_mask.Mask._original_set_default_input_output_masks = aimet_mask.Mask._set_default_input_output_masks

    # 3. 새로운 패치 함수 정의 
    # [중요] 라이브러리와 동일하게 인자를 (self, in_channels, out_channels)로 변경했습니다.
    def patched_set_default_input_output_masks(self, in_channels, out_channels):
        
        if self._op_type in ['DepthToSpace', 'PixelShuffle', 'AddOp']:
            # is_null_connectivity=False로 설정하면 'StopInternalConnectivity'가 생성됩니다.
            # (즉, 마스크 전파를 여기서 멈추라는 뜻)
            self._set_default_masks_for_null_and_stop_connectivity_ops(
                in_channels, out_channels, is_null_connectivity=False
            )
            return

        # [Patch 2] Hardtanh 처리
        # ReLU처럼 'Direct Connectivity'(입출력 1:1 매핑) 함수를 호출합니다.
        if self._op_type == 'Hardtanh':
            self._set_default_masks_for_direct_connectivity_ops(
                in_channels, out_channels
            )
            return

        # [Default] 그 외 연산자는 원본 로직 수행
        return aimet_mask.Mask._original_set_default_input_output_masks(self, in_channels, out_channels)

    aimet_mask.Mask._set_default_input_output_masks = patched_set_default_input_output_masks
    print("[Info] Monkey Patch Applied: Compatible with (in_channels, out_channels) signature.")

except ImportError as e:
    print(f"[Warning] Could not import aimet_torch.common.winnow.mask. Patch skipped. : {e}")
except Exception as e:
    print(f"[Warning] Failed to apply AIMET monkey patch: {e}")
    import traceback
    traceback.print_exc()

