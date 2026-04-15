import torch
import copy
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.model_preparer import prepare_model
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms

def prepare_model_for_quantization(model, input_shape, device='cuda'):
    """
    Prepare model for quantization:
    1. Prepare (Validate & Simplify)
    2. Batch Norm Folding
    3. Cross Layer Equalization (Optional, good for MobileNet/Depthwise layers)
    """
    print("[AIMET] Preparing model...")
    dummy_input = torch.randn(input_shape).to(device)
    
    # 1. Prepare
    model = prepare_model(model)
    
    # 2. Batch Norm Folding
    print("[AIMET] Folding Batch Norms...")
    _ = fold_all_batch_norms(model, input_shapes=input_shape)
    
    # 3. CLE (Optional - skipped for now as SR models usually don't have BN or complicated topology)
    # equalize_model(model, input_shapes=input_shape)
    
    return model

def create_quantsim(model, dummy_input, quant_scheme='tf_enhanced', default_output_bw=8, default_param_bw=8):
    """
    Create QuantizationSimModel
    """
    print(f"[AIMET] Creating QuantizationSimModel (Scheme: {quant_scheme}, BW: {default_output_bw}/{default_param_bw})")
    
    # Define Quantization Scheme (TF-Enhanced is generally good)
    if quant_scheme == 'tf_enhanced':
        from aimet_torch.common.defs import QuantScheme
        scheme = QuantScheme.post_training_tf_enhanced
    else:
        from aimet_torch.common.defs import QuantScheme
        scheme = QuantScheme.post_training_tf  # Standard Min-Max
        
    sim = QuantizationSimModel(model=model,
                               dummy_input=dummy_input,
                               quant_scheme=scheme,
                               default_output_bw=default_output_bw,
                               default_param_bw=default_param_bw)
    
    return sim

def calibrate_quantsim(sim, model, data_loader, num_batches=1, device='cuda'):
    """
    Calibrate the QuantizationSimModel using a subset of data
    """
    print(f"[AIMET] Calibrating with {num_batches} batches...")
    
    def pass_data(model, loader):
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                    
                # Unified Dataset returns dict {'lr': ..., 'hr': ...}
                if isinstance(batch, dict):
                    inputs = batch['lr'].to(device)
                else:
                    inputs = batch[0].to(device)
                    
                model(inputs)

    # Compute Encodings
    sim.compute_encodings(forward_pass_callback=pass_data, forward_pass_callback_args=data_loader)
    
    print("[AIMET] Calibration Done.")
    return sim

def export_quantsim(sim, save_dir, filename_prefix, dummy_input=None):
    """
    Export encodings and ONNX
    """
    print(f"[AIMET] Exporting to {save_dir}...")
    if dummy_input is None:
         # Fallback check, though likely to fail if model is GraphModule
        if hasattr(sim.model, 'dummy_input'):
            dummy_input = sim.model.dummy_input
        else:
            raise ValueError("dummy_input must be provided for export_quantsim as model does not have it.")
            
    sim.export(path=save_dir, filename_prefix=filename_prefix, dummy_input=dummy_input.cpu())
