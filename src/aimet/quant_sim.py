"""Single entry point for QuantSim construction. Every AIMET script calls only this.

The transform order defines the quantizer graph; change it and .encodings/.onnx
silently stop matching. Order:
switch_to_deploy -> prepare_model -> eval -> fold_all_batch_norms -> equalize_model -> QuantizationSimModel
For an IR (achromatic) Bicubic++ model, switch_to_deploy also folds the channel mean into
the last layer, so every script that calls it gets the shipped graph.
"""
from pathlib import Path

from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.model_preparer import prepare_model
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms

# NOT the builtin "htp_v73": the builtin fuses Conv+LeakyRelu (3 supergroups) and thus
# omits Conv output encodings, dropping 6 of 8 Bicubic++ (LeakyReLU 0.1) ops to float in
# the DLC. This repo config is builtin htp_v73 minus exactly those 3 LeakyRelu supergroups
# (11 vs 14); Conv+Clip is kept, so Hardtanh models get an identical quantizer set.
DEFAULT_QUANTSIM_CONFIG = str(
    Path(__file__).resolve().parents[2]
    / "configs" / "aimet" / "htp_v73_no_leakyrelu_supergroup.json"
)


def prepare_deploy_model(model):
    """switch_to_deploy (reparameterization; idempotent) -> prepare_model -> eval."""
    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    model = prepare_model(model)
    model.eval()
    return model


def fold_and_equalize(model, dummy_input, apply_cle=True):
    """BN fold + (optional) CLE. Both receive dummy_input, exactly as 5/6/7 did."""
    _ = fold_all_batch_norms(model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
    if apply_cle:
        _ = equalize_model(model, input_shapes=dummy_input.shape, dummy_input=dummy_input)
    return model


def build_quantsim(model, dummy_input, *, quant_scheme="post_training_tf_enhanced",
                   default_output_bw=8, default_param_bw=8,
                   config_file=DEFAULT_QUANTSIM_CONFIG,
                   prepare=True, fold=True, apply_cle=True, **quantsim_kwargs):
    """Trained model -> QuantizationSimModel.

    prepare=False: caller already ran prepare_deploy_model (FP32 eval interleaved, scripts 5/6).
    fold=False: caller already ran fold_and_equalize (script 3 loop, script 4 export sim).
    quantsim_kwargs are forwarded verbatim (in_place, default_data_type, ...).
    """
    if isinstance(quant_scheme, str):
        quant_scheme = getattr(QuantScheme, quant_scheme)
    if prepare:
        model = prepare_deploy_model(model)
    if fold:
        model = fold_and_equalize(model, dummy_input, apply_cle=apply_cle)
    return QuantizationSimModel(model=model,
                                dummy_input=dummy_input,
                                quant_scheme=quant_scheme,
                                default_output_bw=default_output_bw,
                                default_param_bw=default_param_bw,
                                config_file=config_file,
                                **quantsim_kwargs)
