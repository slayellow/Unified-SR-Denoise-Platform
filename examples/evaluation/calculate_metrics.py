
import torch
from thop import profile
from src.models.quicksrnet import QuickSRNetLarge
from src.models.svsrnet import SVSRNet
from src.models.rrdbnet import RRDBNet

def get_model_metrics(model, input_size=(1, 3, 360, 640)):
    input_tensor = torch.randn(input_size)
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    # Convert to GFLOPs (1 MAC = 2 FLOPs usually, but often reported as GFLOPs = GMACs in some contexts. 
    # QCS8550 specs usually refer to TOPS. GFLOPs is a reasonable proxy.
    # We will report GMACs and Params(M)
    gmacs = macs / 1e9
    params_m = params / 1e6
    return gmacs, params_m

def main():
    print(f"{'Model':<20} | {'Params (M)':<10} | {'GMACs (640x360)':<15}")
    print("-" * 55)

    # 1. QuickSRNet Large
    # Config: scale=2, dim=64
    qs_model = QuickSRNetLarge(scaling_factor=2, dim=64)
    qs_params, qs_gmacs = get_model_metrics(qs_model)
    print(f"{'QuickSRNet Large':<20} | {qs_params:<10.2f} | {qs_gmacs:<15.2f}")

    # 2. SVSRNet
    # Config: scale=2, dim=64. Defaults: n_resblocks=12 from config assumption (standard)
    # Checking svsrnet.py default is 12. Config doesn't specify explicitly but implies default or matches typical setup.
    # Let's assume n_resblocks=12 based on code default.
    sv_model = SVSRNet(scaling_factor=2, n_resblocks=12, n_feats=64)
    # SVSRNet has 'switch_to_deploy' which merges blocks. We should measure deploy mode!
    sv_model.eval()
    sv_model.switch_to_deploy()
    sv_params, sv_gmacs = get_model_metrics(sv_model)
    print(f"{'SVSRNet (Deploy)':<20} | {sv_params:<10.2f} | {sv_gmacs:<15.2f}")

    # 3. RRDBNet
    # Config: dim=32, num_block=3, num_grow_ch=16. scale=2.
    rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=32, num_block=3, num_grow_ch=16)
    rrdb_params, rrdb_gmacs = get_model_metrics(rrdb_model)
    print(f"{'RRDBNet':<20} | {rrdb_params:<10.2f} | {rrdb_gmacs:<15.2f}")

if __name__ == "__main__":
    main()
