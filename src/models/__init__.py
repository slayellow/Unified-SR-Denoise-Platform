from .quicksrnet import QuickSRNetSmall, QuickSRNetMedium, QuickSRNetLarge, QuickDenoiseNet, QuickDenoiseOpt, LRCSR as QuickLRCSR
from .lrcsr import LRCSR
from .svsrnet import SVSRNet
from .ddrnet import DDRNet
from .rrdbnet import RRDBNet
from .qcsawaresrnet import QCSAwareSRNetSmall, QCSAwareSRNetMedium, QCSAwareSRNetLarge
from .svfocussrnet import SVFocusSRNet
from .mambair import MambaIR
from .mambairv2 import MambaIRv2
from .corefusion import CoReFusion
from .lapgsr import LapGSR, LapGSRDiscriminator

__all__ = [
    "QuickSRNetSmall", "QuickSRNetMedium", "QuickSRNetLarge",
    "QuickDenoiseNet", "QuickDenoiseOpt",
    "LRCSR", "SVSRNet", "DDRNet", "RRDBNet",
    "QCSAwareSRNetSmall", "QCSAwareSRNetMedium", "QCSAwareSRNetLarge",
    "SVFocusSRNet", "MambaIR", "MambaIRv2",
    "CoReFusion", "LapGSR", "LapGSRDiscriminator",
    "build_model"
]

def build_model(config):
    """
    Factory function to create a model based on configuration.
    
    Args:
        config (dict or Namespace): Must contain 'name' and model-specific args.
    
    Returns:
        nn.Module
    """
    # config가 dict가 아니면 (Namespace 등) dict로 변환 시도
    if not isinstance(config, dict):
        if hasattr(config, '__dict__'):
            config = config.__dict__
        else:
            # Fallback (Just try to access attributes if needed, but dict is safer)
            pass

    name = config.get('name', '').lower()
    
    # Common Args
    scale = config.get('scale', 2)
    dim = config.get('dim', None) # If None, use default of class
    
    kwargs = {'scaling_factor': scale}
    if dim:
        kwargs['dim'] = dim

    # Special handling for Denoise models (scale=1 fixed usually, but we pass it anyway or handle inside)
    if 'denoise' in name:
        kwargs['mode'] = config.get('mode', 'medium')
        # Remove scaling_factor for Denoise classes if they don't accept it or ignore it
        # QuickDenoiseNet takes no scaling_factor arg (it calls super with fixed 1)
        if 'scaling_factor' in kwargs:
            del kwargs['scaling_factor']

    if name == 'quicksrnet_small':
        return QuickSRNetSmall(**kwargs)
    elif name == 'quicksrnet_medium':
        return QuickSRNetMedium(**kwargs)
    elif name == 'quicksrnet_large':
        return QuickSRNetLarge(**kwargs)
    elif name == 'quicksrnet_denoise':
        return QuickDenoiseNet(**kwargs)
    elif name == 'quicksrnet_denoise_opt':
        return QuickDenoiseOpt(**kwargs)
    
    elif name == 'lrcsr':
        # LRCSR takes scale_factor, dim.
        # config might use 'scale' but class uses 'scale_factor'.
        return LRCSR(scale_factor=scale, dim=dim if dim else 32)
        
    elif name == 'svsrnet':
        # SVSRNet args: scaling_factor, n_resblocks=12, n_feats=64
        n_resblocks = config.get('n_resblocks', 12)
        n_feats = dim if dim else 64
        return SVSRNet(scaling_factor=scale, n_resblocks=n_resblocks, n_feats=n_feats)
        
    elif name == 'ddrnet':
        # DDRNet args: scale_factor, dim=48
        return DDRNet(scale_factor=scale, dim=dim if dim else 48)
        
    elif name == 'rrdbnet':
        # RRDBNet args
        num_in_ch = config.get('num_in_ch', 3)
        num_out_ch = config.get('num_out_ch', 3)
        num_feat = dim if dim else config.get('num_feat', 64)
        num_block = config.get('num_block', 23)
        num_grow_ch = config.get('num_grow_ch', 32)
        
        return RRDBNet(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch
        )
    elif name == 'qcsawaresrnet_small':
        return QCSAwareSRNetSmall(**kwargs)
    elif name == 'qcsawaresrnet_medium':
        return QCSAwareSRNetMedium(**kwargs)
    elif name == 'qcsawaresrnet_large':
        return QCSAwareSRNetLarge(**kwargs)

    elif name == 'svfocussrnet':
        n_resblocks = config.get('n_resblocks', 8)
        n_feats = dim if dim else 32
        use_advanced_rep = config.get('use_advanced_rep', False)
        return SVFocusSRNet(scaling_factor=scale, n_resblocks=n_resblocks, n_feats=n_feats, use_advanced_rep=use_advanced_rep)

    elif name == 'mambair':
        return MambaIR(
            upscale=scale,
            in_chans=config.get('in_chans', 3),
            img_size=config.get('img_size', 64),
            embed_dim=dim if dim else config.get('embed_dim', 180),
            d_state=config.get('d_state', 16),
            depths=config.get('depths', (6, 6, 6, 6, 6, 6)),
            upsampler=config.get('upsampler', 'pixelshuffle')
        )
    
    elif name == 'mambairv2':
        return MambaIRv2(
            upscale=scale,
            in_chans=config.get('in_chans', 3),
            img_size=config.get('img_size', 64),
            embed_dim=dim if dim else config.get('embed_dim', 48),
            d_state=config.get('d_state', 8),
            depths=config.get('depths', (6, 6, 6, 6)),
            upsampler=config.get('upsampler', 'pixelshuffle')
        )

    # ====== GSR Models ======
    elif name == 'corefusion':
        return CoReFusion(
            scale_factor=scale,
            dim=dim,
            pretrained=config.get('pretrained', True),
            contrastive=config.get('contrastive', True),
        )
    elif name == 'lapgsr':
        return LapGSR(
            scale_factor=scale,
            num_high=config.get('num_high', 2),
            nrb_low=config.get('nrb_low', 3),
            nrb_high=config.get('nrb_high', 5),
            nrb_top=config.get('nrb_top', 4),
        )
    elif name == 'lapgsr_disc':
        return LapGSRDiscriminator(in_channels=config.get('in_channels', 3))

    else:
        raise ValueError(f"Unknown model name: {name}")
