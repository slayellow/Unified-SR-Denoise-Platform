from .svfocussrnet import SVFocusSRNet
from .bicubicplusplus import BicubicPlusPlus

__all__ = ["SVFocusSRNet", "BicubicPlusPlus", "build_model"]


def build_model(config):
    """
    Factory function to create a model based on configuration.
    
    Args:
        config (dict): Must contain 'name' and model-specific args.
    
    Returns:
        nn.Module
    """
    name = config.get('name', '').lower()
    
    # Common Args
    scale = config.get('scale', 2)
    dim = config.get('dim', None) # If None, use default of class
    
    if name == 'svfocussrnet':
        n_resblocks = config.get('n_resblocks', 8)
        n_feats = dim if dim else 32
        use_advanced_rep = config.get('use_advanced_rep', False)
        return SVFocusSRNet(scaling_factor=scale, n_resblocks=n_resblocks, n_feats=n_feats, use_advanced_rep=use_advanced_rep)

    elif name == 'bicubicplusplus':
        # Every other key goes to the constructor, so a typo raises TypeError instead of
        # training the default model. The defaults live in BicubicPlusPlus.__init__.
        opts = {k: v for k, v in config.items() if k not in ('name', 'scale', 'dim')}
        if dim:
            opts['n_feats'] = dim
        return BicubicPlusPlus(scaling_factor=scale, **opts)

    else:
        raise ValueError(f"Unknown model name: {name}")
