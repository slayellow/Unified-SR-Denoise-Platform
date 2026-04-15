import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================================
# 1. Basic Modules (NAFBlock, RDB, Frequency Decomposition)
# ==================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """ Nonlinear Activation Free Block (Used in HFE & SFE) """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)
        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout(x)
        y = inp + x * self.beta
        
        x = self.conv4(y)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout(x)
        return y + x * self.gamma

class RDB(nn.Module):
    """ Residual Dense Block (Used in HFE) """
    def __init__(self, nChannels, nDenselayer=3, growthRate=32):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList() 
        nChannels_ = nChannels
        
        for i in range(nDenselayer):
            self.layers.append(self.make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
            
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def make_dense(self, nChannels, growthRate):
        return nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(inputs, dim=1))
            inputs.append(out)
            
        out = self.conv_1x1(torch.cat(inputs, dim=1))
        return out + x 

class FrequencyDecomposition(nn.Module):
    """ Gaussian Blur로 LF/HF 분리 """
    def __init__(self, kernel_size=5, sigma=1.0, channels=3):
        super().__init__()
        self.channels = channels
        
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        gaussian_kernel = (1./(2.*3.14159*variance)) * \
                          torch.exp( -torch.sum((xy_grid - mean)**2., dim=-1) / \
                          (2*variance) )

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        
        self.register_buffer('filter', gaussian_kernel)
        self.pad = kernel_size // 2

    def forward(self, x):
        lf = F.conv2d(x, self.filter, padding=self.pad, groups=self.channels)
        hf = x - lf
        return lf, hf

# ==================================================================
# 2. Core Modules (SDCPA, HFE, SFE, Fusion)
# ==================================================================

class SDCPA(nn.Module):
    """ Self-Dual Calibrated Projection Attention """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) 
        
        self.pa_conv = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        self.out_conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        local_feat = self.conv3x3(self.conv1(x))
        
        pa_in = self.pa_conv(x)
        pa1 = self.sigmoid(pa_in) * self.conv3x3(x)
        
        b, c, h, w = x.shape
        pa2_att = self.softmax(pa_in.view(b, c, -1)).view(b, c, h, w)
        pa2 = pa2_att + self.conv3x3(pa_in)
        
        dual_feat = torch.cat([local_feat, pa1 + pa2], dim=1)
        out = self.out_conv(dual_feat)
        
        return x + out 

class HFE(nn.Module):
    """ Hierarchical Feature Enhancer (Low-Freq Stream) """
    def __init__(self, dim):
        super().__init__()
        self.naf_branch = nn.Sequential(*[NAFBlock(dim) for _ in range(4)])
        self.rdb_branch = nn.Sequential(*[RDB(dim) for _ in range(3)])
        
        self.fusion = nn.Conv2d(dim * 2, dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x):
        noise_naf = self.naf_branch(x)
        noise_rdb = self.rdb_branch(x)
        
        noise_map = self.fusion(torch.cat([noise_naf, noise_rdb], dim=1))
        noise_map = noise_map * self.alpha
        
        clean_feat = x - noise_map
        return clean_feat, noise_map

class SFE(nn.Module):
    """ Simple Feature Enhancer (High-Freq Stream) """
    def __init__(self, dim):
        super().__init__()
        self.naf_blocks = nn.Sequential(*[NAFBlock(dim) for _ in range(4)])
        self.beta = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x):
        feat = self.naf_blocks(x)
        return x + feat * self.beta

class DualStreamFusion(nn.Module):
    """ Dual-stream Attention Fusion """
    def __init__(self, dim):
        super().__init__()
        self.merge = nn.Conv2d(dim * 2, dim, 1)
        
        self.ca_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
        self.final_conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, lf_feat, hf_feat):
        x = torch.cat([lf_feat, hf_feat], dim=1)
        x = self.merge(x)
        
        ca = self.ca_mlp(self.ca_pool(x))
        x = x * ca
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.sa_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * sa
        
        out = self.final_conv(x)
        return out

# ==================================================================
# 3. DDR-Net Main Architecture
# ==================================================================

class DDRNet(nn.Module):
    def __init__(self, scale_factor=2, dim=48):
        super().__init__()
        self.freq_decomp = FrequencyDecomposition()
        
        self.lf_head = nn.Conv2d(3, dim, 3, padding=1)
        self.hf_head = nn.Conv2d(3, dim, 3, padding=1)
        
        self.sdcpa_lf = SDCPA(dim)
        self.sdcpa_hf = SDCPA(dim)
        
        self.hfe = HFE(dim)
        self.sfe = SFE(dim)
        
        self.fusion = DualStreamFusion(dim)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(dim, 3, 3, padding=1)
        )
        
    def forward(self, x):
        lf_img, hf_img = self.freq_decomp(x)
        
        lf_feat = self.lf_head(lf_img)
        hf_feat = self.hf_head(hf_img)
        
        lf_feat = self.sdcpa_lf(lf_feat)
        hf_feat = self.sdcpa_hf(hf_feat)
        
        clean_lf, noise_map = self.hfe(lf_feat)
        enhanced_hf = self.sfe(hf_feat)
        
        fused = self.fusion(clean_lf, enhanced_hf)
        
        out = self.upsample(fused)
        
        # NOTE: Original implementation returns (out, noise_map, enhanced_hf) in train mode
        # We standarize to just 'out' for unified pipeline, unless losses need them.
        # DDRNet loss usually requires aux outputs. We will handle this in training loop or return dict.
        # For now, let's keep it simple for inference, but if training, we might need to adjust.
        # But 'forward' should preferably return the main output.
        # We can attach aux outputs to the model object or return a dict/tuple if needed.
        # Let's check how the training loop uses it. The original code (line 257) returns tuple in training.
        # To be safe and compatible with standard "model(x)" calls in Eval/Inference, we return 'out' only by default,
        # or we check self.training.
        
        if self.training:
            return out, noise_map, enhanced_hf
        else:
            return out
