"""
LapGSR: Laplacian Pyramid based Guided Super-Resolution
Based on Scientific Reports 2024 paper.

Architecture:
  - Laplacian Pyramid decomposition of HR RGB guide (non-learnable)
  - Replace low-frequency component with upsampled LR thermal
  - Trans_low, Trans_high, Trans_top learnable transform modules
  - PatchGAN discriminator for adversarial training

Training:
  - GAN loss (LSGAN) + MSE (weight=2000) + Gradient Penalty (weight=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Building Blocks
# ==========================================

class ResidualBlock(nn.Module):
    """Conv-BN-LeakyReLU-Conv-BN + skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Trans_low(nn.Module):
    """Transform module for low-frequency (base) component."""
    def __init__(self, in_ch=3, nrb=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(nrb):
            layers.append(ResidualBlock(64))
        layers.append(nn.Conv2d(64, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Trans_high(nn.Module):
    """Transform module for high-frequency (detail) components."""
    def __init__(self, in_ch=9, nrb=5):
        super().__init__()
        # in_ch = 3 (high_freq) + 3 (low_output upsampled) + 3 (original low) = 9
        layers = [nn.Conv2d(in_ch, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(nrb):
            layers.append(ResidualBlock(64))
        layers.append(nn.Conv2d(64, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Trans_top(nn.Module):
    """Final transform module at the top (full resolution) level."""
    def __init__(self, in_ch=6, nrb=4):
        super().__init__()
        # in_ch = 3 (accumulated) + 3 (original HR detail)
        layers = [nn.Conv2d(in_ch, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(nrb):
            layers.append(ResidualBlock(64))
        layers.append(nn.Conv2d(64, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================
# Laplacian Pyramid (Fixed, Non-learnable)
# ==========================================

class LaplacianPyramid(nn.Module):
    """
    Decomposes an image into Laplacian pyramid levels.
    Returns [low_freq, high_1, high_2, ...] where high are detail bands.
    """
    def __init__(self, num_high=2):
        super().__init__()
        self.num_high = num_high
        # Gaussian kernel for downsampling
        kernel = self._gauss_kernel(5, channels=3)
        self.register_buffer('kernel', kernel)

    @staticmethod
    def _gauss_kernel(size=5, channels=3):
        """Create a Gaussian smoothing kernel."""
        import numpy as np
        kernel = torch.tensor([
            [1., 4., 6., 4., 1.],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.]
        ]) / 256.0
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        return kernel

    def _downsample(self, x):
        return F.conv2d(x, self.kernel, stride=2, padding=2, groups=x.shape[1])

    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            pyramid: list of [low, high_1, high_2, ...high_num_high]
                     ordered from lowest to highest resolution
        """
        current = x
        pyramid = []  # will hold high-freq bands
        
        for _ in range(self.num_high):
            down = self._downsample(current)
            up = self._upsample(down, size=current.shape[2:])
            high = current - up  # detail (high-freq) at this level
            pyramid.append(high)
            current = down
        
        # current is the low-frequency base
        # Return: [low_freq, high_from_lowest_res, ..., high_from_highest_res]
        return [current] + pyramid[::-1]  # reverse so index 0 = low, last = finest detail


# ==========================================
# LapGSR Generator
# ==========================================

class LapGSR(nn.Module):
    """
    Laplacian Pyramid Guided Super-Resolution Generator.
    
    Args:
        scale_factor (int): SR scale factor (e.g. 2, 4)
        num_high (int): number of high-frequency pyramid levels (default 2)
        nrb_low (int): residual blocks in Trans_low (default 3)
        nrb_high (int): residual blocks in Trans_high (default 5)
        nrb_top (int): residual blocks in Trans_top (default 4)
    """
    def __init__(self, scale_factor=2, num_high=2, nrb_low=3, nrb_high=5, nrb_top=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_high = num_high
        
        # Fixed Laplacian pyramid decomposition
        self.lap_pyramid = LaplacianPyramid(num_high=num_high)
        
        # Learnable transform modules
        self.trans_low = Trans_low(in_ch=3, nrb=nrb_low)
        
        # One Trans_high per intermediate level (num_high - 1 levels)
        self.trans_highs = nn.ModuleList()
        for _ in range(max(num_high - 1, 0)):
            self.trans_highs.append(Trans_high(in_ch=9, nrb=nrb_high))
        
        # Top-level (finest) transform
        self.trans_top = Trans_top(in_ch=6, nrb=nrb_top)

    def forward(self, lr, guide=None):
        """
        Args:
            lr: (B, 3, H_lr, W_lr) low-resolution thermal
            guide: (B, 3, H_hr, W_hr) high-resolution RGB guide.
                   If None, LR is bilinear-upsampled and used as self-guide.
        Returns:
            sr: (B, 3, H_hr, W_hr) super-resolved thermal
        """
        # Fallback: guide가 없으면 LR upsample을 guide로 사용
        if guide is None:
            H_hr = lr.shape[2] * self.scale_factor
            W_hr = lr.shape[3] * self.scale_factor
            guide = F.interpolate(lr, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        
        # Decompose guide into Laplacian pyramid
        # pyr = [low_base, high_1 (lowest res detail), ..., high_N (finest detail)]
        pyr = self.lap_pyramid(guide)
        
        # Replace low-frequency base with upsampled LR thermal
        low_base_size = pyr[0].shape[2:]
        lr_up = F.interpolate(lr, size=low_base_size, mode='bilinear', align_corners=False)
        
        # Transform low-frequency
        low_out = self.trans_low(lr_up)
        
        # Process intermediate high-frequency levels
        current = low_out
        for i, trans_high in enumerate(self.trans_highs):
            # Upsample current to next level size
            high_band = pyr[i + 1]  # corresponding high-freq band
            current_up = F.interpolate(current, size=high_band.shape[2:], mode='bilinear', align_corners=False)
            low_orig_up = F.interpolate(lr_up, size=high_band.shape[2:], mode='bilinear', align_corners=False)
            
            # Concat: high_freq + upsampled_current + upsampled_original_low = 9ch
            concat = torch.cat([high_band, current_up, low_orig_up], dim=1)
            current = trans_high(concat)
        
        # Top level (finest resolution)
        top_high = pyr[-1]  # finest detail
        current_up = F.interpolate(current, size=top_high.shape[2:], mode='bilinear', align_corners=False)
        concat_top = torch.cat([current_up, top_high], dim=1)  # 6ch
        sr = self.trans_top(concat_top)
        
        sr = torch.clamp(sr, 0.0, 1.0)
        return sr


# ==========================================
# PatchGAN Discriminator
# ==========================================

class LapGSRDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for LapGSR adversarial training.
    4-layer architecture with stride-2 convolutions.
    
    Args:
        in_channels (int): input image channels (default 3)
        ndf (int): base number of discriminator filters (default 64)
    """
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        
        self.net = nn.Sequential(
            # Layer 1: no BN
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 1-channel patch map
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)
