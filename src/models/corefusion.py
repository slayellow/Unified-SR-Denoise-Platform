"""
CoReFusion: Contrastive Regularized Fusion for Guided Thermal Super-Resolution
Based on CVPR 2023 Workshop paper.

Architecture:
  - Dual ResNet34 encoders (thermal LR upsampled + RGB guide)
  - Element-wise Maximum fusion at each encoder level
  - UNet decoder with skip connections
  - Contrastive projection heads (training only)

Modified from original:
  - Output range [0, 1] (clamp) instead of [-1, 1] (tanh)
  - 3ch output instead of 1ch for platform compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet34Encoder(nn.Module):
    """Feature extractor using ResNet34 backbone, returns 5 levels of features."""
    
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)
        
        # Adjust first conv if in_channels != 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Level 0: conv1 + bn1 + relu → (B, 64, H/2, W/2)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # Level 1: maxpool + layer1 → (B, 64, H/4, W/4)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        # Level 2: layer2 → (B, 128, H/8, W/8)
        self.layer2 = resnet.layer2
        # Level 3: layer3 → (B, 256, H/16, W/16)
        self.layer3 = resnet.layer3
        # Level 4: layer4 → (B, 512, H/32, W/32)
        self.layer4 = resnet.layer4

    def forward(self, x):
        f0 = self.layer0(x)   # (B, 64,  H/2,  W/2)
        f1 = self.layer1(f0)  # (B, 64,  H/4,  W/4)
        f2 = self.layer2(f1)  # (B, 128, H/8,  W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 512, H/32, W/32)
        return [f0, f1, f2, f3, f4]


class DecoderBlock(nn.Module):
    """UNet decoder block: upsample + concat skip + 2x ConvBN-ReLU."""
    
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CoReFusion(nn.Module):
    """
    CoReFusion Guided Super-Resolution Network.
    
    Args:
        scale_factor (int): upscaling factor (e.g. 2, 4)
        dim (int): not used (kept for build_model compatibility)
        in_channels (int): input channels for thermal (default 3)
        out_channels (int): output channels (default 3)
        pretrained (bool): use pretrained ResNet34 encoders
        contrastive (bool): enable contrastive projection heads
    """
    
    def __init__(self, scale_factor=2, dim=None, in_channels=3, out_channels=3,
                 pretrained=True, contrastive=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.contrastive = contrastive
        
        # Dual encoders
        self.encoder_thermal = ResNet34Encoder(in_channels=in_channels, pretrained=pretrained)
        self.encoder_rgb = ResNet34Encoder(in_channels=3, pretrained=pretrained)
        
        # Decoder (channels: 512 → 256 → 128 → 64 → 32)
        # Decoder input is fused bottleneck (512), skip channels from fusion
        self.decoder4 = DecoderBlock(512, 256, 256)   # from f4, skip=fused_f3
        self.decoder3 = DecoderBlock(256, 128, 128)    # from d4, skip=fused_f2
        self.decoder2 = DecoderBlock(128, 64, 64)      # from d3, skip=fused_f1
        self.decoder1 = DecoderBlock(64, 64, 32)       # from d2, skip=fused_f0
        
        # Output head
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Contrastive projection heads (training only)
        if contrastive:
            self.proj_thermal = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 128),
            )
            self.proj_rgb = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 128),
            )

    def forward(self, lr, guide=None):
        """
        Args:
            lr: (B, C, H_lr, W_lr) low-resolution thermal
            guide: (B, 3, H_hr, W_hr) high-resolution RGB guide.
                   If None, LR is bilinear-upsampled and used as self-guide.
        Returns:
            sr: (B, out_channels, H_hr, W_hr)
            (optional) proj_t, proj_r: contrastive projections during training
        """
        # Fallback: guide가 없으면 RGB feature를 0으로 → max fusion에서 thermal만 통과
        H_hr = guide.shape[2] if guide is not None else lr.shape[2] * self.scale_factor
        W_hr = guide.shape[3] if guide is not None else lr.shape[3] * self.scale_factor
        
        # Upsample LR thermal to HR size
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        
        # Encode
        t_feats = self.encoder_thermal(lr_up)  # [f0..f4]
        if guide is not None:
            r_feats = self.encoder_rgb(guide)   # [f0..f4]
        else:
            r_feats = [torch.zeros_like(t) for t in t_feats]
        
        # Element-wise maximum fusion at each level
        fused = [torch.max(t, r) for t, r in zip(t_feats, r_feats)]
        
        # Decode
        d4 = self.decoder4(fused[4], fused[3])  # 512 + skip 256 → 256
        d3 = self.decoder3(d4, fused[2])          # 256 + skip 128 → 128
        d2 = self.decoder2(d3, fused[1])          # 128 + skip 64 → 64
        d1 = self.decoder1(d2, fused[0])          # 64 + skip 64 → 32
        
        # Upsample decoder output to HR size (decoder output is H/2 × W/2)
        d1 = F.interpolate(d1, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        
        sr = self.head(d1)
        sr = torch.clamp(sr, 0.0, 1.0)
        
        # Contrastive projections (training)
        if self.contrastive and self.training:
            proj_t = self.proj_thermal(t_feats[4])
            proj_r = self.proj_rgb(r_feats[4])
            return sr, proj_t, proj_r
        
        return sr
