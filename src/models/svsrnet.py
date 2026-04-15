import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Reparameterization Block (Key Upgrade)
# ==========================================
class RepBlock(nn.Module):
    """
    RepVGG Block 스타일:
    학습 시: 3x3 Conv + 1x1 Conv + Identity (3갈래) -> 성능 극대화
    추론 시: 3x3 Conv 1개로 합체 (Deploy) -> 속도 극대화 (EDSR과 동일 속도)
    """
    def __init__(self, n_feats, kernel_size=3, dilation=1, padding=1):
        super(RepBlock, self).__init__()
        self.deploy = False
        self.n_feats = n_feats
        
        # 학습용 3갈래 브랜치
        self.branch_3x3 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=True)
        self.branch_1x1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=True)
        self.branch_identity = nn.Identity()
        
        self.act = nn.Hardtanh(0., 1.)

    def forward(self, x):
        if self.deploy:
            # 추론 모드: 단순한 Conv 하나만 통과 (매우 빠름)
            return self.act(self.branch_3x3(x))
        else:
            # 학습 모드: 3갈래의 정보를 모두 합침 (강력한 성능)
            x3 = self.branch_3x3(x)
            x1 = self.branch_1x1(x)
            xi = self.branch_identity(x)
            return self.act(x3 + x1 + xi)

    def switch_to_deploy(self):
        """
        학습이 끝난 후 호출. 3개의 브랜치를 하나의 3x3 Conv로 수학적으로 병합함.
        """
        if self.deploy:
            return
        
        # 1. 3x3 커널과 바이어스 가져오기
        kernel_3x3 = self.branch_3x3.weight.data
        bias_3x3 = self.branch_3x3.bias.data

        # 2. 1x1 커널을 3x3 중심으로 패딩하여 변환
        kernel_1x1 = self.branch_1x1.weight.data
        # (C, C, 1, 1) -> (C, C, 3, 3) : 중앙에만 값 있고 나머지는 0
        kernel_1x1_padded = F.pad(kernel_1x1, (1, 1, 1, 1)) 
        bias_1x1 = self.branch_1x1.bias.data

        # 3. Identity를 3x3 커널로 표현 (Dirac Delta)
        # 자기 자신을 그대로 내보내는 Conv는 중앙값이 1이고 나머지가 0인 필터와 같음
        kernel_identity = torch.zeros_like(kernel_3x3)
        for i in range(self.n_feats):
            kernel_identity[i, i, 1, 1] = 1.0
        
        # 4. 모든 가중치와 바이어스 합산
        fused_kernel = kernel_3x3 + kernel_1x1_padded + kernel_identity
        fused_bias = bias_3x3 + bias_1x1 
        # (Identity는 bias가 0이므로 더할 필요 없음)

        # 5. 기존 branch 삭제 및 단일 Conv로 교체
        self.branch_3x3 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True)
        self.branch_3x3.weight.data = fused_kernel
        self.branch_3x3.bias.data = fused_bias
        
        del self.branch_1x1
        del self.branch_identity
        
        self.deploy = True
        
# ==========================================
# 2. SVSRNet (RepSR Version)
# ==========================================
class SVSRNet(nn.Module):
    def __init__(self, scaling_factor, n_resblocks=12, n_feats=64, in_channels=3, out_channels=3):
        super(SVSRNet, self).__init__()
        self.scaling_factor = int(scaling_factor)
        
        # Head
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1, bias=True)

        # Body (Using RepBlock instead of ResBlock)
        self.body = nn.Sequential(*[
            RepBlock(n_feats) for _ in range(n_resblocks)
        ])
        
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)

        # Tail (Upsample)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, out_channels * (scaling_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scaling_factor)
        )
        
        self.final_act = nn.Hardtanh(0., 1.)

    def forward(self, x):
        x_head = self.head(x)
        res = self.body(x_head)
        res = self.body_tail(res)
        res += x_head # Global Skip Connection
        out = self.upsample(res)
        out = self.final_act(out)
        return out

    def switch_to_deploy(self):
        for m in self.modules():
            if m is not self and isinstance(m, RepBlock):
                m.switch_to_deploy()
