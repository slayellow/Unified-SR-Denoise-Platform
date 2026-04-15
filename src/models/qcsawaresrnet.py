import torch
import torch.nn as nn
from collections import OrderedDict
import math

# ==========================================
# 1. Helper Blocks
# ==========================================

class AddOp(nn.Module):
    """Simple Element-wise Addition"""
    def forward(self, x1, x2):
        return x1 + x2

class AnchorOp(nn.Module):
    """
    [Skip Connection Layer - HTP Optimized]
    입력을 Scaling Factor만큼 복제하되, HTP 크루톤 메모리 정렬(32배수)을 위해 
    나머지 더미 채널은 0으로 패딩(Padding)하여 출력합니다.
    """
    def __init__(self, scaling_factor, in_channels=3, kernel_size=1):
        super().__init__()
        
        # 1. 수학적 최소 필요 채널 (예: 2x -> 12, 4x -> 48)
        self.min_required_channels = in_channels * (scaling_factor ** 2)
        
        # 2. HTP 정렬 채널: 32의 배수로 올림 (예: 12 -> 32, 48 -> 64)
        self.htp_aligned_channels = math.ceil(self.min_required_channels / 32.0) * 32
        
        # 3. 32배수 정렬된 Conv2d 생성
        self.net = nn.Conv2d(in_channels=in_channels,
                             out_channels=self.htp_aligned_channels,
                             kernel_size=kernel_size,
                             padding=kernel_size//2,
                             bias=True)

        self.init_weights(in_channels, scaling_factor, kernel_size)
        
        for param in self.net.parameters():
            param.requires_grad = False

    def init_weights(self, in_channels, scaling_factor, kernel_size):
        num_channels_per_group = in_channels // self.net.groups
        
        # 가중치와 바이어스를 htp_aligned_channels 크기에 맞춰 0으로 생성
        weight = torch.zeros(self.htp_aligned_channels, num_channels_per_group, kernel_size, kernel_size)
        bias = torch.zeros(self.htp_aligned_channels)
        
        # 유효한 채널(min_required_channels)까지만 1:1 매핑 복사, 나머지는 그대로 0(Zero Padding) 유지
        for ii in range(in_channels):
            weight[ii * scaling_factor**2: (ii + 1) * scaling_factor**2, 
                   ii % num_channels_per_group,
                   kernel_size // 2, kernel_size // 2] = 1.

        self.net.load_state_dict(OrderedDict({'weight': weight, 'bias': bias}))

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. QuickSRNet Base Architecture
# ==========================================

class QCSAwareSRNetBase(nn.Module):
    def __init__(self,
                 scaling_factor,
                 num_channels,
                 num_intermediate_layers,
                 use_ito_connection,
                 in_channels=3,
                 out_channels=3):
        super().__init__()
        
        self.scaling_factor = int(scaling_factor)
        self._use_ito_connection = use_ito_connection

        # --- HTP Aligned Channel Calculation ---
        self.min_required_channels = out_channels * (self.scaling_factor ** 2)
        self.htp_aligned_channels = math.ceil(self.min_required_channels / 32.0) * 32

        # 1. Input Conv (Conv3x3 + ReLU1)
        layers = [
            nn.Conv2d(in_channels, num_channels, 3, padding=1),
            nn.Hardtanh(0., 1.) 
        ]

        # 2. Hidden Layers (Conv3x3 + ReLU1)
        for _ in range(num_intermediate_layers):
            layers.extend([
                nn.Conv2d(num_channels, num_channels, 3, padding=1),
                nn.Hardtanh(0., 1.)
            ])
        
        self.cnn = nn.Sequential(*layers)

        # 3. Last Conv (Channel Expansion for PixelShuffle)
        # [핵심 수정] 12/48채널이 아닌, 계산된 32/64채널을 출력하도록 강제합니다.
        self.conv_last = nn.Conv2d(num_channels, self.htp_aligned_channels, 3, padding=1)

        # 4. Skip Connection (Input-to-Output)
        if use_ito_connection:
            self.add_op = AddOp()
            self.anchor = AnchorOp(self.scaling_factor, in_channels=in_channels)

        # 5. Upsampling Block
        self.pixel_shuffle = nn.PixelShuffle(self.scaling_factor)
        
        # 6. Final Activation (ReLU1)
        self.clip_output = nn.Hardtanh(0., 1.)

        # 초기화 (Identity Initialization)
        self.initialize()

    def forward(self, x):
        # Main Branch
        feat = self.cnn(x)
        residual = self.conv_last(feat) # [B, 32/64, H, W]

        # Skip Connection Branch
        if self._use_ito_connection:
            input_upsampled = self.anchor(x) # [B, 32/64, H, W]
            
            # HTP Fast-path: 완벽하게 32배수로 정렬된 텐서끼리의 Add 연산
            out = self.add_op(input_upsampled, residual) 
        else:
            out = residual

        # 32/64채널 텐서를 PixelShuffle 수행
        out = self.pixel_shuffle(out)
        
        # [핵심 수정] HTP 정렬을 위해 추가된 더미 채널들을 제거하고 필요한 3채널(RGB)만 슬라이싱
        out = out[:, :3, :, :]
        
        out = self.clip_output(out)
        return out
    
    def switch_to_deploy(self):
        pass
    
    def initialize(self):
        """Identity Initialization"""
        # CNN Layers Initialization
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                mid = m.kernel_size[0] // 2
                with torch.no_grad():
                    for i in range(min(m.in_channels, m.out_channels)):
                        m.weight[i, i, mid, mid] += 1.0

        # Last Conv Initialization
        m = self.conv_last
        mid = m.kernel_size[0] // 2
        
        # [수정됨] 전체 32/64 채널 중, 유효한 12/48 채널 영역만 초기화를 진행합니다.
        valid_out_ch = self.min_required_channels 
        scale_sq = self.scaling_factor ** 2
        
        with torch.no_grad():
            for i_out in range(valid_out_ch):
                i_in = (i_out % valid_out_ch) // scale_sq
                if i_in < m.in_channels:
                    m.weight[i_out, i_in, mid, mid] += 0.1

# ==========================================
# 3. Model Variants & Alias
# ==========================================

class QCSAwareSRNetSmall(QCSAwareSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        # dim 인자가 들어오면 num_channels로 매핑
        dim = kwargs.get('dim', 32) 
        super().__init__(scaling_factor, num_channels=dim, num_intermediate_layers=2, use_ito_connection=False)

class QCSAwareSRNetMedium(QCSAwareSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        dim = kwargs.get('dim', 32)
        super().__init__(scaling_factor, num_channels=dim, num_intermediate_layers=5, use_ito_connection=False)

class QCSAwareSRNetLarge(QCSAwareSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        dim = kwargs.get('dim', 64)
        # Large 모델은 Skip Connection 사용 (학습 안정성)
        super().__init__(scaling_factor, num_channels=dim, num_intermediate_layers=11, use_ito_connection=True)
