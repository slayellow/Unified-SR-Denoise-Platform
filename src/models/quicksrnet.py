import torch
import torch.nn as nn
from collections import OrderedDict

# ==========================================
# 1. Helper Blocks
# ==========================================

class AddOp(nn.Module):
    """Simple Element-wise Addition"""
    def forward(self, x1, x2):
        return x1 + x2

class AnchorOp(nn.Module):
    """
    [Skip Connection Layer]
    Input을 Scaling Factor만큼 복제하여 Nearest Neighbor Upsampling 효과를 냄.
    Conv2d로 구현되어 있어 NPU에서 매우 효율적임.
    """
    def __init__(self, scaling_factor, in_channels=3, kernel_size=1):
        super().__init__()
        
        # PixelShuffle은 (C * r^2) 채널을 입력으로 받음
        out_channels = in_channels * (scaling_factor ** 2)
        
        self.net = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             padding=kernel_size//2,
                             bias=True) # Bias는 0으로 초기화됨

        # [핵심] 가중치를 조작하여 입력 픽셀을 복제(Repeat)하도록 설정
        # 학습되지 않도록 freeze_weights=True 적용
        self.init_weights(in_channels, scaling_factor, kernel_size)
        
        for param in self.net.parameters():
            param.requires_grad = False

    def init_weights(self, in_channels, scaling_factor, kernel_size):
        num_channels_per_group = in_channels // self.net.groups
        weight = torch.zeros(in_channels * scaling_factor**2, num_channels_per_group, kernel_size, kernel_size)
        bias = torch.zeros(weight.shape[0])
        
        for ii in range(in_channels):
            # 각 입력 채널을 scaling_factor^2 만큼의 출력 채널로 1:1 매핑 (복사)
            weight[ii * scaling_factor**2: (ii + 1) * scaling_factor**2, 
                   ii % num_channels_per_group,
                   kernel_size // 2, kernel_size // 2] = 1.

        self.net.load_state_dict(OrderedDict({'weight': weight, 'bias': bias}))

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. QuickSRNet Base Architecture
# ==========================================

class QuickSRNetBase(nn.Module):
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

        # 1. Input Conv (Conv3x3 + ReLU1)
        layers = [
            nn.Conv2d(in_channels, num_channels, 3, padding=1),
            nn.Hardtanh(0., 1.) # 0~1 Clipping (Quantization Friendly)
        ]

        # 2. Hidden Layers (Conv3x3 + ReLU1)
        for _ in range(num_intermediate_layers):
            layers.extend([
                nn.Conv2d(num_channels, num_channels, 3, padding=1),
                nn.Hardtanh(0., 1.)
            ])
        
        self.cnn = nn.Sequential(*layers)

        # 3. Last Conv (Channel Expansion for PixelShuffle)
        # Output Channels = 3 * (scale^2)
        self.conv_last = nn.Conv2d(num_channels, out_channels * (self.scaling_factor ** 2), 3, padding=1)

        # 4. Skip Connection (Input-to-Output) - Optional
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
        residual = self.conv_last(feat)

        # Skip Connection Branch
        if self._use_ito_connection:
            input_upsampled = self.anchor(x)
            out = self.add_op(input_upsampled, residual)
        else:
            out = residual

        # [수정됨] PixelShuffle 먼저 수행 후 마지막에 Clamp
        # 논리적으로 최종 출력 이미지의 픽셀 범위를 제한하는 것이 더 안전함
        out = self.pixel_shuffle(out)
        out = self.clip_output(out)

        return out
    
    def switch_to_deploy(self):
        """
        QuickSRNet은 RepConv를 쓰지 않는 Plain CNN 구조이므로,
        별도의 구조 변경(Reparameterization) 과정이 필요 없음.
        """
        pass
    
    def initialize(self):
        """Identity Initialization: 초기 학습 시 입력이 그대로 출력되도록 유도"""
        # CNN Layers Initialization
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                # 기본적으로 0에 가까운 값이나 작은 랜덤값
                # 여기서는 간단히 Xavier 등을 쓰지 않고, 중심 픽셀만 강화하는 전략 사용 가능
                # 하지만 제공해주신 코드의 전략(기존 가중치 + 1)을 따름
                mid = m.kernel_size[0] // 2
                with torch.no_grad():
                    # Identity Mapping 강화
                    for i in range(min(m.in_channels, m.out_channels)):
                        m.weight[i, i, mid, mid] += 1.0

        # Last Conv Initialization
        # PixelShuffle 직전 레이어이므로, 채널 매핑을 고려하여 초기화
        m = self.conv_last
        mid = m.kernel_size[0] // 2
        out_ch = m.out_channels
        scale_sq = out_ch // 3 # assuming out_channels=3
        
        with torch.no_grad():
            for i_out in range(out_ch):
                # 입력 채널과 매핑되는 출력 채널 계산
                i_in = (i_out % out_ch) // scale_sq
                if i_in < m.in_channels:
                    m.weight[i_out, i_in, mid, mid] += 0.1 # 마지막 층은 잔차(Residual) 성격이 강하므로 작게 시작

# ==========================================
# 3. Model Variants & Alias
# ==========================================

class QuickSRNetSmall(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        # dim 인자가 들어오면 num_channels로 매핑
        dim = kwargs.get('dim', 32) 
        super().__init__(scaling_factor, num_channels=dim, num_intermediate_layers=2, use_ito_connection=False)

class QuickSRNetMedium(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        dim = kwargs.get('dim', 32)
        super().__init__(scaling_factor, num_channels=dim, num_intermediate_layers=5, use_ito_connection=False)

class QuickSRNetLarge(QuickSRNetBase):
    def __init__(self, scaling_factor, **kwargs):
        dim = kwargs.get('dim', 64)
        # Large 모델은 Skip Connection 사용 (학습 안정성)
        super().__init__(scaling_factor, num_channels=dim, num_intermediate_layers=11, use_ito_connection=True)

# Alias
LRCSR = QuickSRNetMedium

# ==========================================
# 4. Denoise Models
# ==========================================

class QuickDenoiseNet(QuickSRNetBase):
    """
    [User Guide]
    Q8550 1080p Real-time Denoising을 위한 추천 모델입니다.
    
    - mode='small': 매우 빠름, 흰색 점 제거 가능, 텍스처 디테일 약간 부족할 수 있음.
    - mode='medium': (추천) 속도와 화질의 균형. 흰색 점 제거 및 기본적인 텍스처 보존 우수.
    - mode='large': 화질 우선, 하지만 1080p에서는 프레임 드랍 가능성 있음.
    """
    def __init__(self, mode='medium', dim=24):
        
        # 레이어 깊이 설정
        if mode == 'small':
            num_layers = 2
        elif mode == 'medium':
            num_layers = 5  # Q8550 추천
        elif mode == 'large':
            num_layers = 11
        else:
            raise ValueError("mode must be 'small', 'medium', or 'large'")

        super().__init__(
            scaling_factor=1,           # 해상도 유지
            num_channels=dim,           # 채널 수 (32가 모바일 NPU에서 효율적)
            num_intermediate_layers=num_layers,
            use_ito_connection=True,    # Denoising은 Skip Connection 필수
            in_channels=3,
            out_channels=3
        )

class QuickDenoiseOpt(nn.Module):
    """
    [Qualcomm QCS8550 Optimized Denoise Model]
    
    Optimizations:
    1. Channel Alignment: dim=32 (Default) matches NPU vector width (32/64 bytes).
    2. Zero Overhead Skip: Removed 'AnchorOp'. Uses direct tensor addition.
    3. Operation Fusion: Standard Conv+Bias+Hardtanh sequence for perfect NPU folding.
    4. Removed PixelShuffle: Direct Channel-to-Channel mapping for 1:1 scale.
    """
    def __init__(self, mode='medium', dim=32):
        super().__init__()
        
        # --- 1. Mode Configuration ---
        # medium: 5 layers (Recommend for 1080p/30fps on QCS8550)
        # small:  2 layers (Ultra fast, for 4K or 60fps)
        if mode == 'small':
            num_body_layers = 2
        elif mode == 'medium':
            num_body_layers = 5 
        elif mode == 'large':
            num_body_layers = 11
        else:
            raise ValueError("mode must be 'small', 'medium', or 'large'")

        # --- 2. Architecture Definition ---
        
        # [Head]: Input(3) -> Feature(dim)
        # padding=1 ensures output size equals input size
        self.head = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1, bias=True),
            nn.Hardtanh(0., 1.) # NPU friendly activation
        )

        # [Body]: Feature(dim) -> Feature(dim)
        # Stacking Conv + Act layers
        body_layers = []
        for _ in range(num_body_layers):
            body_layers.extend([
                nn.Conv2d(dim, dim, 3, padding=1, bias=True),
                nn.Hardtanh(0., 1.)
            ])
        self.body = nn.Sequential(*body_layers)

        # [Tail]: Feature(dim) -> Residual(3)
        # Projects back to image space. 
        # Note: No activation here yet. We add residual first.
        self.tail = nn.Conv2d(dim, 3, 3, padding=1, bias=True)

        # [Final Activation]
        # Ensures output is within valid image range [0, 1]
        # This acts as the Quantization Clipping node for the output.
        self.final_clip = nn.Hardtanh(0., 1.)

        # --- 3. Weight Initialization ---
        self._initialize_weights()

    def forward(self, x):
        """
        Forward Pass: Out = Clip(Input + Residual(Input))
        """
        # 1. Main Branch (Calculate Noise/Residual)
        feat = self.head(x)
        feat = self.body(feat)
        residual = self.tail(feat)

        # 2. Skip Connection (Input Add)
        # Original 'AnchorOp' is removed. Direct addition is faster.
        # NPU handles Element-wise Add very efficiently.
        out = x + residual

        # 3. Final Clip
        return self.final_clip(out)

    def _initialize_weights(self):
        """
        Initialize weights to behave like an Identity Mapper at the start.
        This helps faster convergence for Denoising tasks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Standard Initialization (He or Xavier is fine, using simplified here)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Last Layer Initialization strategy:
        # Make the initial residual close to 0, so the network starts as Identity.
        # This is crucial for Denoising/SR.
        with torch.no_grad():
            self.tail.weight.data.mul_(0.1) 
            if self.tail.bias is not None:
                self.tail.bias.data.zero_()
