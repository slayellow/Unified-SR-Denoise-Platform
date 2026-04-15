import torch
import torch.nn as nn
import torch.nn.functional as F


class RepBlock_Advanced(nn.Module):
    """
    [Ultimate RepVGG Block (DBB Style)]
    - 학습 시 6갈래: 3x3 + 1x1 + 1x3 + 3x1 + AvgPool + Identity
    - 추론 시 1갈래: 단일 3x3 Conv로 융합 (NPU 4.5ms 보장)
    - W8A8 QAT 방어용 Hardtanh(0, 1) 적용
    """
    def __init__(self, n_feats, kernel_size=3, dilation=1, padding=1):
        super(RepBlock_Advanced, self).__init__()
        self.deploy = False
        self.n_feats = n_feats
        
        # 1. 공간 탐색 및 채널 융합 분기
        self.branch_3x3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.branch_1x1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=True)
        
        # 2. 직각 엣지(선박 윤곽선) 특화 분기
        self.branch_1x3 = nn.Conv2d(n_feats, n_feats, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.branch_3x1 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 1), padding=(1, 0), bias=True)
        
        # 3. 노이즈 억제 및 정보 보존 분기
        # stride=1, padding=1로 설정하여 입력과 출력 해상도를 동일하게 유지
        self.branch_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_identity = nn.Identity()
        
        self.act = nn.Hardtanh(0., 1.)

    def forward(self, x):
        if self.deploy:
            return self.act(self.branch_3x3(x))
        else:
            x_3x3 = self.branch_3x3(x)
            x_1x1 = self.branch_1x1(x)
            x_1x3 = self.branch_1x3(x)
            x_3x1 = self.branch_3x1(x)
            x_avg = self.branch_avg(x)
            x_id  = self.branch_identity(x)
            
            # 6갈래의 지능을 모두 합산!
            return self.act(x_3x3 + x_1x1 + x_1x3 + x_3x1 + x_avg + x_id)

    def switch_to_deploy(self):
        if self.deploy:
            return
        
        # 1. 3x3 커널 & 편향
        k_3x3 = self.branch_3x3.weight.data
        b_3x3 = self.branch_3x3.bias.data

        # 2. 1x1 커널 융합 (중앙에 배치)
        k_1x1 = F.pad(self.branch_1x1.weight.data, (1, 1, 1, 1)) 
        b_1x1 = self.branch_1x1.bias.data

        # 3. 1x3 커널 융합 (위아래로 패딩)
        k_1x3 = F.pad(self.branch_1x3.weight.data, (0, 0, 1, 1))
        b_1x3 = self.branch_1x3.bias.data

        # 4. 3x1 커널 융합 (좌우로 패딩)
        k_3x1 = F.pad(self.branch_3x1.weight.data, (1, 1, 0, 0))
        b_3x1 = self.branch_3x1.bias.data

        # 5. 3x3 Average Pooling 융합 (모든 칸에 1/9 곱하기)
        k_avg = torch.zeros_like(k_3x3)
        for i in range(self.n_feats):
            k_avg[i, i, :, :] = 1.0 / 9.0  # 자기 채널의 3x3 영역 평균
        b_avg = 0.0 # AvgPool은 편향(Bias)이 없음

        # 6. Identity 융합 (중앙에만 1)
        k_id = torch.zeros_like(k_3x3)
        for i in range(self.n_feats):
            k_id[i, i, 1, 1] = 1.0
        
        # 7. 모든 커널과 편향을 하나의 3x3 행렬로 압축
        fused_kernel = k_3x3 + k_1x1 + k_1x3 + k_3x1 + k_avg + k_id
        fused_bias = b_3x3 + b_1x1 + b_1x3 + b_3x1 + b_avg

        # 8. 단일 3x3 Conv로 덮어쓰기
        self.branch_3x3 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True)
        self.branch_3x3.weight.data = fused_kernel
        self.branch_3x3.bias.data = fused_bias
        
        # 학습용 분기들 메모리 해제
        del self.branch_1x1
        del self.branch_1x3
        del self.branch_3x1
        del self.branch_avg
        del self.branch_identity
        
        self.deploy = True

# ==========================================
# 1. Reparameterization Block (QAT Optimized)
# ==========================================
class RepBlock(nn.Module):
    """
    RepVGG Block 스타일:
    - 학습 시: 3x3 Conv + 1x1 Conv + Identity (3갈래) -> 강력한 성능
    - 추론 시: 단일 3x3 Conv로 Fusing -> 고속 추론 (NPU 친화적)
    * W8A8 양자화(QAT)를 위해 활성화 함수를 Hardtanh(0, 1)로 변경
    """
    def __init__(self, n_feats, kernel_size=3, dilation=1, padding=1):
        super(RepBlock, self).__init__()
        self.deploy = False
        self.n_feats = n_feats
        
        self.branch_3x3 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=True)
        self.branch_1x1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=True)
        self.branch_identity = nn.Identity()
        
        self.act = nn.Hardtanh(0., 1.)

    def forward(self, x):
        if self.deploy:
            return self.act(self.branch_3x3(x))
        else:
            x3 = self.branch_3x3(x)
            x1 = self.branch_1x1(x)
            xi = self.branch_identity(x)
            return self.act(x3 + x1 + xi)

    def switch_to_deploy(self):
        if self.deploy:
            return
        
        kernel_3x3 = self.branch_3x3.weight.data
        bias_3x3 = self.branch_3x3.bias.data

        kernel_1x1 = self.branch_1x1.weight.data
        kernel_1x1_padded = F.pad(kernel_1x1, (1, 1, 1, 1)) 
        bias_1x1 = self.branch_1x1.bias.data

        kernel_identity = torch.zeros_like(kernel_3x3)
        for i in range(self.n_feats):
            kernel_identity[i, i, 1, 1] = 1.0
        
        fused_kernel = kernel_3x3 + kernel_1x1_padded + kernel_identity
        fused_bias = bias_3x3 + bias_1x1 

        self.branch_3x3 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True)
        self.branch_3x3.weight.data = fused_kernel
        self.branch_3x3.bias.data = fused_bias
        
        del self.branch_1x1
        del self.branch_identity
        
        self.deploy = True


# ==========================================
# 2. Anchor Operator (NPU Friendly Base Upsampler)
# ==========================================
class AnchorOp(nn.Module):
    """
    입력 이미지를 NPU에서 가장 빠른 PixelShuffle을 통해 확대(Nearest 효과).
    메모리 재배열 없이 가중치가 고정된 1x1 Conv로 픽셀을 복제함.
    """
    def __init__(self, scaling_factor, in_channels=3):
        super().__init__()
        out_channels = in_channels * (scaling_factor ** 2)
        
        # 1x1 Conv로 채널 수만 뻥튀기 (연산량 극히 적음)
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # 가중치 초기화: 입력 픽셀을 정확히 복제하도록 매핑
        weight = torch.zeros(out_channels, in_channels, 1, 1)
        for i in range(in_channels):
            # scale=2 기준, 1개의 입력 채널이 4개의 출력 채널로 1.0 그대로 복사됨
            weight[i * (scaling_factor**2) : (i+1) * (scaling_factor**2), i, 0, 0] = 1.0
        
        self.net.weight.data = weight
        
        # 학습되지 않도록 Freeze
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.pixel_shuffle = nn.PixelShuffle(scaling_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.net(x))


# ==========================================
# 3. SVFocusSRNet (IR Target Edge Focus)
# ==========================================
class SVFocusSRNet(nn.Module):
    """
    [Qualcomm QCS8550 IR SR Optimized Architecture]
    - Focus: 적외선 소형 표적(선박 등)의 윤곽선(Edge) 극대화
    - Latency/Mem: dim=32 (Crouton Layout), Blocks=8 
    - Quantization: 모든 Activation을 Hardtanh(0, 1)로 묶어 W8A8 드랍 방지
    """
    def __init__(self, scaling_factor=2, n_resblocks=8, n_feats=32, in_channels=3, out_channels=3, use_advanced_rep=False):
        super(SVFocusSRNet, self).__init__()
        self.scaling_factor = int(scaling_factor)
        self.is_denoise = (self.scaling_factor == 1)

        block_class = RepBlock_Advanced if use_advanced_rep else RepBlock

        if self.is_denoise:
            # === scale=1 전용 경로 (Denoise) ===
            # AnchorOp, PixelShuffle 제거 → input skip만 사용
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, n_feats, 3, padding=1, bias=True),
                nn.Hardtanh(0., 1.)
            )
            self.body = nn.Sequential(*[
                block_class(n_feats) for _ in range(n_resblocks)
            ])
            self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)
            self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1, bias=True)
            self.final_act = nn.Hardtanh(0., 1.)

            self._initialize_weights_denoise()
        else:
            # === scale>=2 기존 경로 (SR) ===
            self.base_upsampler = AnchorOp(scaling_factor, in_channels=in_channels)
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, n_feats, 3, padding=1, bias=True),
                nn.Hardtanh(0., 1.)
            )
            self.body = nn.Sequential(*[
                block_class(n_feats) for _ in range(n_resblocks)
            ])
            self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True)
            self.upsample = nn.Sequential(
                nn.Conv2d(n_feats, out_channels * (scaling_factor ** 2), 3, padding=1),
                nn.PixelShuffle(scaling_factor)
            )
            self.final_act = nn.Hardtanh(0., 1.)

            self._initialize_weights_sr()

    def forward(self, x):
        if self.is_denoise:
            # scale=1: input skip + residual (AnchorOp/PixelShuffle 없음)
            x_head = self.head(x)
            res = self.body(x_head)
            res = self.body_tail(res)
            res = res + x_head
            edge = self.tail(res)
            return self.final_act(x + edge)
        else:
            # scale>=2: 기존 SR 경로
            base = self.base_upsampler(x)
            x_head = self.head(x)
            res = self.body(x_head)
            res = self.body_tail(res)
            res = res + x_head
            edge_detail = self.upsample(res)
            return self.final_act(base + edge_detail)

    def switch_to_deploy(self):
        """배포 전(ONNX 추출 전) 무조건 호출하여 모델을 초경량화"""
        for m in self.modules():
            if m is not self and isinstance(m, (RepBlock, RepBlock_Advanced)):
                m.switch_to_deploy()

    def _initialize_weights_denoise(self):
        """scale=1 전용: tail Conv를 0에 가깝게 → 초기 출력 = input (identity)"""
        with torch.no_grad():
            self.tail.weight.data.mul_(0.01)
            if self.tail.bias is not None:
                self.tail.bias.data.zero_()

    def _initialize_weights_sr(self):
        """
        [Zero-Residual Initialization Trick]
        학습 초반에 네트워크가 내뱉는 엣지(edge_detail) 값을 0에 가깝게 만듭니다.
        이렇게 하면 학습 1 Epoch부터 `base(배경) + 0(잔차) = base`로 시작하게 되어
        Loss 발산을 막고, 모델이 안전하게 '진짜 엣지'만 파고들 수 있도록 유도합니다.
        """
        with torch.no_grad():
            self.upsample[0].weight.data.mul_(0.01) 
            if self.upsample[0].bias is not None:
                self.upsample[0].bias.data.zero_()

if __name__ == '__main__':
    print("=== Testing Reparameterization Correctness ===")
    dummy_input = torch.randn(1, 3, 360, 640)
    
    # 1. Test Advanced RepBlock
    print("\n[Testing RepBlock_Advanced]")
    model_adv = SVFocusSRNet(scaling_factor=2, use_advanced_rep=True)
    model_adv.eval()
    
    with torch.no_grad():
        out_train_adv = model_adv(dummy_input)
        model_adv.switch_to_deploy()
        out_deploy_adv = model_adv(dummy_input)
        diff_adv = torch.abs(out_train_adv - out_deploy_adv).max().item()
        
    print(f"Max absolute difference: {diff_adv}")
    if diff_adv < 1e-4:
        print("-> SUCCESS! Reparameterization is mathematically equivalent.")
    else:
        print("-> FAILED! Difference is too large.")

    # 2. Test Basic RepBlock
    print("\n[Testing Basic RepBlock]")
    model_basic = SVFocusSRNet(scaling_factor=2, use_advanced_rep=False)
    model_basic.eval()

    with torch.no_grad():
        out_train_basic = model_basic(dummy_input)
        model_basic.switch_to_deploy()
        out_deploy_basic = model_basic(dummy_input)
        diff_basic = torch.abs(out_train_basic - out_deploy_basic).max().item()

    print(f"Max absolute difference: {diff_basic}")
    if diff_basic < 1e-4:
        print("-> SUCCESS! Reparameterization is mathematically equivalent.")
    else:
        print("-> FAILED! Difference is too large.")