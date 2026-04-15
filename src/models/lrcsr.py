import torch
import torch.nn as nn
import torch.nn.functional as F

class RepConv(nn.Module):
    """
    RepConv with LeakyReLU:
    - Better for Super-Resolution (Preserves negative gradients/features)
    - Slightly heavier than ReLU on NPU, but worth the quality gain.
    """
    def __init__(self, in_channels, out_channels):
        super(RepConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = False
        
        # 1. Left Branch (Expanded)
        self.left_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )
        
        # 2. Right Branch (Identity-like)
        self.right_branch = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # [최종 결정] Activation: LeakyReLU
        # negative_slope=0.2는 SR 연구들(ESRGAN 등)에서 검증된 값입니다.
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.reparam_conv = None

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam_conv(x))
        else:
            return self.act(self.left_branch(x) + self.right_branch(x))

    def get_equivalent_kernel_bias(self):
        # (기존 병합 로직 동일)
        identity_input = torch.zeros(self.in_channels, self.in_channels, 3, 3, device=self.right_branch.weight.device)
        for i in range(self.in_channels):
            identity_input[i, i, 1, 1] = 1.0
        zero_input = torch.zeros_like(identity_input)

        out_left_I = self.left_branch(identity_input)
        out_left_Z = self.left_branch(zero_input)
        k_left = out_left_I - out_left_Z
        b_left = out_left_Z

        out_right_I = self.right_branch(identity_input)
        out_right_Z = self.right_branch(zero_input)
        k_right = out_right_I - out_right_Z
        b_right = out_right_Z
        
        total_kernel_raw = k_left + k_right
        final_kernel = total_kernel_raw.permute(1, 0, 2, 3)
        total_bias_raw = b_left + b_right
        final_bias = total_bias_raw[0, :, 1, 1]

        return final_kernel, final_bias

    def switch_to_deploy(self):
        if self.deploy: return
        kernel, bias = self.get_equivalent_kernel_bias()
        
        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        
        del self.left_branch
        del self.right_branch
        self.deploy = True


class LRCSR(nn.Module):
    def __init__(self, scale_factor=2, dim=32):
        super(LRCSR, self).__init__()
        
        self.head = nn.Conv2d(3, dim, 3, padding=1)

        self.body = nn.Sequential(
            RepConv(dim, dim),
            RepConv(dim, dim),
            RepConv(dim, dim),
            RepConv(dim, dim)
        )

        self.tail = nn.Conv2d(dim, 3 * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        self.initialize()

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.body(x_head)
        
        # LeakyReLU를 사용하므로 Residual Addition 시 음수 정보가 잘 보존되어 유리합니다.
        x_res = x_body + x_head 
        
        out = self.tail(x_res)
        out = self.pixel_shuffle(out)
        
        # [Output] 0~1 Range Clamping (NPU Safety)
        out = torch.clamp(out, 0.0, 1.0)                
        return out

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepConv):
                m.switch_to_deploy()
                
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        for m in self.modules():
            if isinstance(m, RepConv):
                nn.init.zeros_(m.right_branch.weight)
                if m.right_branch.bias is not None: nn.init.zeros_(m.right_branch.bias)
                
                for i in range(min(m.in_channels, m.out_channels)):
                    m.right_branch.weight.data[i, i, 0, 0] = 1.0
                
                for layer in m.left_branch:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.normal_(layer.weight, std=1e-3)
                        if layer.bias is not None: nn.init.zeros_(layer.bias)
