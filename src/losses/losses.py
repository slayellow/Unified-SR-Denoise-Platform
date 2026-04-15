import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp

# ==========================================
# 1. Charbonnier Loss
# ==========================================
class CharbonnierLoss(nn.Module):
    """L1 Loss variation that is more robust to outliers"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss_map = torch.sqrt(diff * diff + self.eps**2)
        return loss_map.mean()

# ==========================================
# 2. Edge Loss
# ==========================================
class EdgeLoss(nn.Module):
    """
    Computes loss on Sobel gradients with an optional threshold mask.
    Supports 'gray' (convert to grayscale first) or 'rgb' mode.
    """
    def __init__(self, mode='gray', threshold=0.2): # threshold 파라미터 추가
        super(EdgeLoss, self).__init__()
        self.mode = mode
        self.threshold = threshold
        
        # Sobel Kernel
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        
        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)

    def rgb_to_gray(self, x):
        # ITU-R BT.601
        return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

    def forward(self, pred, target):
        if self.mode == 'gray':
            pred = self.rgb_to_gray(pred)
            target = self.rgb_to_gray(target)
            groups = 1
            k_x = self.k_x
            k_y = self.k_y
        else:
            # RGB mode: Apply to each channel independently
            groups = 3
            k_x = self.k_x.repeat(3, 1, 1, 1)
            k_y = self.k_y.repeat(3, 1, 1, 1)

        # Sobel Filtering
        pred_grad_x = F.conv2d(pred, k_x, padding=1, groups=groups)
        pred_grad_y = F.conv2d(pred, k_y, padding=1, groups=groups)
        
        target_grad_x = F.conv2d(target, k_x, padding=1, groups=groups)
        target_grad_y = F.conv2d(target, k_y, padding=1, groups=groups)
        
        # Magnitude
        pred_grad = torch.abs(pred_grad_x) + torch.abs(pred_grad_y)
        target_grad = torch.abs(target_grad_x) + torch.abs(target_grad_y)
        
        # --- 핵심 수정 부분 (Threshold Masking) ---
        if self.threshold > 0.0:
            # Target(정답) 이미지 기준으로 진짜 엣지만 1로 만드는 마스크 생성
            # gradient 값은 픽셀 범위(0~1)에 따라 달라질 수 있으므로 threshold 튜닝 필요
            edge_mask = (target_grad > self.threshold).float()
            
            # 마스크가 씌워진 영역(진짜 엣지)에 대해서만 Loss 계산
            pred_grad = pred_grad * edge_mask
            target_grad = target_grad * edge_mask
            
            # 전체 픽셀 수가 아닌 마스크된 엣지 픽셀 수로 나누기 위해 Sum 후 평균 계산
            # (만약 배경 평탄화도 유지하고 싶다면 아래 주석 처리된 일반 L1 Loss를 쓰셔도 됩니다)
            loss = F.l1_loss(pred_grad, target_grad, reduction='sum') / (edge_mask.sum() + 1e-8)
        else:
            loss = F.l1_loss(pred_grad, target_grad)
            
        return loss

# ==========================================
# 3. SSIM Loss
# ==========================================
class SSIMLoss(nn.Module):
    """Structural Similarity Loss"""
    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        return 1.0 - self._ssim(img1, img2, self.window, self.window_size, self.channel)

# ==========================================
# 4. Perceptual Loss (VGG)
# ==========================================
class PerceptualLoss(nn.Module):
    """VGG19 based Perceptual Loss"""
    def __init__(self, layer_idx=34):
        super(PerceptualLoss, self).__init__()
        # Load VGG19
        vgg = models.vgg19(pretrained=True)
        # Use features up to layer_idx (default 34 is relu5_4)
        loss_network = nn.Sequential(*list(vgg.features)[:layer_idx+1]).eval()
        
        for param in loss_network.parameters():
            param.requires_grad = False
            
        self.loss_network = loss_network
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Normalize
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract Features
        pred_feat = self.loss_network(pred_norm)
        target_feat = self.loss_network(target_norm)
        
        
        return F.l1_loss(pred_feat, target_feat)

# ==========================================
# 5. TV Loss (Total Variation)
# ==========================================
class TVLoss(nn.Module):
    """
    Total Variation Loss to reduce checkerboard artifacts (noise).
    """
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x, y=None):
        # TV Loss usually calculates smoothness of the prediction (x) itself, 
        # independent of ground truth (y). But can be used with y if needed.
        # Here standard TV on prediction:
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# ==========================================
# 6. Contrastive Loss (InfoNCE) - for CoReFusion
# ==========================================
class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss.
    Takes two projection feature tensors (B, D) and maximizes
    agreement between corresponding pairs.
    """
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: (B, D) — L2 normalized projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        B = z1.shape[0]
        logits = torch.mm(z1, z2.t()) / self.temperature  # (B, B)
        labels = torch.arange(B, device=z1.device)
        
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss

# ==========================================
# 7. GAN Loss - for LapGSR
# ==========================================
class GANLoss(nn.Module):
    """
    LSGAN loss (MSE-based).
    real_label=1.0, fake_label=0.0
    """
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


def gradient_penalty(discriminator, real, fake, device):
    """WGAN-GP gradient penalty."""
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    d_interp = discriminator(interpolated)
    
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.view(B, -1)
    penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# ==========================================
# 8. Unified Loss Wrapper
# ==========================================
class UnifiedLoss(nn.Module):
    """
    Wrapper to manage multiple losses dynamically from config.
    Config schema:
      loss:
        l1: {enabled: true, weight: 1.0}
        charbonnier: {enabled: false, weight: 1.0, eps: 1e-3}
        edge: {enabled: true, weight: 0.05, mode: 'gray'}
        ssim: {enabled: false, weight: 0.1}

        perceptual: {enabled: false, weight: 0.01}
        tv: {enabled: true, weight: 0.1}
    """
    def __init__(self, config=None):
        super(UnifiedLoss, self).__init__()
        self.loss_funcs = nn.ModuleDict()
        self.weights = {}
        self.config = config if config else {}
        
        if config is None: config = {}
        
        # 0. MSE Loss
        cfg_mse = config.get('mse', {})
        if cfg_mse.get('enabled', False):
            self.loss_funcs['mse'] = nn.MSELoss()
            self.weights['mse'] = cfg_mse.get('weight', 1.0)
        
        # 1. Standard L1 Loss
        cfg_l1 = config.get('l1', {})
        if cfg_l1.get('enabled', False):
            self.loss_funcs['l1'] = nn.L1Loss()
            self.weights['l1'] = cfg_l1.get('weight', 1.0)

        # 2. Charbonnier
        cfg_char = config.get('charbonnier', {})
        if cfg_char.get('enabled', False):
            self.loss_funcs['charbonnier'] = CharbonnierLoss(eps=cfg_char.get('eps', 1e-3))
            self.weights['charbonnier'] = cfg_char.get('weight', 1.0)
            
        # 3. Edge Loss
        cfg_edge = config.get('edge', {})
        if cfg_edge.get('enabled', False):
            self.loss_funcs['edge'] = EdgeLoss(mode=cfg_edge.get('mode', 'gray'))
            self.weights['edge'] = cfg_edge.get('weight', 0.05)
            
        # 4. SSIM Loss
        cfg_ssim = config.get('ssim', {})
        if cfg_ssim.get('enabled', False):
            self.loss_funcs['ssim'] = SSIMLoss()
            self.weights['ssim'] = cfg_ssim.get('weight', 0.1)
            
        # 5. Perceptual Loss
        cfg_perc = config.get('perceptual', {})
        if cfg_perc.get('enabled', False):
            self.loss_funcs['perceptual'] = PerceptualLoss()

            self.weights['perceptual'] = cfg_perc.get('weight', 0.01)

        # 6. TV Loss
        cfg_tv = config.get('tv', {})
        if cfg_tv.get('enabled', False):
            self.loss_funcs['tv'] = TVLoss(tv_loss_weight=1.0) # Weight handled in UnifiedLoss wrapper
            self.weights['tv'] = cfg_tv.get('weight', 0.1)

    def forward(self, pred, target):
        total_loss = 0.0
        loss_dict = {}
        
        for name, loss_fn in self.loss_funcs.items():
            l = loss_fn(pred, target)
            w = self.weights[name]
            total_loss += w * l
            loss_dict[name] = l.item()
            
        return total_loss, loss_dict
