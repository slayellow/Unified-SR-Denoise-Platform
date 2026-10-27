import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp


def rgb_to_ycbcr(x):
    """Convert RGB tensor in [0, 1] to YCbCr-like channels in [0, 1]."""
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return y, cb, cr

# ==========================================
# 1. Charbonnier Loss
# ==========================================
class CharbonnierLoss(nn.Module):
    """L1 Loss variation that is more robust to outliers"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = float(eps)

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
    def __init__(self, mode='gray', threshold=0.2, signed=False):
        super(EdgeLoss, self).__init__()
        self.mode = mode
        self.threshold = float(threshold)
        # Comparing |Gx|+|Gy| is blind to the sign of the transition, so a
        # ringing edge satisfies it exactly as well as a narrower one -- and
        # ringing is the only one of the two a 8-10px receptive field can
        # actually produce at x3/x4. Measured on this project's own IR runs:
        # the magnitude form raised the fraction of pixels outside the bicubic
        # local min/max from 6.3% to 11.5% at x3 and 12.5% to 29.4% at x4,
        # while at x2 -- where the LR still carries the high frequencies -- it
        # rose only 2.8% to 4.3%. Comparing the SIGNED derivatives instead
        # charges for the overshoot lobes, which have the wrong sign.
        self.signed = bool(signed)
        
        # Sobel Kernel
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        
        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)

    def rgb_to_gray(self, x):
        # ITU-R BT.601
        return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

    def forward(self, pred, target):
        # 'mean' collapses with the channel MEAN instead of BT.601 luma. The
        # luma weights are themselves a chroma direction: measured at an
        # achromatic input, 49% of mode='gray''s gradient magnitude pushed
        # color into R=G=B frames. The mean hands all three channels an equal
        # gradient (exactly zero chroma), and on achromatic content the two
        # definitions produce the same grayscale anyway -- use 'mean' for IR.
        if self.mode == 'mean':
            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)
            groups = 1
            k_x = self.k_x
            k_y = self.k_y
        elif self.mode == 'gray':
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
        
        if self.signed:
            # Sign-aware: match the derivative field itself, so an overshoot
            # lobe is an error rather than a way to score. The threshold mask
            # is deliberately ignored here -- masking to strong edges leaves
            # zero gradient in the flat regions where halos live, which is
            # exactly where this form is supposed to be able to object.
            return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

        # Magnitude
        pred_grad = torch.abs(pred_grad_x) + torch.abs(pred_grad_y)
        target_grad = torch.abs(target_grad_x) + torch.abs(target_grad_y)
        
        if self.threshold > 0.0:
            # Keep only the target's strong edges (gradient > threshold; the scale follows the
            # 0-1 pixel range, so tune it) and average over the masked pixels, not the whole image.
            edge_mask = (target_grad > self.threshold).float()
            pred_grad = pred_grad * edge_mask
            target_grad = target_grad * edge_mask
            loss = F.l1_loss(pred_grad, target_grad, reduction='sum') / (edge_mask.sum() + 1e-8)
        else:
            loss = F.l1_loss(pred_grad, target_grad)
            
        return loss

# ==========================================
# 3. SSIM Loss
# ==========================================
class SSIMLoss(nn.Module):
    """1 - SSIM, computed by pyiqa's implementation.

    The previous hand-rolled SSIM here disagreed with the pyiqa SSIM the
    Validator reports (different windowing constants), so the quantity being
    optimized was not the quantity being logged. The loss runs on all three
    RGB channels for dense gradients; reporting stays luminance-only via
    src/metrics -- one implementation, two views.
    """
    def __init__(self):
        super(SSIMLoss, self).__init__()
        from pyiqa.archs.ssim_arch import ssim as _pyiqa_ssim
        self._ssim = _pyiqa_ssim

    def forward(self, img1, img2):
        return 1.0 - self._ssim(img1, img2, data_range=1.0).mean()

# ==========================================
# 4. Perceptual Loss (VGG)
# ==========================================
class PerceptualLoss(nn.Module):
    """VGG19 perceptual loss over one or several depths.

    `layers` maps a VGG19 features index to its weight, e.g. the LPIPS-style
    {8: 0.25, 17: 0.5, 26: 0.25} supervises relu2_2 (stride 2, hair-width
    structure) through relu4_4 (stride 8, semantics) in ONE forward pass.
    The old single-layer behavior is the default ({34: 1.0}, relu5_4) -- at
    stride 16 that layer physically cannot see hair-width detail, which is why
    experiment 11 exists; a multi-layer set removes the depth-vs-detail
    trade-off instead of just moving it.
    """
    def __init__(self, layers=None, gray=False):
        super(PerceptualLoss, self).__init__()
        if not layers:
            layers = {34: 1.0}
        self.layers = {int(k): float(v) for k, v in layers.items()}
        # gray=True: both images are collapsed to their channel MEAN and
        # replicated before VGG. The mean distributes its gradient equally to
        # all three input channels, so the loss cannot push chroma at all --
        # measured on achromatic IR, RGB-VGG gradients were tinting outputs
        # (+50% mean chroma within a handful of epochs). Channel mean, not
        # BT.601: luma weights would push along (0.299,0.587,0.114), which is
        # itself a chroma direction. For R=G=B inputs the two are identical.
        self.gray = bool(gray)
        max_idx = max(self.layers)

        vgg = models.vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:max_idx + 1]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False

        self.loss_network = loss_network
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x):
        feats = {}
        for idx, module in enumerate(self.loss_network):
            x = module(x)
            if idx in self.layers:
                feats[idx] = x
        return feats

    def forward(self, pred, target):
        if self.gray:
            pred = pred.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            target = target.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        pred_feats = self._features((pred - self.mean) / self.std)
        with torch.no_grad():
            target_feats = self._features((target - self.mean) / self.std)
        loss = 0.0
        for idx, w in self.layers.items():
            loss = loss + w * F.l1_loss(pred_feats[idx], target_feats[idx])
        return loss


# ==========================================
# 5-1. Color Consistency Loss
# ==========================================
class ColorConsistencyLoss(nn.Module):
    """
    Penalize color cast drift by matching chroma channels.
    Uses YCbCr-like conversion and compares only Cb/Cr channels.
    """
    def __init__(self, loss_type='l1'):
        super(ColorConsistencyLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        _, pred_cb, pred_cr = rgb_to_ycbcr(pred)
        _, target_cb, target_cr = rgb_to_ycbcr(target)

        if self.loss_type == 'charbonnier':
            diff_cb = torch.sqrt((pred_cb - target_cb) ** 2 + 1e-6)
            diff_cr = torch.sqrt((pred_cr - target_cr) ** 2 + 1e-6)
            return (diff_cb.mean() + diff_cr.mean()) / 2

        return (F.l1_loss(pred_cb, target_cb) + F.l1_loss(pred_cr, target_cr)) / 2


# ==========================================
# 5-2. Hot Pixel Masked Loss
# ==========================================
class HotPixelMaskedLoss(nn.Module):
    """
    Focus the loss on sparse bright outliers detected from the input image.
    This is meant for hot-pixel-like defects that are easy to miss in full-image averaging losses.
    """
    def __init__(self, threshold=0.15, base_loss='charbonnier', kernel_size=3):
        super(HotPixelMaskedLoss, self).__init__()
        self.threshold = float(threshold)
        self.base_loss = base_loss
        self.kernel_size = int(kernel_size)

    def forward(self, pred, target, input_tensor):
        if input_tensor is None:
            return pred.new_tensor(0.0)

        input_max = torch.max(input_tensor, dim=1, keepdim=True).values
        local_mean = F.avg_pool2d(input_max, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        hot_mask = ((input_max - local_mean) > self.threshold).float()

        # The mask is detected on the LR input; pred/target are at SR
        # resolution. Without this the elementwise product below is a shape
        # error for any scale > 1 -- the loss only ever worked for denoise.
        if hot_mask.shape[-2:] != pred.shape[-2:]:
            hot_mask = F.interpolate(hot_mask, size=pred.shape[-2:], mode='nearest')

        if hot_mask.sum() < 1:
            return pred.new_tensor(0.0)

        diff = pred - target
        if self.base_loss == 'l1':
            loss_map = torch.abs(diff)
        else:
            loss_map = torch.sqrt(diff * diff + 1e-6)

        masked_loss = loss_map * hot_mask
        return masked_loss.sum() / (hot_mask.sum() * pred.shape[1] + 1e-8)

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


class GradientVarianceLoss(nn.Module):
    """Charge the model for producing flatter gradients than the target.

    MSE has no term for how *varied* the gradients are, only for where they
    land, so where the edge position is uncertain the cheapest answer is to
    average -- which is the softening this project has been measuring. Abrarov
    et al. (arXiv:2202.00997) close that gap by comparing the *variance* of the
    Sobel response inside each patch: a smooth patch and a detailed patch can
    share a mean gradient, but not a gradient variance. On a 21k-parameter
    ESPCN -- Bicubic++'s own weight class -- it was worth +0.13 dB at x2.

    It is the only loss here whose objective is the quantity the field
    complaint is stated in. The deployment gate scores mean Sobel gradient
    against the anchor (examples/evaluation/score_sr_tensors.py); this trains
    against the same signal instead of hoping MSE correlates with it.

    ``mode`` follows EdgeLoss: 'mean' collapses channels by the plain mean and
    is the achromatic-safe choice for IR, 'gray' uses BT.601 luma (whose
    weights are themselves a chroma direction -- do not use it on IR), 'rgb'
    keeps the three channels separate.

    ``patch`` is the window the variance is taken over. 8 for x2 and 16 for
    x3/x4 in the paper; the default follows the smaller one because a window
    that spans several structures measures the scene, not the sharpness.
    """

    def __init__(self, patch=8, mode='mean'):
        super().__init__()
        self.patch = int(patch)
        self.mode = mode
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)

    def _collapse(self, x):
        if self.mode == 'mean':
            return x.mean(dim=1, keepdim=True), 1
        if self.mode == 'gray':
            return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]), 1
        return x, x.shape[1]

    def _patch_var(self, g):
        """Variance of g inside each non-overlapping patch, as a map.

        unfold gives every patch as a column, so the variance is one reduction
        over that axis -- no python loop over patches, and the whole thing stays
        differentiable.
        """
        p = self.patch
        # Trim rather than pad: a padded partial patch has a variance the image
        # never had, and at these sizes the remainder is a few pixels.
        h = g.shape[2] - g.shape[2] % p
        w = g.shape[3] - g.shape[3] % p
        g = g[:, :, :h, :w]
        cols = F.unfold(g, kernel_size=p, stride=p)          # B, C*p*p, L
        b, _, l = cols.shape
        cols = cols.view(b, g.shape[1], p * p, l)
        return cols.var(dim=2, unbiased=False)               # B, C, L

    def forward(self, pred, target):
        pred, groups = self._collapse(pred)
        target, _ = self._collapse(target)
        k_x = self.k_x if groups == 1 else self.k_x.repeat(groups, 1, 1, 1)
        k_y = self.k_y if groups == 1 else self.k_y.repeat(groups, 1, 1, 1)

        loss = 0.0
        for k in (k_x, k_y):
            gp = F.conv2d(pred, k, padding=1, groups=groups)
            gt = F.conv2d(target, k, padding=1, groups=groups)
            loss = loss + F.mse_loss(self._patch_var(gp), self._patch_var(gt))
        return loss


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
            self.weights['mse'] = float(cfg_mse.get('weight', 1.0))
        
        # 1. Standard L1 Loss
        cfg_l1 = config.get('l1', {})
        if cfg_l1.get('enabled', False):
            self.loss_funcs['l1'] = nn.L1Loss()
            self.weights['l1'] = float(cfg_l1.get('weight', 1.0))

        # 2. Charbonnier
        cfg_char = config.get('charbonnier', {})
        if cfg_char.get('enabled', False):
            self.loss_funcs['charbonnier'] = CharbonnierLoss(eps=cfg_char.get('eps', 1e-3))
            self.weights['charbonnier'] = float(cfg_char.get('weight', 1.0))
            
        # 3. Edge Loss
        cfg_edge = config.get('edge', {})
        if cfg_edge.get('enabled', False):
            self.loss_funcs['edge'] = EdgeLoss(
                mode=cfg_edge.get('mode', 'gray'),
                threshold=cfg_edge.get('threshold', 0.2),
                signed=cfg_edge.get('signed', False)
            )
            self.weights['edge'] = float(cfg_edge.get('weight', 0.05))
            
        # 4. SSIM Loss
        cfg_ssim = config.get('ssim', {})
        if cfg_ssim.get('enabled', False):
            self.loss_funcs['ssim'] = SSIMLoss()
            self.weights['ssim'] = float(cfg_ssim.get('weight', 0.1))
            
        # 5. Perceptual Loss
        cfg_perc = config.get('perceptual', {})
        if cfg_perc.get('enabled', False):
            # `layers` (idx -> weight) reaches the loss now; it used to be
            # dropped here, pinning every run to relu5_4 regardless of config.
            self.loss_funcs['perceptual'] = PerceptualLoss(layers=cfg_perc.get('layers'),
                                                           gray=cfg_perc.get('gray', False))
            self.weights['perceptual'] = float(cfg_perc.get('weight', 0.01))

        # 6. Color Consistency Loss
        cfg_color = config.get('color_consistency', {})
        if cfg_color.get('enabled', False):
            self.loss_funcs['color_consistency'] = ColorConsistencyLoss(
                loss_type=cfg_color.get('loss_type', 'l1')
            )
            self.weights['color_consistency'] = float(cfg_color.get('weight', 0.05))

        # 7. Hot Pixel Masked Loss
        cfg_hot = config.get('hot_pixel_masked', {})
        if cfg_hot.get('enabled', False):
            self.loss_funcs['hot_pixel_masked'] = HotPixelMaskedLoss(
                threshold=cfg_hot.get('threshold', 0.15),
                base_loss=cfg_hot.get('base_loss', 'charbonnier'),
                kernel_size=cfg_hot.get('kernel_size', 3)
            )
            self.weights['hot_pixel_masked'] = float(cfg_hot.get('weight', 0.1))

        # 8. TV Loss
        cfg_tv = config.get('tv', {})
        if cfg_tv.get('enabled', False):
            self.loss_funcs['tv'] = TVLoss(tv_loss_weight=1.0) # Weight handled in UnifiedLoss wrapper
            self.weights['tv'] = float(cfg_tv.get('weight', 0.1))

        # Gradient variance -- Phase-2 only. Phase-1 stays plain MSE so the
        # pretrain remains the baseline every later change is measured against;
        # the degradation rework (denoise-in-the-loop) already moves Phase-1
        # enough that adding a loss there would make the two indistinguishable.
        # mode defaults to 'mean' rather than 'gray': BT.601 luma weights are a
        # chroma direction, which is what put colour into achromatic IR frames
        # through the edge and perceptual terms in August.
        cfg_gv = config.get('gradient_variance', {})
        if cfg_gv.get('enabled', False):
            self.loss_funcs['gradient_variance'] = GradientVarianceLoss(
                patch=cfg_gv.get('patch', 8),
                mode=cfg_gv.get('mode', 'mean'),
            )
            self.weights['gradient_variance'] = float(cfg_gv.get('weight', 0.05))

        # Optional per-loss warm-up. A finetune starts from weights that sit at
        # the optimum of the PREVIOUS loss, so switching the objective on in one
        # step drags the model off that optimum before it learns anything: five
        # of six Phase-2 runs validated WORSE than their own starting point at
        # epoch 1, and the three that needed more than the patience budget to
        # climb back were killed still descending. Ramping the auxiliary terms
        # in from zero keeps the model at the fidelity optimum on epoch 1 and
        # lets it walk to the new one, so the run's first validation reproduces
        # the baseline instead of undercutting it.
        #
        #   perceptual: {weight: 0.05, warmup_epochs: 10}   # 0 -> 0.05 by ep 11
        #
        # Absent or 0 means the weight is constant, which is every config that
        # does not opt in.
        self.target_weights = dict(self.weights)
        self.warmup_epochs = {name: int(config.get(name, {}).get('warmup_epochs', 0) or 0)
                              for name in self.weights}

    def set_epoch(self, epoch):
        """Scale warmed-up losses for a 1-indexed training epoch.

        Epoch 1 gets exactly zero so it is a clean fidelity-only pass, and the
        target weight is reached at ``warmup_epochs + 1``. Called once per epoch
        by the Trainer; a criterion nobody calls this on keeps its configured
        weights forever.
        """
        for name, target in self.target_weights.items():
            n = self.warmup_epochs.get(name, 0)
            if n > 0:
                self.weights[name] = target * min(1.0, max(0, epoch - 1) / n)

    def active_weights(self):
        """Weights currently in force -- for logging what a warm-up is doing."""
        return {n: w for n, w in self.weights.items() if w != self.target_weights[n]}

    def forward(self, pred, target, input_tensor=None):
        if not self.loss_funcs:
            # Every term disabled means total_loss stays a python float and
            # .backward() dies with an AttributeError several frames away from
            # the config that caused it. Say so here instead.
            raise ValueError(
                "UnifiedLoss: 활성화된 손실이 하나도 없습니다. "
                "config 의 loss 블록에서 최소 하나는 enabled: true 여야 합니다."
            )
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.loss_funcs.items():
            w = self.weights[name]
            if w == 0.0:
                # Warmed-up-from-zero terms cost nothing on the epochs they are
                # switched off, but stay in the dict so the CSV keeps its shape.
                loss_dict[name] = 0.0
                continue
            if name == 'hot_pixel_masked':
                l = loss_fn(pred, target, input_tensor)
            else:
                l = loss_fn(pred, target)
            total_loss += w * l
            loss_dict[name] = l.item()

        return total_loss, loss_dict
