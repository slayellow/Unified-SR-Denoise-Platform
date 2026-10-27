import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def icnr_(weight, factor):
    """Initialise a ConvTranspose kernel so it starts as nearest-neighbour upsampling.

    A ConvTranspose output at position ``s*n + r`` collects the taps ``kk`` with
    ``kk % s == (r + p) % s`` for padding ``p``. With ``kernel = 2*stride`` every
    output of a block reads the same two input positions, so making the weight
    depend only on ``kk // s`` -- not on ``kk % s`` -- makes all ``s`` outputs of
    a block identical. The blocks start at output offset ``s - p``, not on the
    input grid (offset 1 at x2 and x3, 2 at x4 with this model's padding). The
    layer then begins life free of checkerboard and only picks up sub-pixel
    detail as training pulls the groups apart. This is the ConvTranspose analogue
    of ICNR (Aitken et al.), which does the same thing for PixelShuffle.
    """
    in_ch, _, kh, kw = weight.shape
    sh, sw = kh // factor, kw // factor
    # Replication means an output pixel only ever sums in_ch * sh * sw products,
    # not the in_ch * kh * kw the kernel shape suggests, so the bound is
    # 1/sqrt(in_ch * sh * sw) (0.088 for 32 channels). That is not PyTorch's
    # ConvTranspose default, which counts out_ch * kh * kw: 0.144 / 0.096 / 0.072
    # at x2 / x3 / x4.
    fan_in = in_ch * sh * sw
    bound = 1.0 / math.sqrt(fan_in)
    seed = torch.empty(in_ch, weight.shape[1], sh, sw).uniform_(-bound, bound)
    with torch.no_grad():
        for i in range(kh):
            for j in range(kw):
                weight[:, :, i, j] = seed[:, :, i // factor, j // factor]


def split_factor(factor):
    """Smallest first stage that splits `factor` into two blocksizes >= 2.

    Returns None when `factor` is prime, i.e. when no two-stage split exists.
    """
    for first in range(2, int(factor**0.5) + 1):
        if factor % first == 0:
            return first, factor // first
    return None


class ECBConv(nn.Module):
    """A 3x3 conv that trains as several branches and deploys as one 3x3 conv.

    The branches are summed, and every one of them is linear and bias-free, so
    the sum is a single convolution -- the fusion is exact arithmetic, not an
    approximation. (A bias would not fold at the zero-padded border, so ECBConv
    refuses one.)
    Deploying it therefore costs nothing: same op, same shape, same DLC, same
    latency. Only training sees the extra width.

    That the branches help at all is the part worth stating. The plain 3x3 has
    to discover edge-detecting structure from scratch; the Sobel and Laplacian
    branches hand it that structure fixed and let training set only how much of
    it to use. On this project that is the exact axis under repair -- gradient
    reproduction against GT -- and it is why the edge-oriented ECB (Zhang et
    al., ACM MM 2021) is used here rather than the RepVGG-style 1x1+identity
    block: measured on the same ablation, ECB was worth +0.1 dB against
    RepVGG's +0.02 on super-resolution.

    (Some references give ECBSR an arXiv id of 2108.11341. That id belongs to
    an unrelated physics paper; ECBSR has no arXiv entry.)

    The fixed kernels are buffers, not parameters: they are the prior, and the
    per-channel 1x1 in front of each is what learns. `identity` is only
    available when in_ch == out_ch.
    """

    def __init__(self, in_ch, out_ch, bias=False, identity=True):
        super().__init__()
        if bias:
            # A constant entering a fixed kernel cancels inside the image but not at the
            # zero-padded border, so a bias branch does not fold into the 3x3.
            raise ValueError("ECBConv folds exactly only without bias; use bias=False with use_rep")
        self.in_ch, self.out_ch = in_ch, out_ch
        self.use_identity = bool(identity) and in_ch == out_ch
        self.deploy = False

        self.conv3x3 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias)
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        # Each prior gets its own 1x1 so the branch can be scaled per output
        # channel; the 3x3 half is fixed.
        self.sobel_x = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.sobel_y = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.laplacian = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

        self.register_buffer('k_sobel_x', torch.tensor(
            [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('k_sobel_y', torch.tensor(
            [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('k_laplacian', torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3) / 4.0)

    def _prior(self, x, conv1x1, kernel):
        """1x1 mixes channels, then the fixed 3x3 runs depthwise on the result.

        Depthwise keeps it to out_ch * 9 multiply-adds instead of
        out_ch^2 * 9, and it is what makes the fusion a plain outer product of
        the 1x1 weight with the fixed kernel.
        """
        y = conv1x1(x)
        k = kernel.to(y.dtype).repeat(self.out_ch, 1, 1, 1)
        return F.conv2d(y, k, padding=1, groups=self.out_ch)

    def forward(self, x):
        if self.deploy:
            return self.conv3x3(x)
        y = self.conv3x3(x) + self.conv1x1(x)
        y = y + self._prior(x, self.sobel_x, self.k_sobel_x)
        y = y + self._prior(x, self.sobel_y, self.k_sobel_y)
        y = y + self._prior(x, self.laplacian, self.k_laplacian)
        if self.use_identity:
            y = y + x
        return y

    @torch.no_grad()
    def switch_to_deploy(self):
        """Collapse every branch into conv3x3. Idempotent."""
        if self.deploy:
            return
        w = self.conv3x3.weight.data.clone()
        # 1x1 is a 3x3 whose outer ring is zero.
        w += F.pad(self.conv1x1.weight.data, (1, 1, 1, 1))
        # A per-output-channel 1x1 followed by a depthwise fixed kernel is,
        # per output channel o and input channel i, weight[o, i] * kernel.
        for conv, buf in ((self.sobel_x, self.k_sobel_x),
                          (self.sobel_y, self.k_sobel_y),
                          (self.laplacian, self.k_laplacian)):
            w += conv.weight.data.view(self.out_ch, self.in_ch, 1, 1) * buf.view(1, 1, 3, 3)
        if self.use_identity:
            eye = torch.zeros_like(w)
            for c in range(self.out_ch):
                eye[c, c, 1, 1] = 1.0
            w += eye
        self.conv3x3.weight.data.copy_(w)

        for name in ('conv1x1', 'sobel_x', 'sobel_y', 'laplacian'):
            delattr(self, name)
        self.deploy = True


class BicubicPlusPlus(nn.Module):
    """Bicubic++-style lightweight SR network.

    ``upsample_mode`` selects how features return to full resolution:

    ``legacy``
        The original design: a stride-2 stem shrinks the input, the body runs at
        half resolution, and a single PixelShuffle makes up a factor of
        ``2 * scaling_factor``. Cheap in MACs but expensive on Hexagon HTP,
        where a DepthToSpace costs roughly ``blocksize**2`` -- measured at
        ~291k cycles per unit of ``blocksize**2`` for a 1280x720 output. At
        scale 4 that blocksize is 8 and the shuffle alone took 17.5M cycles,
        98% of the whole network.

    ``auto``
        Drops the stride-2 stem so the shuffle only has to cover
        ``scaling_factor``, and splits it into two stages when that factor is
        composite. The body then runs at full input resolution, which costs
        more MACs, but the shuffle shrinks by 4x (scale 2 and 3) to 5.5x
        (scale 4). Per scale this gives:

        =====  ==========================  =========================
        scale  legacy                      auto
        =====  ==========================  =========================
        2      stride2 + PixelShuffle(4)   PixelShuffle(2)
        3      stride2 + PixelShuffle(6)   PixelShuffle(3)
        4      stride2 + PixelShuffle(8)   PixelShuffle(2) x2
        =====  ==========================  =========================

    ``deconv``
        Drops the stride-2 stem like ``auto`` but replaces conv + PixelShuffle
        with a single ConvTranspose. DepthToSpace only reaches the HTP fast path
        when the blocksize divides the 8x8 crouton chunk, so an odd scale such as
        3 is stuck ~7x off the cost curve no matter how the tensor is sized;
        ConvTranspose sidesteps DepthToSpace entirely and measured 5.24 ms at
        scale 3 against 9.18 ms for PixelShuffle(3).

    ``legacy`` is the default. No legacy SR checkpoint remains; BicubicPPDenoise[v1]
    uses legacy at scale 1 and names it in its config.

    ``scaling_factor=1`` (EO denoise, docs/modules/20261012-MOD-BicubicPPDenoise.md)
        ``legacy`` runs the body at half resolution and returns with
        PixelShuffle(2); ``auto`` runs it at full resolution with no shuffle at
        all (a PixelShuffle(1) would export as DepthToSpace(1)). Both add the
        input back (output = clamp(x + residual)) and scale the residual's last
        layer by 0.01, so the network begins close to the identity, as
        SVFocusDenoiseNet does.
        ``n_body`` sets the number of 3x3 body convs after the stem (2 = the SR
        body; more only at scale 1). ECB is allowed on every stride-1 conv here.
        The half-resolution form needs even height and width (an odd size fails
        at the input add); no shape check sits in forward, so torch.fx / AIMET
        tracing stays free of data-dependent control flow.
    """

    def __init__(
        self,
        scaling_factor=3,
        n_feats=32,
        in_channels=3,
        out_channels=3,
        bias=False,
        negative_slope=0.1,
        clamp_output=True,
        upsample_mode="legacy",
        mid_feats=16,
        icnr_init=True,
        achromatic_output=False,
        use_rep=False,
        act="leaky_relu",
        n_body=2,
    ):
        super().__init__()
        if scaling_factor != int(scaling_factor):
            raise ValueError(f"BicubicPlusPlus needs an integer scaling_factor, got {scaling_factor!r}")
        if int(scaling_factor) < 1:
            raise ValueError(f"BicubicPlusPlus needs scaling_factor >= 1, got {scaling_factor}")
        if upsample_mode not in ("legacy", "auto", "deconv"):
            raise ValueError(f"unknown upsample_mode: {upsample_mode!r}")
        self.denoise = int(scaling_factor) == 1
        if self.denoise and upsample_mode == "deconv":
            raise ValueError("scaling_factor 1 (denoise) supports upsample_mode legacy or auto, not deconv")
        if self.denoise and achromatic_output:
            raise ValueError("achromatic_output is an IR SR option; the scale-1 denoiser keeps colour")
        if int(n_body) < 2 or (int(n_body) != 2 and not self.denoise):
            raise ValueError(f"n_body must be 2 for SR and >= 2 for scale 1, got {n_body}")
        if act not in ("leaky_relu", "relu"):
            raise ValueError(f"unknown act: {act!r}")
        if use_rep and upsample_mode == "legacy" and not self.denoise:
            # use_rep used to be switched off here without a word, training a model without ECB.
            raise ValueError("use_rep needs upsample_mode auto or deconv for SR; legacy SR has no ECB form")

        self.scaling_factor = int(scaling_factor)
        # True by default: every shipped model ends on Hardtanh 'final_act', the output layer
        # the HTP conversion (--set_output_layers) and gst (layers=) name.
        self.clamp_output = bool(clamp_output)
        # Thermal imagery is achromatic by construction: the IR ground truth and
        # the bicubic anchor both measure 0.000 DN of channel deviation. This
        # model does not, because its three output channels come from three
        # independent filters and nothing ties them together -- the colour is
        # inherited from the Phase-1 pretrain on colour photographs and survives
        # an IR finetune (measured 0.35 DN mean, 17.85 DN peak, which reads as a
        # colour cast over a grey wall). No loss term causes it: the IR recipe
        # already runs edge mode='mean' and perceptual off, both exactly 0%
        # chroma. Tying the channels in forward makes one shared filter learn
        # the achromatic output, and because the last layer is linear the mean
        # folds back into its weights in switch_to_deploy (_fold_achromatic), so
        # the deployed graph keeps three output channels and gains no ops.
        self.achromatic_output = bool(achromatic_output)
        self.upsample_mode = upsample_mode

        stem_stride = 2 if upsample_mode == "legacy" else 1
        factor = self.scaling_factor * stem_stride

        # ECB widens each 3x3 during training and folds back to exactly that
        # 3x3 before export, so the deployed graph is byte-identical either
        # way. Only usable at stride 1: a strided fixed kernel is not the same
        # operator as the strided 3x3 it would be folded into, so conv3() keeps
        # a strided stem plain. Legacy SR has no ECB form at all (rejected
        # above); the scale-1 denoiser puts ECB on every stride-1 conv.
        self.use_rep = bool(use_rep)
        self.rep_folded = False

        def conv3(cin, cout, stride=1, identity=True):
            if self.use_rep and stride == 1:
                return ECBConv(cin, cout, bias=bias, identity=identity)
            return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=bias)

        # conv0 changes channel count (3 -> n_feats), so no identity branch.
        self.conv0 = conv3(in_channels, n_feats, stride=stem_stride, identity=False)
        self.conv1 = conv3(n_feats, n_feats)
        self.conv2 = conv3(n_feats, n_feats)
        self.body_extra = nn.ModuleList(conv3(n_feats, n_feats) for _ in range(int(n_body) - 2)) if int(n_body) > 2 else None
        # relu exists to measure HTP latency against LeakyReLU; every shipped
        # checkpoint was trained with LeakyReLU(0.1), so that stays the default.
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(inplace=True, negative_slope=negative_slope)

        split = split_factor(factor) if upsample_mode == "auto" else None
        self.upsample_split = split is not None

        if upsample_mode == "deconv":
            # out = (in-1)*f - 2p + k + output_padding, and we want exactly in*f.
            # Fixing k = 2f keeps kernel/stride at 2, so every output position
            # collects the same number of taps and the uneven overlap behind
            # checkerboard artefacts cannot arise. p = (f+1)//2 is the smallest
            # padding that keeps output_padding non-negative for both parities.
            padding = (factor + 1) // 2
            self.deconv = nn.ConvTranspose2d(
                n_feats,
                out_channels,
                kernel_size=2 * factor,
                stride=factor,
                padding=padding,
                output_padding=2 * padding - factor,
                bias=bias,
            )
            if icnr_init:
                icnr_(self.deconv.weight, factor)
        elif self.upsample_split:
            first, second = split
            self.conv_out = conv3(n_feats, mid_feats * first**2, identity=False)
            self.pixel_shuffle = nn.PixelShuffle(first)
            self.conv_out2 = conv3(mid_feats, out_channels * second**2, identity=False)
            self.pixel_shuffle2 = nn.PixelShuffle(second)
        else:
            self.conv_out = conv3(n_feats, out_channels * factor**2, identity=False)
            self.pixel_shuffle = nn.PixelShuffle(factor) if factor > 1 else nn.Identity()

        if self.clamp_output:
            self.final_act = nn.Hardtanh(0.0, 1.0)
        if self.denoise:
            # Residual starts near zero so the denoiser begins as the identity.
            with torch.no_grad():
                for p in self.conv_out.parameters():
                    p.mul_(0.01)

    def forward(self, x):
        x0 = self.act(self.conv0(x))
        x1 = self.act(self.conv1(x0))
        x2 = self.act(self.conv2(x1))
        if self.body_extra is not None:
            for conv in self.body_extra:
                x2 = self.act(conv(x2))
        x2 = x2 + x0

        if self.upsample_mode == "deconv":
            y = self.deconv(x2)
        else:
            y = self.conv_out(x2)
            if self.upsample_split:
                # LeakyReLU is elementwise and PixelShuffle is a permutation, so
                # the two commute exactly. Activating first keeps the work on the
                # wide pre-shuffle tensor (mid_feats * first**2 channels) rather
                # than the narrow mid_feats one, which HTP pads out to a
                # 32-channel crouton and then spends half its cycles on padding:
                # measured 1.28M cycles, 31% of the x4 network, for what x2's
                # equivalent does in 0.9M on twice the elements.
                y = self.pixel_shuffle(self.act(y))
                y = self.pixel_shuffle2(self.conv_out2(y))
            else:
                y = self.pixel_shuffle(y)
        if self.achromatic_output:
            y = y.mean(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)
        if self.denoise:
            y = x + y
        if self.clamp_output:
            y = self.final_act(y)
        return y

    def switch_to_deploy(self):
        """Fold every ECB branch into its 3x3, then an IR model's channel mean into the last layer.

        Idempotent; a no-op without ECB and without achromatic_output.

        Order matters and is fixed by what reads the graph afterwards. This runs
        FIRST in the AIMET chain -- src/aimet/quant_sim.py calls it from
        prepare_deploy_model() before prepare_model / BN fold / CLE -- because
        the quantizer set and every entry in the .encodings file are decided by
        the graph that exists at that moment. Folding after QuantSim would leave
        encodings attached to branches the exported ONNX does not contain.

            train (ECB branches) -> switch_to_deploy [ECB fold, then IR achromatic fold]
              -> prepare_model -> BN fold -> CLE -> QuantSim -> QAT -> DLC

        Checkpoints therefore come in two incompatible shapes. `use_rep` in the
        stored config says which one a file holds; loading a folded checkpoint
        into an unfolded model raises on the missing branch keys rather than
        silently starting from noise -- the failure mode that nearly cost two
        days of pretrain in August.
        """
        if not self.rep_folded:
            # Fold, then put the bare nn.Conv2d back where the ECBConv was. Leaving
            # the wrapper in place would work numerically but rename every weight
            # to `conv0.conv3x3.weight`, so a folded checkpoint would no longer
            # load into the plain model and the .encodings keys would stop matching
            # the ones the shipped v2 DLCs were built from. After this the module
            # tree is indistinguishable from use_rep=False.
            for name, child in list(self.named_children()):
                if isinstance(child, ECBConv):
                    child.switch_to_deploy()
                    setattr(self, name, child.conv3x3)
                elif isinstance(child, nn.ModuleList):
                    for i, conv in enumerate(child):
                        if isinstance(conv, ECBConv):
                            conv.switch_to_deploy()
                            child[i] = conv.conv3x3
            self.rep_folded = True
        # After the ECB fold: the achromatic fold reads the last layer's single .weight.
        if self.achromatic_output:
            self._fold_achromatic()

    @torch.no_grad()
    def _fold_achromatic(self):
        """Collapse the three output filters into one, in place, and drop the mean from forward.

        That average must NOT reach the deployed graph. Left in, torch.onnx exports it as
        ReduceMean / ConstantOfShape / Mul / Equal / Where / Expand -- six nodes over the full
        HxWx3 output that fall back to float on HTP. Measured on device at x3: 2.746 ms
        without them, 10.401 ms with, at the same W8A8.

        The last layer is linear, so the average folds into its weights exactly:

            y_c = W_c * x   ->  set every W_c to mean_c(W),  giving y_c = mean_c(y)

        Same arithmetic, same op count, and the graph goes back to what the EO models export.
        """
        if self.upsample_mode == "deconv":
            # ConvTranspose2d weight is (in, out, kh, kw) -- output channels are dim 1.
            layer, w = self.deconv, self.deconv.weight.data
            w_new = w.mean(dim=1, keepdim=True).expand_as(w).contiguous()
        else:
            # PixelShuffle path: the last conv emits 3*r^2 channels and the shuffle
            # hands channels [c*r^2:(c+1)*r^2] to output channel c, so the three
            # GROUPS are averaged, not individual channels.
            layer = self.conv_out2 if self.upsample_split else self.conv_out
            w = layer.weight.data
            if w.shape[0] % 3:
                raise ValueError(f"final conv has {w.shape[0]} output channels, not a multiple of 3")
            g = w.view(3, w.shape[0] // 3, *w.shape[1:])
            w_new = g.mean(dim=0, keepdim=True).expand_as(g).reshape_as(w).contiguous()
        layer.weight.data = w_new
        if layer.bias is not None:
            b = layer.bias.data
            if b.numel() % 3 == 0 and b.numel() > 3:
                bg = b.view(3, -1)
                layer.bias.data = bg.mean(dim=0, keepdim=True).expand_as(bg).reshape(-1).contiguous()
            else:
                layer.bias.data = b.mean().expand_as(b).contiguous()
        self.achromatic_output = False
