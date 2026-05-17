import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    """Split channels into two halves and multiply them element-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free block for image restoration.

    The block uses depth-wise spatial filtering, SimpleGate multiplication,
    simplified channel attention, and a gated feed-forward branch.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet-style encoder-decoder restoration network.

    This implementation is intended as a denoise teacher in this repository.
    It preserves input resolution and predicts a residual image.
    """

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: list[int] | tuple[int, ...] = (2, 2, 4, 8),
        dec_blk_nums: list[int] | tuple[int, ...] = (2, 2, 2, 2),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
        clamp_output: bool = False,
    ):
        super().__init__()
        self.img_channel = img_channel
        self.width = width
        self.padder_size = 2 ** len(enc_blk_nums)
        self.clamp_output = clamp_output

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        channels = width
        for num_blocks in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            channels,
                            dw_expand=dw_expand,
                            ffn_expand=ffn_expand,
                            drop_out_rate=drop_out_rate,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )
            self.downs.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2, bias=True))
            channels *= 2

        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(
                    channels,
                    dw_expand=dw_expand,
                    ffn_expand=ffn_expand,
                    drop_out_rate=drop_out_rate,
                )
                for _ in range(middle_blk_num)
            ]
        )

        for num_blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            channels,
                            dw_expand=dw_expand,
                            ffn_expand=ffn_expand,
                            drop_out_rate=drop_out_rate,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

        self._initialize_denoise_residual()

    def _initialize_denoise_residual(self) -> None:
        """Start close to identity for stable denoise training."""
        with torch.no_grad():
            self.ending.weight.data.mul_(0.01)
            if self.ending.bias is not None:
                self.ending.bias.data.zero_()

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inp.shape
        del b, c

        x_in = self.check_image_size(inp)
        x = self.intro(x_in)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        out = x + x_in
        out = out[:, :, :h, :w]

        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)
        return out


def NAFNetDenoiseTeacher(**kwargs) -> NAFNet:
    """Factory alias for denoise teacher configs."""
    kwargs.setdefault("img_channel", 3)
    kwargs.setdefault("width", 32)
    kwargs.setdefault("middle_blk_num", 12)
    kwargs.setdefault("enc_blk_nums", (2, 2, 4, 8))
    kwargs.setdefault("dec_blk_nums", (2, 2, 2, 2))
    kwargs.setdefault("clamp_output", False)
    return NAFNet(**kwargs)
