import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, _, c = x.shape
    return x.transpose(1, 2).contiguous().view(b, c, h, w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("BiasFreeLayerNorm expects a single channel dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("WithBiasLayerNorm expects a single channel dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) * torch.rsqrt(sigma + 1e-5) * self.weight + self.bias


class RestormerLayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError("layer_norm_type must be 'BiasFree' or 'WithBias'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    """Multi-Dconv Head Transposed Self-Attention."""

    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        del c

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, -1, h * w)
        k = k.view(b, self.num_heads, -1, h * w)
        v = v.view(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.contiguous().view(b, -1, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ):
        super().__init__()
        self.norm1 = RestormerLayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = RestormerLayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    """
    Restormer-style high-resolution restoration network.

    This model is intended as a denoise teacher. It pads inputs to a valid
    multi-scale size and crops the output back to the original resolution.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list[int] | tuple[int, int, int, int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: list[int] | tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "BiasFree",
        dual_pixel_task: bool = False,
        clamp_output: bool = False,
    ):
        super().__init__()
        self.padder_size = 8
        self.dual_pixel_task = dual_pixel_task
        self.clamp_output = clamp_output

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        self.encoder_level1 = self._make_blocks(dim, num_blocks[0], heads[0], ffn_expansion_factor, bias, layer_norm_type)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = self._make_blocks(dim * 2, num_blocks[1], heads[1], ffn_expansion_factor, bias, layer_norm_type)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = self._make_blocks(dim * 4, num_blocks[2], heads[2], ffn_expansion_factor, bias, layer_norm_type)
        self.down3_4 = Downsample(dim * 4)
        self.latent = self._make_blocks(dim * 8, num_blocks[3], heads[3], ffn_expansion_factor, bias, layer_norm_type)

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = self._make_blocks(dim * 4, num_blocks[2], heads[2], ffn_expansion_factor, bias, layer_norm_type)

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = self._make_blocks(dim * 2, num_blocks[1], heads[1], ffn_expansion_factor, bias, layer_norm_type)

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = self._make_blocks(dim * 2, num_blocks[0], heads[0], ffn_expansion_factor, bias, layer_norm_type)
        self.refinement = self._make_blocks(
            dim * 2,
            num_refinement_blocks,
            heads[0],
            ffn_expansion_factor,
            bias,
            layer_norm_type,
        )

        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self._initialize_denoise_residual()

    @staticmethod
    def _make_blocks(
        dim: int,
        num_blocks: int,
        heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks)
            ]
        )

    def _initialize_denoise_residual(self) -> None:
        with torch.no_grad():
            self.output.weight.data.mul_(0.01)
            if self.output.bias is not None:
                self.output.bias.data.zero_()

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp_img.shape
        inp_img_padded = self.check_image_size(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img_padded)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out = self.output(out_dec_level1)
        else:
            out = self.output(out_dec_level1) + inp_img_padded

        out = out[:, :, :h, :w]
        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)
        return out


def RestormerDenoiseTeacher(**kwargs) -> Restormer:
    kwargs.setdefault("inp_channels", 3)
    kwargs.setdefault("out_channels", 3)
    kwargs.setdefault("dim", 48)
    kwargs.setdefault("num_blocks", (4, 6, 6, 8))
    kwargs.setdefault("num_refinement_blocks", 4)
    kwargs.setdefault("heads", (1, 2, 4, 8))
    kwargs.setdefault("ffn_expansion_factor", 2.66)
    kwargs.setdefault("bias", False)
    kwargs.setdefault("layer_norm_type", "BiasFree")
    kwargs.setdefault("dual_pixel_task", False)
    kwargs.setdefault("clamp_output", False)
    return Restormer(**kwargs)
