
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

PIXELS_PER_SUBPATCH = 16


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return (x.float() * scale).to(x.dtype) * self.weight


def rotary(x, positions):
    dim = x.shape[-1]
    freq = torch.exp(torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) * (-math.log(500000.0) / dim))
    theta = positions.float()[:, None] * freq
    cos = torch.cat([theta.cos(), theta.cos()], -1)[None, None]
    sin = torch.cat([theta.sin(), theta.sin()], -1)[None, None]
    a, b = x.chunk(2, -1)
    return x * cos.to(x.dtype) + torch.cat([-b, a], -1) * sin.to(x.dtype)


class CausalLlamaBlock(nn.Module):
    def __init__(self, width=4096, intermediate=14336, heads=32, kv_heads=8):
        super().__init__()
        self.width, self.heads, self.kv_heads, self.dim = width, heads, kv_heads, width // heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, kv_heads * self.dim, bias=False)
        self.v = nn.Linear(width, kv_heads * self.dim, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        b, n, c = x.shape
        z = self.norm1(x)
        q = self.q(z).view(b, n, self.heads, self.dim).transpose(1, 2)
        k = self.k(z).view(b, n, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(z).view(b, n, self.kv_heads, self.dim).transpose(1, 2)
        positions = torch.arange(n, device=x.device)
        q, k = rotary(q, positions), rotary(k, positions)
                                                                                           
        k = k.repeat_interleave(self.heads // self.kv_heads, 1)
        v = v.repeat_interleave(self.heads // self.kv_heads, 1)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + self.o(attention.transpose(1, 2).reshape(b, n, c))
        z = self.norm2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(16, outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(16, outputs), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class SpatialUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0, self.e1, self.e2 = ConvBlock(256, 256), ConvBlock(256, 512), ConvBlock(512, 1024)
        self.d1, self.d0 = ConvBlock(1024 + 512, 512), ConvBlock(512 + 256, 256)

    def forward(self, x):
        a = self.e0(x)
        b = self.e1(F.avg_pool2d(a, 2, ceil_mode=True))
        c = self.e2(F.avg_pool2d(b, 2, ceil_mode=True))
        d = self.d1(torch.cat([_resize_like(c, b), b], 1))
        return self.d0(torch.cat([_resize_like(d, a), a], 1))


def _resize_like(x, reference):
    return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=False)


class NativeGridReconstructor(nn.Module):
    

    def __init__(self, input_dim=1024, merge=3, checkpointing=True):
        super().__init__()
        self.merge, self.checkpointing = merge, checkpointing
        self.input = nn.Linear(input_dim, 4096)
        self.block = CausalLlamaBlock()
        self.norm = RMSNorm(4096)
        self.aligned = nn.Linear(4096, 1536)
        self.inverse_align = nn.Sequential(nn.Linear(1536, 3072), nn.GELU(), nn.Linear(3072, 768))
                                                                                                     
        self.inverse_merger = nn.Linear(768, merge * merge * 256)
        self.unet = SpatialUNet()
        layers, channels = [], 256
        for width in (128, 64, 32, 16):
            layers += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                       nn.Conv2d(channels, width, 3, padding=1), nn.SiLU()]
            channels = width
        self.renderer = nn.Sequential(*layers, nn.Conv2d(16, 3, 3, padding=1))

    @property
    def pixels_per_token(self):
        return self.merge * PIXELS_PER_SUBPATCH

    def forward(self, x, grid, output_size=None):
        b, _, _ = x.shape
        h, w = grid
        m = self.merge
        z = self._run(self.block, self.input(x))
        features = self.inverse_align(self.aligned(self.norm(z)))
        sub = self.inverse_merger(features).view(b, h, w, m, m, 256).permute(0, 5, 1, 3, 2, 4).reshape(b, 256, h * m, w * m)
        native = torch.sigmoid(self.renderer(self._run(self.unet, sub)))
        if output_size is None:
            return native
        return F.interpolate(native.float(), size=(output_size, output_size), mode='bilinear', align_corners=False, antialias=True)

    def _run(self, module, x):
        if self.training and self.checkpointing:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)
