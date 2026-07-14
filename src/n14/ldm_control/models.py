#!/usr/bin/env python3
"""
models.py — ControlNet-inspired multi-scale conditioner + LatentUNet.

Architecture:
  MultiScaleConditioner
    ├── FeatureEncoder: (B, C_feat, 256, 256) → f64, f32, f16
    │     ├── Stage 0: conv + GroupNorm + ChannelAttention (SE block)
    │     ├── Stage 1: stride-4 conv → (B, base//2, 64, 64)
    │     ├── Stage 2: stride-2 conv → (B, base,    32, 32)
    │     └── Stage 3: stride-2 conv → (B, base*2,  16, 16)
    └── ZeroConvs: map encoder channels to UNet skip channels
          zero64: base//2 → base      (64×64 level)
          zero32: base    → base*2    (32×32 level)
          zero16: base*2  → base*4    (16×16 level)

  LatentUNet
    Input: noisy latent (B, latent_ch, H, W) — no feature concatenation
    Conditioning: added to skip connections before decoder
      h1 += c64   (64×64)
      h2 += c32   (32×32)
      m  += c16   (16×16, before self-attention)
    CFG null: pass c64=None, c32=None, c16=None → raw skips, no residual

Key design choices:
  - ZeroConv initialization: all weights/biases = 0
    → conditioner contributes nothing at epoch 0
    → conditioning grows gradually — stable from scratch training
  - ChannelAttention (SE): learns which of 15 input channels matter
    → justified by correlation analysis (GR_overflow r=0.28, highest)
  - FeatureEncoder uses base//2 channels at first stage
    → ~1.5M params vs ~10M for full ControlNet encoder clone
    → appropriate for 7,872-sample dataset
  - FeatureProjector kept for loading old checkpoints only
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding
# ─────────────────────────────────────────────────────────────────────────────
def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    device = t.device
    half   = dim // 2
    freqs  = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────
def _gn(channels: int) -> nn.GroupNorm:
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(max(groups, 1), channels, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    """Residual block with FiLM modulation from time embedding."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film  = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_ch))
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(t).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.norm  = _gn(channels)
        self.qkv   = nn.Conv1d(channels, 3 * channels, 1)
        self.proj  = nn.Conv1d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
        self.heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        hd  = C // self.heads
        q   = q.view(B, self.heads, hd, H * W)
        k   = k.view(B, self.heads, hd, H * W)
        v   = v.view(B, self.heads, hd, H * W)
        attn = F.softmax(
            torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        return x + self.proj(out.reshape(B, C, H * W)).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ─────────────────────────────────────────────────────────────────────────────
# Zero convolution — the ControlNet core mechanism
# ─────────────────────────────────────────────────────────────────────────────
class ZeroConv(nn.Module):
    """
    1×1 convolution initialized to exactly zero (weights and bias).

    At initialization: output = 0 for any input.
    The UNet receives zero conditioning residuals at epoch 0 — it trains
    as if unconditioned. As training proceeds, zero-conv weights learn
    non-zero values and conditioning gradually switches on.

    This is the key mechanism from ControlNet (Zhang et al., ICCV 2023)
    that makes training stable even from random initialization.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Channel attention (Squeeze-and-Excite)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excite channel attention.

    Applied to the raw feature map (B, C, 256, 256) before downsampling.
    Learns which of the C input channels are most predictive for the task.

    For DRC: dataset analysis shows GR_overflow_H (r=0.27), GR_overflow_V (r=0.29)
    are highest-correlated channels. SE block should learn to upweight these.

    reduction=4 keeps parameter count minimal (~15*4 + 4*15 = 120 params).
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).view(x.shape[0], x.shape[1], 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Feature encoder — lightweight multi-scale
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    """
    Lightweight multi-scale encoder for routing feature maps.

    Input:  (B, C_feat, 256, 256) — raw routing features after dead channel drop
    Output: three feature maps matching UNet skip connection spatial sizes
      f64: (B, base//2, 64, 64)
      f32: (B, base,    32, 32)
      f16: (B, base*2,  16, 16)

    These are then passed through ZeroConvs to match UNet channel counts
    before being added to UNet skip connections.

    Parameter count: ~1.5M for base=128.
    Full ControlNet encoder clone would be ~10M — too large for 7,872 samples.
    """
    def __init__(self, in_ch: int, base: int):
        super().__init__()
        c1 = base // 2   # 64  for base=128
        c2 = base        # 128
        c3 = base * 2    # 256

        # Initial conv + channel attention on full-resolution features
        self.in_conv   = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.in_norm   = _gn(c1)
        self.chan_attn = ChannelAttention(c1, reduction=4)

        # Stage 1: 256 → 64 (stride 4 to match latent resolution directly)
        self.stage1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=4, stride=4, padding=0),
            _gn(c1), nn.SiLU(),
            nn.Conv2d(c1, c1, 3, padding=1),
            _gn(c1), nn.SiLU(),
        )

        # Stage 2: 64 → 32
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            _gn(c2), nn.SiLU(),
            nn.Conv2d(c2, c2, 3, padding=1),
            _gn(c2), nn.SiLU(),
        )

        # Stage 3: 32 → 16
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            _gn(c3), nn.SiLU(),
            nn.Conv2d(c3, c3, 3, padding=1),
            _gn(c3), nn.SiLU(),
        )

    def forward(self, feat: torch.Tensor):
        x   = self.chan_attn(F.silu(self.in_norm(self.in_conv(feat))))
        f64 = self.stage1(x)
        f32 = self.stage2(f64)
        f16 = self.stage3(f32)
        return f64, f32, f16


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scale conditioner
# ─────────────────────────────────────────────────────────────────────────────
class MultiScaleConditioner(nn.Module):
    """
    ControlNet-inspired multi-scale feature conditioner.

    Combines FeatureEncoder with three ZeroConvs that map encoder outputs
    to exactly the channel dimensions of LatentUNet skip connections.

    ZeroConv mapping (for base=128):
      FeatureEncoder f64: base//2=64  → ZeroConv → base=128    (matches h1)
      FeatureEncoder f32: base=128    → ZeroConv → base*2=256  (matches h2)
      FeatureEncoder f16: base*2=256  → ZeroConv → base*4=512  (matches m)

    Args:
        in_ch: number of feature channels after dead channel drop
               (15 for expanded DRC, 10 for expanded Congestion)
        base:  must equal LatentUNet base — channel dimensions must match
    """
    def __init__(self, in_ch: int, base: int):
        super().__init__()
        self.encoder = FeatureEncoder(in_ch=in_ch, base=base)
        # ZeroConvs: grow conditioning signal from zero during training
        self.zero64  = ZeroConv(base // 2, base)      # 64×64
        self.zero32  = ZeroConv(base,      base * 2)  # 32×32
        self.zero16  = ZeroConv(base * 2,  base * 4)  # 16×16

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: (B, C_feat, 256, 256) — routing feature map
        Returns:
            c64: (B, base,   64, 64) — add to UNet h1 before decoder
            c32: (B, base*2, 32, 32) — add to UNet h2 before decoder
            c16: (B, base*4, 16, 16) — add to UNet m  before self-attention
        """
        f64, f32, f16 = self.encoder(feat)
        return self.zero64(f64), self.zero32(f32), self.zero16(f16)


# ─────────────────────────────────────────────────────────────────────────────
# Latent U-Net — updated for multi-scale conditioning
# ─────────────────────────────────────────────────────────────────────────────
class LatentUNet(nn.Module):
    """
    U-Net for latent diffusion with ControlNet-style conditioning.

    Input: noisy latent (B, latent_ch, H, W) — no feature concatenation.
    Conditioning is injected via residual addition at three scales:
      m  += c16  (before self-attention in bottleneck)
      h2 += c32  (into decoder skip connection at 32×32)
      h1 += c64  (into decoder skip connection at 64×64)

    CFG null conditioning:
      Pass c64=None, c32=None, c16=None.
      UNet uses raw skip connections without any conditioning residual.
      This is the true unconditional branch for CFG inference.

    Args:
        in_ch:     latent_ch (NOT latent_ch + feat_proj_ch)
        out_ch:    latent_ch
        base:      base channel width — must match MultiScaleConditioner base
        t_emb_dim: sinusoidal time embedding dimension
        dropout:   dropout rate in ResBlocks
    """
    def __init__(
        self,
        in_ch:     int,
        out_ch:    int,
        base:      int   = 128,
        t_emb_dim: int   = 128,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch

        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc1a   = ResBlock(base,     base,     t_emb_dim, dropout)
        self.enc1b   = ResBlock(base,     base,     t_emb_dim, dropout)
        self.down1   = Downsample(base)
        self.enc2a   = ResBlock(base,     base * 2, t_emb_dim, dropout)
        self.enc2b   = ResBlock(base * 2, base * 2, t_emb_dim, dropout)
        self.down2   = Downsample(base * 2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid1    = ResBlock(base * 2, base * 4, t_emb_dim, dropout)
        self.mid_att = SelfAttention(base * 4, num_heads=4)
        self.mid2    = ResBlock(base * 4, base * 4, t_emb_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Skip connections carry h1 and h2 — same channel counts as before.
        # Conditioning is added to skips before concatenation, not to the cat.
        self.up2   = Upsample(base * 4)
        self.dec2a = ResBlock(base * 4 + base * 2, base * 2, t_emb_dim, dropout)
        self.dec2b = ResBlock(base * 2,             base * 2, t_emb_dim, dropout)
        self.up1   = Upsample(base * 2)
        self.dec1a = ResBlock(base * 2 + base,      base,     t_emb_dim, dropout)
        self.dec1b = ResBlock(base,                 base,     t_emb_dim, dropout)

        self.out_norm = _gn(base)
        self.out_conv = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(
        self,
        x:     torch.Tensor,
        t_emb: torch.Tensor,
        c64:   torch.Tensor | None = None,
        c32:   torch.Tensor | None = None,
        c16:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:     (B, latent_ch, H, W)  noisy latent
            t_emb: (B, t_emb_dim)        sinusoidal time embedding
            c64:   (B, base,   64, 64)   conditioning residual or None
            c32:   (B, base*2, 32, 32)   conditioning residual or None
            c16:   (B, base*4, 16, 16)   conditioning residual or None
        Returns:
            (B, latent_ch, H, W)  predicted target (noise or velocity)
        """
        assert x.shape[1] == self.in_ch, \
            f"in_ch mismatch: expected {self.in_ch}, got {x.shape[1]}"

        t = self.time_mlp(t_emb)

        # ── Encoder ───────────────────────────────────────────────────────────
        h0 = self.in_conv(x)
        h1 = self.enc1b(self.enc1a(h0, t), t)              # (B, base,   64, 64)
        h2 = self.enc2b(self.enc2a(self.down1(h1), t), t)  # (B, base*2, 32, 32)

        # ── Bottleneck with coarse conditioning ───────────────────────────────
        m = self.mid1(self.down2(h2), t)                   # (B, base*4, 16, 16)
        if c16 is not None:
            m = m + c16      # inject coarse conditioning before attention
        m = self.mid_att(m)
        m = self.mid2(m, t)

        # ── Decoder with skip conditioning ────────────────────────────────────
        # Add conditioning to skip connections before concatenation.
        # When c* is None (CFG null branch), skips are used as-is.
        skip2 = (h2 + c32) if c32 is not None else h2
        u2    = self.up2(m)
        u2    = self.dec2b(self.dec2a(torch.cat([u2, skip2], dim=1), t), t)

        skip1 = (h1 + c64) if c64 is not None else h1
        u1    = self.up1(u2)
        u1    = self.dec1b(self.dec1a(torch.cat([u1, skip1], dim=1), t), t)

        return self.out_conv(F.silu(self.out_norm(u1)))


# ─────────────────────────────────────────────────────────────────────────────
# Legacy FeatureProjector — for loading old ldm_unified checkpoints only
# ─────────────────────────────────────────────────────────────────────────────
class FeatureProjector(nn.Module):
    """
    Single-scale feature projector from ldm_unified.
    Not used in ldm_control training.
    Kept to allow loading and evaluating old checkpoints without code changes.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=stride, stride=stride, padding=0),
            _gn(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)