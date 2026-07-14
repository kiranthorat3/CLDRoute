#!/usr/bin/env python3
"""
models.py — Vanilla conditional diffusion U-Net.

Key design choices vs. original:
  1. Feature conditioning via CHANNEL CONCATENATION with noisy input
     (not add-at-first-layer). Every encoder layer sees the features directly.
  2. Self-attention at bottleneck (32×32 for 256×256 input with 3 downsamples).
  3. No metadata/text encoder — conditioning is purely spatial.
  4. Self-conditioning is always handled EXTERNALLY (caller concatenates).
     The model never does internal concatenation — eliminates train/sample mismatch.
  5. FiLM conditioning from time embedding only.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Sinusoidal time embedding (standard, from DDPM)
# ------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """
    Args:
        t: (B,) integer timesteps
        dim: embedding dimension
    Returns:
        (B, dim) sinusoidal embeddings
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ------------------------------------------------------------------
# GroupNorm helper
# ------------------------------------------------------------------

def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with adaptive group count."""
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(num_groups=max(groups, 1), num_channels=channels, eps=1e-6, affine=True)


# ------------------------------------------------------------------
# ResBlock with FiLM conditioning (time only)
# ------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # FiLM: time → (scale, shift) for norm2 output
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * out_ch))

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.film(t_emb).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.dropout(F.silu(h)))

        return h + self.skip(x)


# ------------------------------------------------------------------
# Self-Attention (for bottleneck resolution)
# ------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Standard self-attention for spatial feature maps.
    Applied at bottleneck (e.g. 32×32 = 1024 tokens — cheap).
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = _group_norm(channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)  # (B, C, HW)

        qkv = self.qkv(h)  # (B, 3C, HW)
        q, k, v = qkv.chunk(3, dim=1)  # each (B, C, HW)

        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)

        # Attention: (B, heads, HW, HW)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H * W)
        out = self.proj(out).view(B, C, H, W)

        return x + out


# ------------------------------------------------------------------
# Down/Up sampling
# ------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ------------------------------------------------------------------
# U-Net
# ------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for diffusion.

    Conditioning strategy:
      - Feature images are CONCATENATED with the noisy input along channel dim.
        The model input is [x_t, features] (and optionally [x_t, self_cond, features]).
        This means every layer sees the conditioning features spatially.
      - Time embedding is injected via FiLM (scale/shift) at every ResBlock.
      - No metadata/text encoder.

    Self-conditioning contract:
      - When use_self_cond=True, the CALLER is responsible for concatenating
        x_self_cond with x_t before passing to forward().
      - This model never does internal self-cond concatenation.
      - This ensures training and sampling use identical code paths.

    Architecture:
      - 3 downsampling stages: 256→128→64→32
      - Self-attention at 32×32 (bottleneck)
      - Skip connections via concatenation (standard U-Net)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 1,
        base: int = 64,
        t_emb_dim: int = 128,
        dropout: float = 0.0,
        num_attn_heads: int = 4,
    ):
        """
        Args:
            in_ch: Total input channels = C_label + C_feat (+ C_label if self_cond).
                   For DRC without self-cond: 1 + 9 = 10.
                   For DRC with self-cond: 1 + 1 + 9 = 11.
                   For congestion without self-cond: 1 + 3 = 4.
            out_ch: Output channels (1 for congestion/DRC).
            base: Base channel width.
            t_emb_dim: Time embedding dimension.
            dropout: Dropout rate in ResBlocks.
            num_attn_heads: Number of attention heads at bottleneck.
        """
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        # Time MLP: sinusoidal → projected
        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        # Encoder
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.enc1a = ResBlock(base, base, t_emb_dim, dropout)
        self.enc1b = ResBlock(base, base, t_emb_dim, dropout)
        self.down1 = Downsample(base)                               # 256 → 128

        self.enc2a = ResBlock(base, base * 2, t_emb_dim, dropout)
        self.enc2b = ResBlock(base * 2, base * 2, t_emb_dim, dropout)
        self.down2 = Downsample(base * 2)                           # 128 → 64

        self.enc3a = ResBlock(base * 2, base * 4, t_emb_dim, dropout)
        self.enc3b = ResBlock(base * 4, base * 4, t_emb_dim, dropout)
        self.down3 = Downsample(base * 4)                           # 64 → 32

        # Bottleneck (at 32×32 — self-attention is cheap here)
        self.mid1 = ResBlock(base * 4, base * 4, t_emb_dim, dropout)
        self.mid_attn = SelfAttention(base * 4, num_heads=num_attn_heads)
        self.mid2 = ResBlock(base * 4, base * 4, t_emb_dim, dropout)

        # Decoder (skip connections via concatenation → double channels)
        self.up3 = Upsample(base * 4)                               # 32 → 64
        self.dec3a = ResBlock(base * 8, base * 4, t_emb_dim, dropout)  # cat: 4+4=8
        self.dec3b = ResBlock(base * 4, base * 2, t_emb_dim, dropout)

        self.up2 = Upsample(base * 2)                               # 64 → 128
        self.dec2a = ResBlock(base * 4, base * 2, t_emb_dim, dropout)  # cat: 2+2=4
        self.dec2b = ResBlock(base * 2, base, t_emb_dim, dropout)

        self.up1 = Upsample(base)                                   # 128 → 256
        self.dec1a = ResBlock(base * 2, base, t_emb_dim, dropout)      # cat: 1+1=2
        self.dec1b = ResBlock(base, base, t_emb_dim, dropout)

        # Output
        self.out_norm = _group_norm(base)
        self.out_conv = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W) — already contains [x_t, features] concatenated
               (and optionally self_cond). The caller handles this.
            t_emb: (B, t_emb_dim) — sinusoidal time embedding (raw, not yet projected)

        Returns:
            (B, out_ch, H, W) — predicted noise (eps) or velocity (v)
        """
        assert x.shape[1] == self.in_ch, (
            f"Input channels mismatch: expected {self.in_ch}, got {x.shape[1]}. "
            f"Caller must concatenate [x_t, features] (and self_cond if used)."
        )

        t = self.time_mlp(t_emb)

        # Encoder
        h0 = self.in_conv(x)                    # (B, base, 256, 256)

        h1 = self.enc1a(h0, t)
        h1 = self.enc1b(h1, t)                  # skip-1: (B, base, 256, 256)
        d1 = self.down1(h1)

        h2 = self.enc2a(d1, t)
        h2 = self.enc2b(h2, t)                  # skip-2: (B, base*2, 128, 128)
        d2 = self.down2(h2)

        h3 = self.enc3a(d2, t)
        h3 = self.enc3b(h3, t)                  # skip-3: (B, base*4, 64, 64)
        d3 = self.down3(h3)

        # Bottleneck
        m = self.mid1(d3, t)                     # (B, base*4, 32, 32)
        m = self.mid_attn(m)
        m = self.mid2(m, t)

        # Decoder with skip connections
        u3 = self.up3(m)
        u3 = torch.cat([u3, h3], dim=1)         # (B, base*8, 64, 64)
        u3 = self.dec3a(u3, t)
        u3 = self.dec3b(u3, t)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, h2], dim=1)         # (B, base*4, 128, 128)
        u2 = self.dec2a(u2, t)
        u2 = self.dec2b(u2, t)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)         # (B, base*2, 256, 256)
        u1 = self.dec1a(u1, t)
        u1 = self.dec1b(u1, t)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out