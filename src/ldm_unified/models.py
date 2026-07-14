#!/usr/bin/env python3
"""
models.py — U-Net for latent diffusion (updated for VAE-based LDM).

This file is shared by both DRC and Congestion latent diffusion training.

What changed vs the old version:
  1. Added FeatureProjector:
     maps full-resolution feature maps (B, C_feat, 256, 256)
     to latent-resolution conditioning maps (B, feat_proj_ch, 64, 64).

  2. LatentUNet now expects:
       [noisy_latent | projected_features]
     concatenated along channel dimension.

  3. The trainer no longer avg-pools features before calling the model.
     Downsampling is learned through FeatureProjector instead.

  4. The model operates entirely in latent space.
     It never sees pixels directly.

Input to LatentUNet:
  x = concat(
        noisy_latent      : (B, latent_ch,   H_lat, W_lat),
        projected_features: (B, feat_proj_ch, H_lat, W_lat)
      )
    → (B, latent_ch + feat_proj_ch, H_lat, W_lat)

Output:
  predicted latent target (noise or velocity), shape:
    (B, latent_ch, H_lat, W_lat)
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
# Building blocks
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

        self.film  = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 2 * out_ch),
        )
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(t).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Self-attention block for spatial feature maps."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm  = _gn(channels)
        self.qkv   = nn.Conv1d(channels, 3 * channels, 1)
        self.proj  = nn.Conv1d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
        self.heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        hd = C // self.heads
        q  = q.view(B, self.heads, hd, H * W)
        k  = k.view(B, self.heads, hd, H * W)
        v  = v.view(B, self.heads, hd, H * W)

        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = self.proj(out.reshape(B, C, H * W)).view(B, C, H, W)
        return x + out


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
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ─────────────────────────────────────────────────────────────────────────────
# Feature projector
# ─────────────────────────────────────────────────────────────────────────────
class FeatureProjector(nn.Module):
    """
    Learned projection from full-resolution feature maps to latent resolution.

    Input:
      feat: (B, C_feat, 256, 256)

    Output:
      cond: (B, out_ch, 64, 64)

    Why this exists:
      - Replaces fixed avg-pooling.
      - Lets the model learn what spatial information to preserve.
      - Works for both DRC (sparse/localized cues) and Congestion (dense/smooth cues).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=stride, stride=stride, padding=0),  # 256 -> latent_size
            _gn(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Latent U-Net
# ─────────────────────────────────────────────────────────────────────────────
class LatentUNet(nn.Module):
    """
    U-Net for latent diffusion.

    Operates on latent-resolution tensors, usually 64×64 for your VAE setup.

    Args:
        in_ch:     latent_ch + feat_proj_ch
        out_ch:    latent_ch
        base:      base feature width
        t_emb_dim: time embedding dimension
        dropout:   dropout rate
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base: int = 64,
        t_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch

        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        # Encoder: H×W -> H/2×W/2 -> H/4×W/4
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.enc1a = ResBlock(base, base, t_emb_dim, dropout)
        self.enc1b = ResBlock(base, base, t_emb_dim, dropout)
        self.down1 = Downsample(base)

        self.enc2a = ResBlock(base, base * 2, t_emb_dim, dropout)
        self.enc2b = ResBlock(base * 2, base * 2, t_emb_dim, dropout)
        self.down2 = Downsample(base * 2)

        # Bottleneck
        self.mid1    = ResBlock(base * 2, base * 4, t_emb_dim, dropout)
        self.mid_att = SelfAttention(base * 4, num_heads=4)
        self.mid2    = ResBlock(base * 4, base * 4, t_emb_dim, dropout)

        # Decoder
        self.up2   = Upsample(base * 4)
        self.dec2a = ResBlock(base * 4 + base * 2, base * 2, t_emb_dim, dropout)
        self.dec2b = ResBlock(base * 2, base * 2, t_emb_dim, dropout)

        self.up1   = Upsample(base * 2)
        self.dec1a = ResBlock(base * 2 + base, base, t_emb_dim, dropout)
        self.dec1b = ResBlock(base, base, t_emb_dim, dropout)

        self.out_norm = _gn(base)
        self.out_conv = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, in_ch, H, W)
            t_emb: (B, t_emb_dim)

        Returns:
            (B, out_ch, H, W)
        """
        assert x.shape[1] == self.in_ch, (
            f"in_ch mismatch: expected {self.in_ch}, got {x.shape[1]}"
        )

        t = self.time_mlp(t_emb)

        # Encoder
        h0 = self.in_conv(x)
        h1 = self.enc1b(self.enc1a(h0, t), t)                 # (B, base,   H,   W)
        h2 = self.enc2b(self.enc2a(self.down1(h1), t), t)    # (B, 2base, H/2, W/2)

        # Bottleneck
        m  = self.mid1(self.down2(h2), t)                    # (B, 4base, H/4, W/4)
        m  = self.mid_att(m)
        m  = self.mid2(m, t)

        # Decoder
        u2 = self.up2(m)
        u2 = self.dec2b(self.dec2a(torch.cat([u2, h2], dim=1), t), t)

        u1 = self.up1(u2)
        u1 = self.dec1b(self.dec1a(torch.cat([u1, h1], dim=1), t), t)

        return self.out_conv(F.silu(self.out_norm(u1)))