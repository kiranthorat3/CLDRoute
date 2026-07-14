#!/usr/bin/env python3
"""
models_congestion.py — U-Net and improved FeatureProjector for Congestion LDM.

What changed vs models.py:
  FeatureProjectorMultiStage replaces FeatureProjector for congestion.
  Instead of one stride-8 conv (256→32 in one shot), uses three learned
  2x downsampling stages with residual processing at each scale.

  256×256 → 128×128 → 64×64 → 32×32

  This preserves more spatial structure from the 3 conditioning channels
  before collapsing to the latent resolution.

Everything else (LatentUNet, ResBlock, SelfAttention, sinusoidal_embedding)
is identical to models.py. DRC uses models.py unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# All shared components — identical to models.py
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

def _gn(channels: int) -> nn.GroupNorm:
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(max(groups, 1), channels, eps=1e-6, affine=True)

class ResBlock(nn.Module):
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
        hd   = C // self.heads
        q    = q.view(B, self.heads, hd, H * W)
        k    = k.view(B, self.heads, hd, H * W)
        v    = v.view(B, self.heads, hd, H * W)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out  = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out  = self.proj(out.reshape(B, C, H * W)).view(B, C, H, W)
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
# NEW: Multi-stage feature projector for congestion
# ─────────────────────────────────────────────────────────────────────────────
class FeatureProjectorMultiStage(nn.Module):
    """
    Multi-stage feature projector for congestion conditioning.

    Replaces the single stride-8 conv in FeatureProjector.
    Three learned 2x downsampling stages with nonlinear processing:
      256×256 → 128×128 → 64×64 → 32×32

    Why: congestion is a dense regression map. The 3 conditioning channels
    have real spatial correlation with the label (Pearson up to 0.37).
    A single stride-8 conv loses most of this spatial structure immediately.
    Three 2x stages preserve it progressively.

    Args:
        in_ch:  input feature channels (3 for congestion)
        out_ch: output conditioning channels (feat_proj_ch, default 64)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 2   # 32 intermediate channels

        # Stage 1: 256 → 128
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            _gn(mid),
            nn.SiLU(),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1),   # 256→128
            _gn(mid),
            nn.SiLU(),
        )

        # Stage 2: 128 → 64
        self.stage2 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            _gn(mid),
            nn.SiLU(),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1),   # 128→64
            _gn(mid),
            nn.SiLU(),
        )

        # Stage 3: 64 → 32
        self.stage3 = nn.Sequential(
            nn.Conv2d(mid, out_ch, 3, padding=1),
            _gn(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),  # 64→32
            _gn(out_ch),
            nn.SiLU(),
        )

        # Final 1x1 projection
        self.final = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, in_ch, 256, 256) → (B, out_ch, 32, 32)"""
        x = self.stage1(feat)   # → (B, mid, 128, 128)
        x = self.stage2(x)      # → (B, mid, 64, 64)
        x = self.stage3(x)      # → (B, out_ch, 32, 32)
        return self.final(x)    # → (B, out_ch, 32, 32)

# ─────────────────────────────────────────────────────────────────────────────
# Keep original FeatureProjector for compatibility (used by DRC via models.py)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureProjector(nn.Module):
    """
    Original single-stride projector. Kept here for API compatibility.
    DRC uses this via models.py — do not remove.
    For congestion, use FeatureProjectorMultiStage instead.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=stride, stride=stride, padding=0),
            _gn(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)

# ─────────────────────────────────────────────────────────────────────────────
# LatentUNet — identical to models.py
# ─────────────────────────────────────────────────────────────────────────────
class LatentUNet(nn.Module):
    """
    U-Net for latent diffusion. Identical to models.py.
    Operates on latent-resolution tensors (32×32 for congestion).
    """
    def __init__(
        self,
        in_ch:     int,
        out_ch:    int,
        base:      int   = 64,
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

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc1a   = ResBlock(base,     base,     t_emb_dim, dropout)
        self.enc1b   = ResBlock(base,     base,     t_emb_dim, dropout)
        self.down1   = Downsample(base)
        self.enc2a   = ResBlock(base,     base * 2, t_emb_dim, dropout)
        self.enc2b   = ResBlock(base * 2, base * 2, t_emb_dim, dropout)
        self.down2   = Downsample(base * 2)
        self.mid1    = ResBlock(base * 2, base * 4, t_emb_dim, dropout)
        self.mid_att = SelfAttention(base * 4, num_heads=4)
        self.mid2    = ResBlock(base * 4, base * 4, t_emb_dim, dropout)
        self.up2     = Upsample(base * 4)
        self.dec2a   = ResBlock(base * 4 + base * 2, base * 2, t_emb_dim, dropout)
        self.dec2b   = ResBlock(base * 2, base * 2, t_emb_dim, dropout)
        self.up1     = Upsample(base * 2)
        self.dec1a   = ResBlock(base * 2 + base, base, t_emb_dim, dropout)
        self.dec1b   = ResBlock(base, base, t_emb_dim, dropout)
        self.out_norm = _gn(base)
        self.out_conv = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, \
            f"in_ch mismatch: expected {self.in_ch}, got {x.shape[1]}"
        t  = self.time_mlp(t_emb)
        h0 = self.in_conv(x)
        h1 = self.enc1b(self.enc1a(h0, t), t)
        h2 = self.enc2b(self.enc2a(self.down1(h1), t), t)
        m  = self.mid1(self.down2(h2), t)
        m  = self.mid_att(m)
        m  = self.mid2(m, t)
        u2 = self.up2(m)
        u2 = self.dec2b(self.dec2a(torch.cat([u2, h2], dim=1), t), t)
        u1 = self.up1(u2)
        u1 = self.dec1b(self.dec1a(torch.cat([u1, h1], dim=1), t), t)
        return self.out_conv(F.silu(self.out_norm(u1)))