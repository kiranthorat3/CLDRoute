#!/usr/bin/env python3
"""
vae_model.py — Unconditional VAE for DRC label maps (expanded dataset).

Changes from gen_auto/vae2/vae_model.py:
  - logvar clamp lower bound raised from -2.0 to -1.0
    At -2.0: min std = exp(-1.0) = 0.368 — clamp was permanently active
    At -1.0: min std = exp(-0.5) = 0.607 — closer to unit variance
  - hotspot_loss vectorized — no Python loop over batch
  - free_bits_kl unchanged
  - Architecture unchanged
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOGVAR_MIN = -1.0   # raised from -2.0
_LOGVAR_MAX =  4.0

def log_encode(x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    return torch.log1p(x * scale)

def log_decode(x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    return (torch.expm1(x) / scale).clamp(0.0, 1.0)

def _groupnorm(channels: int) -> nn.GroupNorm:
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(max(groups, 1), channels, eps=1e-6, affine=True)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _groupnorm(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _groupnorm(out_ch), nn.SiLU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))

# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────
def focal_recon_loss(
    recon: torch.Tensor,
    label: torch.Tensor,
    gamma: float = 20.0,
) -> torch.Tensor:
    weight   = 1.0 + gamma * torch.sqrt(label.detach().clamp(0.0, 1.0))
    loss_map = weight * (recon - label).pow(2)
    return loss_map.mean(dim=(1, 2, 3)).mean()

def hotspot_loss(
    recon: torch.Tensor,
    label: torch.Tensor,
    q:     float = 0.99,
) -> torch.Tensor:
    """
    MSE on top-(1-q)% label pixels per sample.
    Vectorized — no Python loop over batch.
    Works by sorting the flattened label and masking top-k positions.
    """
    B, C, H, W = label.shape
    N = H * W
    k = max(1, int((1.0 - q) * N))

    lbl_flat = label[:, 0].reshape(B, N)   # (B, N)
    rec_flat = recon[:, 0].reshape(B, N)   # (B, N)

    # Top-k indices per sample
    _, topk_idx = torch.topk(lbl_flat, k, dim=1, sorted=False)  # (B, k)

    # Gather values at top-k positions
    lbl_hot = lbl_flat.gather(1, topk_idx)  # (B, k)
    rec_hot = rec_flat.gather(1, topk_idx)  # (B, k)

    return F.mse_loss(rec_hot, lbl_hot)

def free_bits_kl(
    mu:        torch.Tensor,
    logvar:    torch.Tensor,
    free_bits: float = 0.5,
) -> torch.Tensor:
    logvar_c  = logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX)
    kl_per_ch = -0.5 * (1.0 + logvar_c - mu.pow(2) - logvar_c.exp())
    kl_per_ch = kl_per_ch.mean(dim=(0, 2, 3))
    return torch.clamp(kl_per_ch, min=free_bits).mean()

# ─────────────────────────────────────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────────────────────────────────────
class LabelVAE(nn.Module):
    """
    Unconditional VAE for DRC label maps.
    Encoder: 256→128→64 with ResBlocks.
    Decoder: 64→128→256, no skip connections.
    Latent:  12 channels at 64×64.
    """
    def __init__(
        self,
        C_label:   int   = 1,
        latent_ch: int   = 12,
        base_ch:   int   = 64,
        log_scale: float = 10.0,
    ):
        super().__init__()
        self.C_label   = C_label
        self.latent_ch = latent_ch
        self.log_scale = log_scale
        self._log_max  = math.log(1.0 + log_scale)

        # Encoder
        self.enc_in    = nn.Conv2d(C_label,    base_ch,     3, padding=1)
        self.enc1      = ResBlock(base_ch,     base_ch)
        self.down1     = Downsample(base_ch)
        self.enc2      = ResBlock(base_ch,     base_ch * 2)
        self.down2     = Downsample(base_ch * 2)
        self.enc3      = ResBlock(base_ch * 2, base_ch * 4)
        self.to_mu     = nn.Conv2d(base_ch * 4, latent_ch, 1)
        self.to_logvar = nn.Conv2d(base_ch * 4, latent_ch, 1)

        # Decoder
        self.dec_in  = nn.Conv2d(latent_ch,    base_ch * 4, 1)
        self.dec3    = ResBlock(base_ch * 4,   base_ch * 2)
        self.up2     = Upsample(base_ch * 2)
        self.dec2    = ResBlock(base_ch * 2,   base_ch)
        self.up1     = Upsample(base_ch)
        self.dec1    = ResBlock(base_ch,       base_ch)
        self.dec_out = nn.Sequential(
            _groupnorm(base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, C_label, 3, padding=1),
        )

    def encode(self, label: torch.Tensor):
        x      = log_encode(label, self.log_scale)
        x      = self.enc_in(x)
        x      = self.enc1(x)
        x      = self.enc2(self.down1(x))
        x      = self.enc3(self.down2(x))
        mu     = self.to_mu(x)
        logvar = self.to_logvar(x).clamp(_LOGVAR_MIN, _LOGVAR_MAX)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(_LOGVAR_MIN, _LOGVAR_MAX))
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_in(z)
        x = self.dec3(x)
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        x = self.dec_out(x)
        x = self._log_max * torch.sigmoid(x)
        return log_decode(x, self.log_scale)

    def forward(self, label: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(label)
        z          = self.reparameterize(mu, logvar) if sample else mu
        recon      = self.decode(z)
        return recon, mu, logvar, z

    @torch.no_grad()
    def encode_to_z(self, label: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(label)
        return mu

    @torch.no_grad()
    def decode_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)