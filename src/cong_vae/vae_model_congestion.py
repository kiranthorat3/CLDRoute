#!/usr/bin/env python3
"""
vae_model_congestion.py — VAE for dense congestion maps (expanded dataset).

Key differences from DRC VAE:
  - No log-space transform: congestion not right-skewed
  - No focal/hotspot loss: 99.49% nonzero, no sparsity
  - L1 reconstruction: better for dense regression maps
  - Tighter logvar clamp (-2, 2): stability without sacrificing flexibility
  - Sigmoid output: maps to (0,1) directly
  - beta=0.005 with free_bits=0.5: prevents channel collapse

Free-bits KL:
  Standard KL averages over all channels together. The optimizer can
  reduce total KL by collapsing low-information channels to zero variance.
  Free-bits guarantees each channel a minimum KL budget (0.5 nats) before
  penalty applies — the encoder cannot profit from collapsing any channel.

model_type = 'CongestionVAE_v2'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOGVAR_MIN = -2.0
_LOGVAR_MAX =  2.0


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
        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

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
def recon_loss_l1(recon: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return (recon - label).abs().mean(dim=(1, 2, 3)).mean()


def kl_loss(
    mu:         torch.Tensor,
    logvar:     torch.Tensor,
    logvar_min: float = _LOGVAR_MIN,
    logvar_max: float = _LOGVAR_MAX,
    free_bits:  float = 0.5,
) -> torch.Tensor:
    """
    KL divergence with free-bits per channel.

    Standard KL averaged over everything allows channel collapse —
    the optimizer zeros out low-information channels to reduce KL.

    Free-bits fix: compute KL per channel (averaged over batch and
    spatial dims), clamp each channel to minimum free_bits nats,
    then average over channels. The encoder cannot reduce its loss
    by collapsing a channel below free_bits — there is no incentive
    to zero it out.

    Args:
        mu:         (B, latent_ch, H, W)
        logvar:     (B, latent_ch, H, W)
        logvar_min: lower clamp for logvar
        logvar_max: upper clamp for logvar
        free_bits:  minimum KL per channel in nats (0.5 is standard)
    """
    logvar_c  = logvar.clamp(logvar_min, logvar_max)
    # Per-element KL: (B, latent_ch, H, W)
    kl        = -0.5 * (1.0 + logvar_c - mu.pow(2) - logvar_c.exp())
    # Average over batch and spatial dims → (latent_ch,)
    kl_per_ch = kl.mean(dim=(0, 2, 3))
    # Free-bits floor: encoder cannot profit from collapsing any channel
    kl_per_ch = torch.clamp(kl_per_ch, min=free_bits)
    # Average over channels
    return kl_per_ch.mean()


# ─────────────────────────────────────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────────────────────────────────────
class CongestionVAE(nn.Module):
    """
    Symmetric VAE for dense congestion maps.

    Encoder: 256 → 128 → 64 with ResBlocks
    Decoder: 64  → 128 → 256 with ResBlocks
    Latent:  8 channels at 64×64
    Output:  sigmoid → (0,1)

    encode_to_z / decode_from_z interface matches LDM trainer expectations.
    encode_to_z returns mu (no noise) — deterministic for LDM training.
    """
    def __init__(
        self,
        C_label:    int   = 1,
        latent_ch:  int   = 8,
        base_ch:    int   = 64,
        logvar_min: float = -2.0,
        logvar_max: float =  2.0,
    ):
        super().__init__()
        self.C_label    = C_label
        self.latent_ch  = latent_ch
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_in    = nn.Conv2d(C_label,    base_ch,     3, padding=1)
        self.enc1      = ResBlock(base_ch,     base_ch)
        self.down1     = Downsample(base_ch)                   # 256→128
        self.enc2      = ResBlock(base_ch,     base_ch * 2)
        self.down2     = Downsample(base_ch * 2)               # 128→64
        self.enc3      = ResBlock(base_ch * 2, base_ch * 4)
        self.to_mu     = nn.Conv2d(base_ch * 4, latent_ch, 1)
        self.to_logvar = nn.Conv2d(base_ch * 4, latent_ch, 1)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec_in  = nn.Conv2d(latent_ch,    base_ch * 4, 1)
        self.dec3    = ResBlock(base_ch * 4,   base_ch * 2)
        self.up2     = Upsample(base_ch * 2)                   # 64→128
        self.dec2    = ResBlock(base_ch * 2,   base_ch)
        self.up1     = Upsample(base_ch)                       # 128→256
        self.dec1    = ResBlock(base_ch,       base_ch)
        self.dec_out = nn.Sequential(
            _groupnorm(base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, C_label, 3, padding=1),
        )

    def encode(self, label: torch.Tensor):
        x      = self.enc_in(label)
        x      = self.enc1(x)
        x      = self.enc2(self.down1(x))
        x      = self.enc3(self.down2(x))
        mu     = self.to_mu(x)
        logvar = self.to_logvar(x).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(self.logvar_min, self.logvar_max))
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_in(z)
        x = self.dec3(x)
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return torch.sigmoid(self.dec_out(x))

    def forward(self, label: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(label)
        z          = self.reparameterize(mu, logvar) if sample else mu
        recon      = self.decode(z)
        return recon, mu, logvar, z

    @torch.no_grad()
    def encode_to_z(self, label: torch.Tensor) -> torch.Tensor:
        """Returns mu — no noise. Used by LDM trainer."""
        mu, _ = self.encode(label)
        return mu

    @torch.no_grad()
    def decode_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Used by LDM sampler."""
        return self.decode(z)