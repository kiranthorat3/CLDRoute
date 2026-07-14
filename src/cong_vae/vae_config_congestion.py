#!/usr/bin/env python3
"""
vae_config_congestion.py — Congestion VAE config (expanded 11-channel dataset).

Design decisions justified by dataset analysis:
  - L1 reconstruction: dense map (99.49% nonzero), MSE over-smooths
  - beta=0.005: light KL — lower than previous 0.02 which caused collapse
  - free_bits=0.5: per-channel KL floor prevents channel collapse
  - no focal/hotspot: zero:nonzero=0:1, completely unjustified
  - no log transform: distribution not right-skewed (mean=0.116, std=0.058)
  - 8×64×64 latent: 4× downsampling, retains 100% signal at 64×64
  - logvar clamp (-2, 2): tighter than before for stability
  - feature channel 6 (macro_boundary_distance) is dead — dropped in LDM

Why beta=0.005 not 0.02:
  Previous run showed channels 0, 3, 7 collapsing to std≈0.01 by E020.
  KL was winning over reconstruction. beta=0.005 reduces pressure 4×.
  Free-bits ensures encoder cannot benefit from collapsing any channel.

Checkpoints:
  best_l1.pt   — lowest val L1 (reconstruction quality)
  best_ssim.pt — highest val SSIM (perceptual quality)
  best_ldm.pt  — best L1 among latent-ready epochs (LDM handoff)
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json, os


@dataclass
class CongestionVAEConfig:
    task: str = "Congestion"
    tech: str = "N28"

    # ── paths ─────────────────────────────────────────────────────────────────
    out_dir:   str = "./runs/vae_Cong_expanded_v2"
    csv_train: str = ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                      "/congestion/files_design/train_N28.csv")
    csv_val:   str = ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                      "/congestion/files_design/val_N28.csv")
    csv_test:  str = ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                      "/congestion/files_design/test_N28.csv")
    label_dir: str = ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                      "/congestion/label")

    # ── data ──────────────────────────────────────────────────────────────────
    C_label: int = 1
    H:       int = 256
    W:       int = 256

    # ── architecture ──────────────────────────────────────────────────────────
    latent_ch: int = 8
    base_ch:   int = 64

    # ── reconstruction loss ───────────────────────────────────────────────────
    recon_loss: str = "l1"

    # ── KL ────────────────────────────────────────────────────────────────────
    # beta=0.005: 4× lower than previous 0.02 — prevents channel collapse
    # free_bits=0.5: per-channel KL floor in nats
    #   encoder cannot reduce loss by collapsing a channel below 0.5 nats
    #   this keeps all 8 channels active even if they carry small signal
    # beta_warmup_epochs=40: slower warmup gives reconstruction time to
    #   establish channel usage before KL pressure ramps up
    beta_target:        float = 0.005
    beta_warmup_epochs: int   = 40
    free_bits:          float = 0.5

    # logvar clamp: tighter than before (-4,4) for training stability
    logvar_min: float = -2.0
    logvar_max: float =  2.0

    # ── training ──────────────────────────────────────────────────────────────
    epochs:       int   = 150
    batch_size:   int   = 32
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int   = 500
    grad_clip:    float = 1.0
    num_workers:  int   = 4
    seed:         int   = 42
    eval_every:   int   = 5
    min_lr_ratio: float = 0.1

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema:       bool  = True
    ema_decay: float = 0.999

    # ── LDM latent readiness gate ─────────────────────────────────────────────
    # Lower thresholds than before — with free-bits we expect moderate stds
    # not necessarily high stds like DRC VAE
    ldm_min_ch_std:      float = 0.10
    ldm_mean_ch_std:     float = 0.20
    ldm_max_clamp_frac:  float = 0.30

    def beta_at_epoch(self, epoch: int) -> float:
        if self.beta_warmup_epochs <= 0:
            return self.beta_target
        return self.beta_target * min(1.0, epoch / self.beta_warmup_epochs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "CongestionVAEConfig":
        with open(path) as f:
            d = json.load(f)
        cfg = CongestionVAEConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg