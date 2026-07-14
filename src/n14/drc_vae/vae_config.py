#!/usr/bin/env python3
"""
vae_config.py — DRC VAE config for CircuitNet-N14.

Current evidence from N14 runs:
  1) beta_target=0.05, warmup=15
     - latent collapsed very early
     - never produced an LDM-ready checkpoint

  2) beta_target=0.01, warmup=40
     - latent became LDM-ready briefly around E010
     - then collapsed again by E015+

Best evidence-based next step:
  - reduce final KL pressure further
  - keep architecture and sparse-loss setup unchanged
  - change only the KL target, so the experiment stays interpretable

Why beta_target=0.0025:
  In the beta=0.01 / warmup=40 run:
    - E010 => effective beta = 0.0025  → latent alive / gate passed
    - E015 => effective beta = 0.00375 → latent collapsed
  So 0.0025 is the strongest data-backed next target.

What is intentionally unchanged:
  - latent_ch
  - base_ch
  - log_scale
  - focal_gamma
  - hotspot_weight
  - hotspot_q
  - free_bits
  - readiness gate thresholds

Reason:
  The logs most strongly implicate KL pressure, not architecture.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json
import os


@dataclass
class VAEConfig:
    task: str = "DRC"
    tech: str = "N14"

    # ── paths ─────────────────────────────────────────────────────────────────
    out_dir: str = "./runs/vae_DRC_N14_splitB_v3_beta00025_w40"

    csv_train: str = (
        "/data2/kgt22001/CircuitNet-N14/training_set_expanded/"
        "DRC/files_design/train_N14.csv"
    )
    csv_val: str = (
        "/data2/kgt22001/CircuitNet-N14/training_set_expanded/"
        "DRC/files_design/val_N14.csv"
    )
    csv_test: str = (
        "/data2/kgt22001/CircuitNet-N14/training_set_expanded/"
        "DRC/files_design/test_N14.csv"
    )
    label_dir: str = (
        "/data2/kgt22001/CircuitNet-N14/training_set_expanded/"
        "DRC/label"
    )

    # ── data ──────────────────────────────────────────────────────────────────
    C_label: int = 1
    H: int = 256
    W: int = 256

    # ── architecture ──────────────────────────────────────────────────────────
    # Kept unchanged: architecture is not yet the main suspect.
    latent_ch: int = 12
    base_ch: int = 64
    log_scale: float = 10.0

    # ── losses ────────────────────────────────────────────────────────────────
    # Kept unchanged for this KL-focused retry.
    # N14 DRC is extremely sparse, so sparse-aware loss is still justified.
    focal_gamma: float = 20.0
    hotspot_weight: float = 0.5
    hotspot_q: float = 0.999   # top 0.1%

    # ── KL ────────────────────────────────────────────────────────────────────
    # Main change:
    #   previous retry used beta_target=0.01 and briefly worked at E010
    #   but collapsed once effective beta rose further
    # So now we hold final beta at the last empirically safe level.
    free_bits: float = 0.5
    beta_target: float = 0.0025
    beta_warmup_epochs: int = 40

    # ── training ──────────────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 42
    eval_every: int = 5
    min_lr_ratio: float = 0.1

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema: bool = True
    ema_decay: float = 0.999

    # ── LDM latent readiness gate ─────────────────────────────────────────────
    # Keep strict. Do not relax thresholds to "force" a pass.
    ldm_min_ch_std: float = 0.30
    ldm_mean_ch_std: float = 0.45
    ldm_max_clamp_frac: float = 0.30

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
    def load(path: str) -> "VAEConfig":
        with open(path) as f:
            d = json.load(f)
        cfg = VAEConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg