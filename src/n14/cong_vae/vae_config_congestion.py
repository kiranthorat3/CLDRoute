#!/usr/bin/env python3
"""
vae_config_congestion.py — Congestion VAE config for CircuitNet-N14.

This is a direct N14 baseline, kept intentionally close to N28:
  - label-only VAE
  - L1 reconstruction
  - 64×64 latent map
  - beta-VAE style KL with free-bits
  - same model/loss family as N28 for fair comparison

What is supported by current N14 analysis:
  - congestion labels are dense (~99.28% nonzero), so focal/hotspot loss is not justified
  - label maps are smooth/dense, so L1 is a sensible default
  - 64×64 latent resolution is safer than 32×32 for the first baseline

What is NOT assumed as a fact:
  - beta=0.005 is the final optimal value for N14
  - free-bits guarantees no collapse
  - latent readiness thresholds are correctness tests
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json
import os


@dataclass
class CongestionVAEConfig:
    task: str = "Congestion"
    tech: str = "N14"

    # ── paths ─────────────────────────────────────────────────────────────────
    out_dir: str = "./runs/vae_Cong_N14_splitB_v1"

    csv_train: str = (
        "/data2/kgt22001/CircuitNet-N14/training_set_expanded/"
        "congestion/files_design/train_N14.csv"
    )
    csv_val: str = (
        "/data2/kgt22001/CircuitNet-N14/training_set_expanded/"
        "congestion/files_design/val_N14.csv"
    )
    csv_test: str = (
        "/data2/kgt22001/CircuitNet-N14/training_set_expanded/"
        "congestion/files_design/test_N14.csv"
    )
    label_dir: str = (
        "/data2/kgt22001/CircuitNet-N14/training_set_expanded/"
        "congestion/label"
    )

    # ── data ──────────────────────────────────────────────────────────────────
    C_label: int = 1
    H: int = 256
    W: int = 256

    # ── architecture ──────────────────────────────────────────────────────────
    # First N14 baseline: keep same bottleneck shape as N28 baseline.
    latent_ch: int = 8
    base_ch: int = 64

    # ── reconstruction loss ───────────────────────────────────────────────────
    recon_loss: str = "l1"

    # ── KL / regularization ───────────────────────────────────────────────────
    # beta_target=0.005 is inherited as the starting baseline from N28.
    # Keep it for the first N14 run, but validate latent usage during training.
    beta_target: float = 0.005
    beta_warmup_epochs: int = 40
    free_bits: float = 0.5

    # Logvar clamp kept same as N28 congestion baseline.
    logvar_min: float = -2.0
    logvar_max: float = 2.0

    # ── training ──────────────────────────────────────────────────────────────
    epochs: int = 150
    batch_size: int = 32
    lr: float = 3e-4
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
    # These are practical heuristics, not guarantees.
    ldm_min_ch_std: float = 0.10
    ldm_mean_ch_std: float = 0.20
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
    def load(path: str) -> "CongestionVAEConfig":
        with open(path) as f:
            d = json.load(f)
        cfg = CongestionVAEConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg