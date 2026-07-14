#!/usr/bin/env python3
"""
vae_config.py — DRC VAE config for expanded 16-channel dataset.
Architecture unchanged from gen_auto/vae2 — label distribution is identical
(zero:nonzero=19.7:1, focal_gamma=20 confirmed by dataset analysis).
Only paths differ — expanded dataset in training_set_expanded/.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any
import json, os

@dataclass
class VAEConfig:
    task: str = "DRC"
    tech: str = "N28"

    # ── paths ─────────────────────────────────────────────────────────────────
    out_dir:   str = "./runs/vae_DRC_expanded"
    csv_train: str = "/data2/kgt22001/CircuitNet-N28/training_set_expanded/DRC/files_design/train_N28.csv"
    csv_val:   str = "/data2/kgt22001/CircuitNet-N28/training_set_expanded/DRC/files_design/val_N28.csv"
    csv_test:  str = "/data2/kgt22001/CircuitNet-N28/training_set_expanded/DRC/files_design/test_N28.csv"
    label_dir: str = "/data2/kgt22001/CircuitNet-N28/training_set_expanded/DRC/label"

    # ── data ──────────────────────────────────────────────────────────────────
    C_label: int = 1
    H:       int = 256
    W:       int = 256

    # ── architecture ──────────────────────────────────────────────────────────
    # Unchanged: label distribution identical, 12ch×64×64 justified as before
    latent_ch: int   = 12
    base_ch:   int   = 64
    log_scale: float = 10.0

    # ── losses ────────────────────────────────────────────────────────────────
    # focal_gamma=20 confirmed by zero:nonzero=19.7:1
    focal_gamma:    float = 20.0
    hotspot_weight: float = 0.5
    hotspot_q:      float = 0.99

    # ── KL ────────────────────────────────────────────────────────────────────
    free_bits:          float = 0.5
    beta_target:        float = 0.05
    beta_warmup_epochs: int   = 15

    # ── training ──────────────────────────────────────────────────────────────
    epochs:       int   = 150
    batch_size:   int   = 32
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int   = 500
    grad_clip:    float = 1.0
    num_workers:  int   = 4
    seed:         int   = 42
    eval_every:   int   = 5

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema:       bool  = True
    ema_decay: float = 0.999

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