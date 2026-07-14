#!/usr/bin/env python3
"""
latent_config.py — Single source of truth for LDM hyperparameters.

N14 update:
  - supports both N28 and N14 roots
  - task and tech are read from the VAE checkpoint
  - no feature channels are dropped by default on N14
    (macro_boundary_distance is weakly correlated but not dead)

Required via CLI:
  --ae_ckpt
  --vae_dir
  --out_dir
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict
import json
import os

_ROOTS: Dict[str, str] = {
    "N28": "/data2/kgt22001/CircuitNet-N28/training_set_expanded",
    "N14": "/data2/kgt22001/CircuitNet-N14/training_set_expanded",
}


@dataclass
class LatentConfig:
    # ── Must be supplied via CLI ──────────────────────────────────────────────
    ae_ckpt: str = ""
    vae_dir: str = ""
    out_dir: str = ""

    # ── Set automatically from VAE checkpoint ────────────────────────────────
    task: str = "DRC"
    tech: str = "N14"
    latent_ch: int = 12
    H_latent: int = 64
    W_latent: int = 64

    # ── Dataset paths — filled by set_paths_for_task() ───────────────────────
    feature_dir: str = ""
    label_dir: str = ""
    csv_train: str = ""
    csv_val: str = ""
    csv_test: str = ""

    # ── Feature channel dropping ──────────────────────────────────────────────
    # Default: keep all channels.
    # For N14, I do not recommend dropping macro_boundary_distance by default.
    drop_feat_channels: List[int] = field(default_factory=list)

    # ── Feature projector ─────────────────────────────────────────────────────
    feat_proj_ch: int = 64

    # ── UNet architecture ─────────────────────────────────────────────────────
    base_channels: int = 128
    dropout: float = 0.1

    # ── Diffusion ─────────────────────────────────────────────────────────────
    diffusion_steps: int = 1000
    beta_schedule: str = "cosine"
    pred_type: str = "v"            # "v" or "eps"
    min_snr_gamma: float = 5.0

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 42

    # ── CFG ───────────────────────────────────────────────────────────────────
    # Congestion: usually 0.0
    # DRC: often >0, e.g. 0.1
    cfg_drop_prob: float = 0.0

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema: bool = True
    ema_decay: float = 0.9999

    # ── Auxiliary z0 loss ─────────────────────────────────────────────────────
    aux_weight: float = 0.1
    aux_t_cutoff: int = 200

    # ── Eval during training ──────────────────────────────────────────────────
    eval_every: int = 10
    eval_gen_batches: int = 30
    eval_gen_steps: int = 100
    eval_gen_eta: float = 0.0
    eval_cfg_scale: float = 1.5
    eval_seeds: List[int] = field(default_factory=lambda: [42, 1234, 2345])
    eval_N: int = 1

    # ── Latent clipping ───────────────────────────────────────────────────────
    # Keep encode/decode consistent with the trainer.
    z_clip: float = 3.0

    def validate(self):
        missing = [f for f in ("ae_ckpt", "vae_dir", "out_dir") if not getattr(self, f)]
        if missing:
            raise ValueError(
                f"Required config fields not set: {missing}\n"
                f"Supply via CLI: --ae_ckpt, --vae_dir, --out_dir"
            )
        if not os.path.isfile(self.ae_ckpt):
            raise FileNotFoundError(f"ae_ckpt not found: {self.ae_ckpt}")
        if not os.path.isdir(self.vae_dir):
            raise FileNotFoundError(f"vae_dir not found: {self.vae_dir}")

    def set_paths_for_task(self):
        if self.tech not in _ROOTS:
            raise ValueError(f"Unknown tech: {self.tech!r}. Known: {sorted(_ROOTS)}")
        root = _ROOTS[self.tech]

        if self.tech == "N14":
            split_tag = "N14"
        elif self.tech == "N28":
            split_tag = "N28"
        else:
            raise ValueError(f"Unsupported tech: {self.tech!r}")

        if self.task == "DRC":
            self.feature_dir = f"{root}/DRC/feature"
            self.label_dir = f"{root}/DRC/label"
            self.csv_train = f"{root}/DRC/files_design/train_{split_tag}.csv"
            self.csv_val = f"{root}/DRC/files_design/val_{split_tag}.csv"
            self.csv_test = f"{root}/DRC/files_design/test_{split_tag}.csv"

            # Keep all channels by default.
            self.drop_feat_channels = []

        elif self.task == "Congestion":
            self.feature_dir = f"{root}/congestion/feature"
            self.label_dir = f"{root}/congestion/label"
            self.csv_train = f"{root}/congestion/files_design/train_{split_tag}.csv"
            self.csv_val = f"{root}/congestion/files_design/val_{split_tag}.csv"
            self.csv_test = f"{root}/congestion/files_design/test_{split_tag}.csv"

            # Keep all channels by default.
            self.drop_feat_channels = []

        else:
            raise ValueError(f"Unknown task: {self.task!r}")

        # Verify dataset paths now.
        for p in [self.feature_dir, self.label_dir, self.csv_train, self.csv_val, self.csv_test]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Dataset path not found: {p}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "LatentConfig":
        with open(path) as f:
            d = json.load(f)
        cfg = LatentConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg