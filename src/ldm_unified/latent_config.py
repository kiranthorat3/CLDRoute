#!/usr/bin/env python3
"""
latent_config.py — Single source of truth for LDM hyperparameters.

Paths (ae_ckpt, vae_dir, out_dir) have no defaults — must be supplied
via CLI. Everything else has a sensible default that CLI can override.

Dead channel policy (confirmed by dataset analysis):
  DRC:        drop channel 13 (macro_boundary_distance) → 15 active channels
  Congestion: drop channel 6  (macro_boundary_distance) → 10 active channels
  Set automatically by set_paths_for_task() after task is read from checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json, os

_EXPANDED_ROOT = "/data2/kgt22001/CircuitNet-N28/training_set_expanded"


@dataclass
class LatentConfig:
    # ── Must be supplied via CLI — no defaults ────────────────────────────────
    ae_ckpt: str = ""   # path to VAE/AE checkpoint (best_ldm.pt etc.)
    vae_dir: str = ""   # directory containing VAE model Python file
    out_dir: str = ""   # output directory for this LDM run

    # ── Task — set automatically from VAE checkpoint, do not set manually ─────
    task:      str = "DRC"
    latent_ch: int = 12
    H_latent:  int = 64
    W_latent:  int = 64

    # ── Dataset paths — set by set_paths_for_task() ───────────────────────────
    feature_dir: str = ""
    label_dir:   str = ""
    csv_train:   str = ""
    csv_val:     str = ""
    csv_test:    str = ""

    # ── Dead channel dropping — set by set_paths_for_task() ──────────────────
    drop_feat_channels: List[int] = field(default_factory=list)

    # ── Feature projector ─────────────────────────────────────────────────────
    feat_proj_ch: int = 64

    # ── UNet architecture — changing these makes checkpoints incompatible ─────
    base_channels: int   = 128
    dropout:       float = 0.1

    # ── Diffusion — stored in checkpoint, sampler reads from there ────────────
    diffusion_steps: int   = 1000
    beta_schedule:   str   = "cosine"
    pred_type:       str   = "v"       # "v" or "eps"
    min_snr_gamma:   float = 5.0       # 0 to disable

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:       int   = 200
    batch_size:   int   = 16
    lr:           float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int   = 1000
    grad_clip:    float = 1.0
    num_workers:  int   = 4
    seed:         int   = 42

    # ── CFG ───────────────────────────────────────────────────────────────────
    # DRC:        0.1  (CFG helps sparse maps)
    # Congestion: 0.0  (no CFG — hurts dense maps empirically)
    cfg_drop_prob: float = 0.1

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema:       bool  = True
    ema_decay: float = 0.9999

    # ── Auxiliary z0 loss ─────────────────────────────────────────────────────
    aux_weight:   float = 0.1
    aux_t_cutoff: int   = 200

    # ── Evaluation during training ────────────────────────────────────────────
    # eval_gen_steps should match the steps you plan to use in final sampling.
    # eval_seeds: averaged before checkpoint selection — prevents single-seed noise
    # eval_N: keep low (1) to control training eval cost
    eval_every:       int       = 10
    eval_gen_batches: int       = 30    # 0 = full val set
    eval_gen_steps:   int       = 100   # matches sampler default
    eval_gen_eta:     float     = 0.0
    eval_cfg_scale:   float     = 1.5
    eval_seeds:       List[int] = field(default_factory=lambda: [42, 1234, 2345])
    eval_N:           int       = 1     # draws per sample during training eval

    def validate(self):
        """Call after CLI args are applied. Raises if required fields missing."""
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
        """
        Set dataset paths and dead channel list from self.task.
        Called automatically after task is read from the VAE checkpoint.
        """
        root = _EXPANDED_ROOT
        if self.task == "DRC":
            self.feature_dir        = f"{root}/DRC/feature"
            self.label_dir          = f"{root}/DRC/label"
            self.csv_train          = f"{root}/DRC/files_design/train_N28.csv"
            self.csv_val            = f"{root}/DRC/files_design/val_N28.csv"
            self.csv_test           = f"{root}/DRC/files_design/test_N28.csv"
            self.drop_feat_channels = [13]
        elif self.task == "Congestion":
            self.feature_dir        = f"{root}/congestion/feature"
            self.label_dir          = f"{root}/congestion/label"
            self.csv_train          = f"{root}/congestion/files_design/train_N28.csv"
            self.csv_val            = f"{root}/congestion/files_design/val_N28.csv"
            self.csv_test           = f"{root}/congestion/files_design/test_N28.csv"
            self.drop_feat_channels = [6]
        else:
            raise ValueError(f"Unknown task: {self.task!r}")

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