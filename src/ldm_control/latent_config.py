#!/usr/bin/env python3
"""
latent_config.py — LDM config for ldm_control (ControlNet-style conditioning).

Key difference from ldm_unified:
  - feat_proj_ch is unused — conditioner channels are derived from base_channels
  - proj_type is always "multiscale_controlnet"
  - in_ch = latent_ch only (no feature concatenation)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json, os

_EXPANDED_ROOT = "/data2/kgt22001/CircuitNet-N28/training_set_expanded"


@dataclass
class LatentConfig:
    # ── Must be supplied via CLI ───────────────────────────────────────────────
    ae_ckpt: str = ""
    vae_dir: str = ""
    out_dir: str = ""

    # ── Task — set from VAE checkpoint ────────────────────────────────────────
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

    # ── Dead channel dropping ─────────────────────────────────────────────────
    drop_feat_channels: List[int] = field(default_factory=list)

    # ── Conditioner — MultiScaleConditioner uses base_channels directly ───────
    # feat_proj_ch is NOT used in ldm_control — kept for config compatibility
    feat_proj_ch: int = 64   # unused, conditioner derives channels from base

    # ── UNet architecture ─────────────────────────────────────────────────────
    base_channels: int   = 128
    dropout:       float = 0.1

    # ── Diffusion ─────────────────────────────────────────────────────────────
    diffusion_steps: int   = 1000
    beta_schedule:   str   = "cosine"
    pred_type:       str   = "v"
    min_snr_gamma:   float = 5.0

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
    cfg_drop_prob: float = 0.1

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema:       bool  = True
    ema_decay: float = 0.9999

    # ── Auxiliary z0 loss ─────────────────────────────────────────────────────
    aux_weight:   float = 0.1
    aux_t_cutoff: int   = 200

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_every:        int       = 10
    eval_gen_batches:  int       = 30
    eval_gen_steps:    int       = 100
    eval_gen_eta:      float     = 0.0
    eval_cfg_scale:    float     = 1.5
    eval_seeds:        List[int] = field(default_factory=lambda: [42, 1234, 2345])
    eval_N:            int       = 1

    def validate(self):
        missing = [f for f in ("ae_ckpt", "vae_dir", "out_dir")
                   if not getattr(self, f)]
        if missing:
            raise ValueError(f"Required fields not set: {missing}")
        if not os.path.isfile(self.ae_ckpt):
            raise FileNotFoundError(f"ae_ckpt not found: {self.ae_ckpt}")
        if not os.path.isdir(self.vae_dir):
            raise FileNotFoundError(f"vae_dir not found: {self.vae_dir}")

    def set_paths_for_task(self):
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