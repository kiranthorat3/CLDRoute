#!/usr/bin/env python3
"""
config.py — Vanilla conditional-diffusion baseline.

Image-only conditioning (no metadata / text encoder).
All knobs documented inline.
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
import json, os


@dataclass
class TrainConfig:
    # ---- task / paths ----
    task: str = ""                  # "congestion" or "DRC"
    tech: str = ""                  # "N28" or "N14"
    out_dir: str = "./runs/default"
    csv_train: str = ""
    csv_val: str = ""
    csv_test: str = ""
    feature_dir: str = ""
    label_dir: str = ""

    # ---- training ----
    epochs: int = 120
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000       # linear LR warmup steps
    num_workers: int = 4
    seed: int = 1234
    grad_clip: float = 1.0

    # ---- diffusion ----
    diffusion_steps: int = 1000
    beta_schedule: str = "cosine"  # "cosine" or "linear"
    pred_type: str = "v"           # "v" (v-prediction) or "eps" (epsilon)
    min_snr_gamma: float = 5.0     # Min-SNR weighting (<=0 to disable)

    # ---- auxiliary x0 loss ----
    aux_weight: float = 0.2        # weight on direct x0 L1 loss (0 to disable)
    aux_t_cutoff: int = 300        # only apply aux loss for t < cutoff

    # ---- classifier-free guidance ----
    cfg_drop_prob: float = 0.1     # probability of dropping conditioning during training
    # WHY KEEP CFG:
    # - Trains model to generate both conditionally and unconditionally
    # - At sampling, cfg_scale > 1 amplifies conditioning signal
    # - For your task: "generate congestion that MORE strongly reflects physical features"
    # - Gives you a tunable inference knob to ablate in the paper
    # - Cost: ~10% of training steps see zeroed features

    # ---- self-conditioning ----
    use_self_cond: bool = False
    self_cond_prob: float = 0.5    # probability of using self-cond during training

    # ---- model ----
    base_channels: int = 64        # base channel width of U-Net
    dropout: float = 0.0

    # ---- EMA ----
    ema: bool = True
    ema_decay: float = 0.999

    # ---- evaluation ----
    eval_every: int = 10
    eval_gen_steps: int = 100      # DDIM steps for eval generation
    eval_gen_eta: float = 0.0      # DDIM eta (0 = deterministic)
    eval_cfg_scale: float = 1.5    # CFG scale for eval generation
    eval_seeds: Optional[List[int]] = None
    eval_gen_batches: int = 0      # 0 = full validation set

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return TrainConfig(**obj)