#!/usr/bin/env python3
"""
latent_trainer.py — LDM training for DRC and Congestion (expanded dataset).

Checkpoint policy:
  best_val.pt — lowest denoising val loss (EMA weights)
  best_gen.pt — best generation metric averaged over eval_seeds (EMA weights)
  latest.pt   — most recent epoch (raw training weights — use for resuming)

Note: best_val.pt and best_gen.pt store EMA weights.
      latest.pt stores raw weights + full optimizer state.
      Resume always from latest.pt, not best_*.pt.

Protocol alignment:
  Training eval uses the same step count (eval_gen_steps=100) and seeding
  strategy (per-sample stable seeds) as the final sampler, so best_gen.pt
  is selected under the same protocol you will report.

Usage:
  # DRC
  python latent_trainer.py \
    --ae_ckpt /data2/.../drc_vae/runs/vae_DRC_expanded/best_ldm.pt \
    --vae_dir /data2/.../drc_vae \
    --out_dir  ./runs/ldm_DRC_expanded \
    --cfg_drop_prob 0.1 \
    --epochs 200

  # Congestion
  python latent_trainer.py \
    --ae_ckpt /data2/.../congestion_ae/runs/ae_Cong_expanded/best_ldm.pt \
    --vae_dir /data2/.../congestion_ae \
    --out_dir  ./runs/ldm_Cong_expanded \
    --cfg_drop_prob 0.0 \
    --epochs 200

  # Resume
  python latent_trainer.py \
    --ae_ckpt ... --vae_dir ... --out_dir ... \
    --resume ./runs/ldm_DRC_expanded/latest.pt
"""
from __future__ import annotations
import os
import json
import time
import math
import random
import hashlib
import argparse
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from latent_config import LatentConfig
from latent_data   import LatentDataset, make_latent_loaders, _collate
from diffusion     import build_betas
from models        import LatentUNet, FeatureProjector, sinusoidal_embedding
from models_congestion import FeatureProjectorMultiStage
from utils_ema     import EMA
from utils_log     import AvgMeter, CSVLogger, TBLogger, gpu_mem_gb


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# Module import — sanitized name avoids path characters in module identifiers
# ─────────────────────────────────────────────────────────────────────────────
def _import_from_file(file_path: str):
    h    = hashlib.sha1(file_path.encode()).hexdigest()[:8]
    name = f"_ldm_vae_{h}"
    spec   = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# VAE / AE loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_vae(ckpt_path: str, vae_dir: str, device: torch.device):
    ck         = torch.load(ckpt_path, map_location=device)
    model_type = ck.get("model_type", "unknown")

    # Verify vae_dir matches what the checkpoint was trained with, if stored
    stored_vae_dir = ck.get("vae_dir", None)
    if stored_vae_dir and os.path.realpath(stored_vae_dir) != os.path.realpath(vae_dir):
        print(f"[VAE] WARNING: supplied vae_dir differs from checkpoint's stored vae_dir")
        print(f"  supplied:  {vae_dir}")
        print(f"  stored:    {stored_vae_dir}")
        print(f"  Proceeding with supplied vae_dir — verify this is intentional.")

    if model_type == "LabelVAE_v2":
        mod   = _import_from_file(os.path.join(vae_dir, "vae_model.py"))
        model = mod.LabelVAE(
            C_label   = int(ck["C_label"]),
            latent_ch = int(ck["latent_ch"]),
            base_ch   = int(ck.get("base_ch", 64)),
            log_scale = float(ck.get("log_scale", 10.0)),
        ).to(device)

    elif model_type in ("CongestionVAE_v2", "CongestionVAE"):
        mod   = _import_from_file(os.path.join(vae_dir, "vae_model_congestion.py"))
        model = mod.CongestionVAE(
            C_label    = int(ck["C_label"]),
            latent_ch  = int(ck["latent_ch"]),
            base_ch    = int(ck.get("base_ch", 64)),
            logvar_min = float(ck.get("logvar_min", -4.0)),
            logvar_max = float(ck.get("logvar_max",  4.0)),
        ).to(device)

    #elif model_type == "CongestionAE":
    elif model_type in ("CongestionAE_v2", "CongestionAE"):
        mod   = _import_from_file(os.path.join(vae_dir, "ae_model_congestion.py"))
        model = mod.CongestionAE(
            C_label   = int(ck["C_label"]),
            latent_ch = int(ck["latent_ch"]),
            base_ch   = int(ck.get("base_ch", 64)),
        ).to(device)

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}' in {ckpt_path}.\n"
            f"Supported: LabelVAE_v2, CongestionVAE_v2, CongestionVAE, CongestionAE."
        )

    model.load_state_dict(ck["net"], strict=True)
    if "ema" in ck:
        _e = EMA(model)
        _e.load_state_dict(ck["ema"])
        _e.copy_to(model)
        print(f"  VAE EMA weights applied")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ck


# ─────────────────────────────────────────────────────────────────────────────
# Latent stats
# ─────────────────────────────────────────────────────────────────────────────
def _load_latent_stats(ckpt_path: str, latent_ch: int, device: torch.device):
    stats_path = os.path.join(
        os.path.dirname(os.path.abspath(ckpt_path)), "latent_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"latent_stats.json not found at {stats_path}\n"
            f"Run vae_latent_stats.py --ckpt {ckpt_path} first."
        )
    with open(stats_path) as f:
        stats = json.load(f)
    z_mean_l = stats["z_mean"]
    z_std_l  = stats["z_std"]
    assert len(z_mean_l) == latent_ch, \
        f"latent_stats has {len(z_mean_l)} channels, VAE has {latent_ch}"
    if min(z_std_l) < 0.05:
        raise ValueError(
            f"Latent std too small (min={min(z_std_l):.4f}). "
            f"Retrain VAE with stronger KL."
        )
    z_mean = torch.tensor(z_mean_l, dtype=torch.float32, device=device).view(1, latent_ch, 1, 1)
    z_std  = torch.tensor(z_std_l,  dtype=torch.float32, device=device).view(1, latent_ch, 1, 1)
    return z_mean, z_std, z_mean_l, z_std_l


# ─────────────────────────────────────────────────────────────────────────────
# Stable per-sample seeding — matches latent_sampler.py exactly
# ─────────────────────────────────────────────────────────────────────────────
def _stable_seed(name: str) -> int:
    return int.from_bytes(
        hashlib.sha1(name.encode()).digest()[:4],
        byteorder="little", signed=False)

def _make_noise(B, C, H, W, device, base_seed, names, draw_k):
    x = torch.empty((B, C, H, W), device=device)
    for i, nm in enumerate(names):
        s = int(base_seed) ^ _stable_seed(str(nm)) ^ (int(draw_k) * 0x9E3779B1)
        g = torch.Generator(device=device)
        g.manual_seed(s & 0x7FFFFFFF)
        x[i:i+1] = torch.randn((1, C, H, W), device=device, generator=g)
    return x

def _make_ts(T: int, steps: int, device):
    ts = np.rint(np.linspace(T-1, 0, steps, dtype=np.float64)).astype(np.int64)
    for i in range(1, len(ts)):
        if ts[i] >= ts[i-1]:
            ts[i] = ts[i-1] - 1
    return torch.tensor(np.clip(ts, 0, T-1), device=device, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Generation metrics
# ─────────────────────────────────────────────────────────────────────────────
_NZ_THRESH = 0.01

def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _metrics_drc(gt: np.ndarray, pr: np.ndarray) -> dict:
    k      = max(1, int(0.01 * gt.size))
    gt_idx = np.argsort(gt.ravel())[-k:]
    pr_idx = np.argsort(pr.ravel())[-k:]
    topk   = float(len(np.intersect1d(gt_idx, pr_idx)) / k)
    nz     = gt.ravel() > _NZ_THRESH
    nz_mae = float(np.mean(np.abs(gt.ravel()[nz] - pr.ravel()[nz]))) \
             if nz.any() else float("nan")
    mae    = float(np.mean(np.abs(gt - pr)))
    ssim   = float(structural_similarity(gt, pr, data_range=1.0))
    return dict(topk_1pct=topk, nz_mae=nz_mae, mae=mae, ssim=ssim)

def _metrics_congestion(gt: np.ndarray, pr: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(gt - pr)))
    ssim = float(structural_similarity(gt, pr, data_range=1.0))
    pear = _safe_pearson(gt, pr)
    return dict(mae=mae, ssim=ssim, pearson=pear)

def _compute_metrics(gt, pr, task):
    return _metrics_drc(gt, pr) if task == "DRC" else _metrics_congestion(gt, pr)

def _trivial_metrics(gt, task, train_mean):
    pr = np.zeros_like(gt) if task == "DRC" else np.full_like(gt, train_mean)
    return _compute_metrics(gt, pr, task)

def _gen_score(m: dict, task: str) -> float:
    return m["topk_1pct"] if task == "DRC" else -m["mae"]


# ─────────────────────────────────────────────────────────────────────────────
# EMA context manager
# ─────────────────────────────────────────────────────────────────────────────
class _EMASwap:
    """Temporarily swap EMA weights into model for evaluation."""
    def __init__(self, ema, module):
        self.ema     = ema
        self.module  = module
        self._stored = None

    def __enter__(self):
        if self.ema is not None:
            self._stored = {k: v.clone() for k, v in self.module.state_dict().items()}
            self.ema.copy_to(self.module)
        return self

    def __exit__(self, *_):
        if self.ema is not None and self._stored is not None:
            self.module.load_state_dict(self._stored)
            self._stored = None


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class LatentTrainer:
    def __init__(self, cfg: LatentConfig, resume_ckpt: str | None = None):
        self.cfg = cfg
        cfg.validate()
        os.makedirs(cfg.out_dir, exist_ok=True)
        seed_everything(cfg.seed)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_type = cfg.pred_type.lower()
        assert self.pred_type in ("v", "eps"), f"pred_type must be v or eps"

        # ── Frozen VAE ────────────────────────────────────────────────────────
        print("[Init] Loading VAE/AE checkpoint...")
        self.ae, ae_ck = _load_vae(cfg.ae_ckpt, cfg.vae_dir, self.device)
        self.vae_type  = ae_ck["model_type"]
        cfg.task       = ae_ck.get("task", cfg.task)
        cfg.latent_ch  = int(ae_ck["latent_ch"])
        self.task      = cfg.task
        cfg.set_paths_for_task()
        print(f"  type={self.vae_type} task={self.task} "
              f"epoch={ae_ck['epoch']} latent={cfg.latent_ch}ch")

        # ── Latent normalization ───────────────────────────────────────────────
        self.z_mean, self.z_std, z_mean_l, z_std_l = _load_latent_stats(
            cfg.ae_ckpt, cfg.latent_ch, self.device)
        print(f"  z_std range: [{min(z_std_l):.3f}, {max(z_std_l):.3f}]")

        # ── Data ──────────────────────────────────────────────────────────────
        self.ds_train, self.ds_val, self.loader_train, self.loader_val = \
            make_latent_loaders(cfg)
        self.C_feat = self.ds_train.C_feat

        # ── Infer latent spatial size ──────────────────────────────────────────
        with torch.no_grad():
            _f, _l, _ = next(iter(self.loader_train))
            _z = self.ae.encode_to_z(_l[:1].float().to(self.device))
            cfg.H_latent = _z.shape[2]
            cfg.W_latent = _z.shape[3]

        # ── Congestion training mean for trivial baseline ──────────────────────
        # Use a separate loader with drop_last=False to get exact dataset mean
        if self.task == "Congestion":
            from latent_data import LatentDataset
            ds_full = LatentDataset(
                csv_path      = cfg.csv_train,
                feature_dir   = cfg.feature_dir,
                label_dir     = cfg.label_dir,
                drop_channels = cfg.drop_feat_channels,
                split         = "train",
                verify        = False,
            )
            ldr_full = DataLoader(
                ds_full, batch_size=64, shuffle=False,
                num_workers=cfg.num_workers, collate_fn=_collate,
                drop_last=False,
            )
            s, n = 0.0, 0
            for _, lb, _ in ldr_full:
                s += float(lb.sum()); n += lb.numel()
            self.train_mean = s / max(n, 1)
            print(f"  Congestion train label mean={self.train_mean:.4f} "
                  f"(computed over full {len(ds_full)} samples)")
        else:
            self.train_mean = 0.0


        # OLD — branches on task name, wrong for new 64×64 congestion AE
        if self.task == "Congestion":
            self.feat_proj = FeatureProjectorMultiStage(
                in_ch=self.C_feat, out_ch=cfg.feat_proj_ch).to(self.device)
            self.proj_type = "multistage"
        else:
            stride         = 256 // cfg.H_latent
            self.feat_proj = FeatureProjector(
                in_ch=self.C_feat, out_ch=cfg.feat_proj_ch,
                stride=stride).to(self.device)
            self.proj_type = "single"

        # NEW — branches on actual latent resolution, always correct
        stride         = 256 // cfg.H_latent   # 64×64 → stride=4, 32×32 → stride=8
        self.feat_proj = FeatureProjector(
            in_ch=self.C_feat, out_ch=cfg.feat_proj_ch,
            stride=stride).to(self.device)
        self.proj_type = "single"

        # ── Feature projector ─────────────────────────────────────────────────
        # if self.task == "Congestion":
        #     self.feat_proj = FeatureProjectorMultiStage(
        #         in_ch=self.C_feat, out_ch=cfg.feat_proj_ch).to(self.device)
        #     self.proj_type = "multistage"
        # else:
        #     stride         = 256 // cfg.H_latent
        #     self.feat_proj = FeatureProjector(
        #         in_ch=self.C_feat, out_ch=cfg.feat_proj_ch,
        #         stride=stride).to(self.device)
        #     self.proj_type = "single"

        # ── UNet ──────────────────────────────────────────────────────────────
        self.in_ch = cfg.latent_ch + cfg.feat_proj_ch
        self.net   = LatentUNet(
            in_ch=self.in_ch, out_ch=cfg.latent_ch,
            base=cfg.base_channels, t_emb_dim=128,
            dropout=cfg.dropout).to(self.device)

        n_unet = sum(p.numel() for p in self.net.parameters())
        n_proj = sum(p.numel() for p in self.feat_proj.parameters())

        # ── Optimizer + EMA ───────────────────────────────────────────────────
        params = list(self.net.parameters()) + list(self.feat_proj.parameters())
        self.optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        class _Combined(torch.nn.Module):
            def __init__(self, net, proj):
                super().__init__()
                self.net = net; self.proj = proj

        self._combined = _Combined(self.net, self.feat_proj)
        self.ema = EMA(self._combined, decay=cfg.ema_decay) if cfg.ema else None

        # ── Diffusion schedule ─────────────────────────────────────────────────
        self.T         = cfg.diffusion_steps
        betas          = build_betas(self.T, cfg.beta_schedule)
        self.alpha_bar = torch.tensor(
            np.cumprod(1.0 - betas), dtype=torch.float32, device=self.device)

        # ── Training state ─────────────────────────────────────────────────────
        self.start_epoch    = 1
        self.global_step    = 0
        self.best_val_loss  = float("inf")
        self.best_gen_score = -float("inf")
        self.best_val_epoch = -1
        self.best_gen_epoch = -1

        # ── Resume ────────────────────────────────────────────────────────────
        if resume_ckpt:
            self._resume(resume_ckpt)

        # ── Loggers ───────────────────────────────────────────────────────────
        self.csv_log = CSVLogger(
            os.path.join(cfg.out_dir, "train_log.csv"),
            fieldnames=["epoch", "train_loss", "val_loss", "lr", "time_s"])
        self.tb_log = TBLogger(os.path.join(cfg.out_dir, "tb"))

        # ── Save invariants and config ─────────────────────────────────────────
        inv = dict(
            ae_ckpt          = cfg.ae_ckpt,
            vae_dir          = cfg.vae_dir,
            vae_type         = self.vae_type,
            task             = self.task,
            latent_ch        = cfg.latent_ch,
            H_latent         = cfg.H_latent,
            W_latent         = cfg.W_latent,
            C_feat           = self.C_feat,
            drop_feat_channels = cfg.drop_feat_channels,
            feat_proj_ch     = cfg.feat_proj_ch,
            proj_type        = self.proj_type,
            in_ch            = self.in_ch,
            pred_type        = self.pred_type,
            beta_schedule    = cfg.beta_schedule,
            diffusion_steps  = self.T,
            base_channels    = cfg.base_channels,
            cfg_drop_prob    = cfg.cfg_drop_prob,
            eval_seeds       = cfg.eval_seeds,
            eval_N           = cfg.eval_N,
            eval_gen_steps   = cfg.eval_gen_steps,
            z_mean           = z_mean_l,
            z_std            = z_std_l,
        )
        with open(os.path.join(cfg.out_dir, "invariants.json"), "w") as f:
            json.dump(inv, f, indent=2)
        cfg.save(os.path.join(cfg.out_dir, "latent_config.json"))

        # ── Header ────────────────────────────────────────────────────────────
        gen_metric = "TopK@1%" if self.task == "DRC" else "MAE"
        print("=" * 70)
        print(f"  LDM | task={self.task} | device={self.device}")
        print(f"  VAE: {self.vae_type} epoch={ae_ck['epoch']} | "
              f"latent {cfg.latent_ch}×{cfg.H_latent}×{cfg.W_latent}")
        print(f"  Features: C={self.C_feat} → {cfg.feat_proj_ch}ch "
              f"({self.proj_type}) | dropped={cfg.drop_feat_channels}")
        print(f"  UNet {n_unet/1e6:.1f}M + Proj {n_proj/1e6:.2f}M | "
              f"T={self.T} {cfg.beta_schedule} | pred={self.pred_type}")
        print(f"  CFG drop={cfg.cfg_drop_prob} | "
              f"eval: {cfg.eval_gen_steps} steps "
              f"N={cfg.eval_N} seeds={cfg.eval_seeds}")
        print(f"  Best gen criterion: {gen_metric} "
              f"(averaged over {len(cfg.eval_seeds)} seeds)")
        print("=" * 70)

        # Sanity forward pass
        with torch.no_grad():
            z0   = self._encode(_l[:2].float().to(self.device))
            proj = self.feat_proj(_f[:2].float().to(self.device))
        print(f"[Sanity] z_norm mean={z0.mean():.3f} std={z0.std():.3f} "
              f"(target ~N(0,1))")
        print(f"[Sanity] feat_proj: {tuple(_f[:2].shape)} → {tuple(proj.shape)}")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # Normalization
    # ─────────────────────────────────────────────────────────────────────────
    def _norm(self, z):   return (z - self.z_mean) / (self.z_std + 1e-8)
    def _denorm(self, z): return z * self.z_std + self.z_mean

    @torch.no_grad()


    def _encode(self, lbl):
        z_norm = self._norm(self.ae.encode_to_z(lbl))
        return z_norm.clamp(-3.0, 3.0)
   # def _encode(self, lbl): return self._norm(self.ae.encode_to_z(lbl))

    @torch.no_grad()
    def _decode(self, z):   return self.ae.decode_from_z(self._denorm(z))

    # ─────────────────────────────────────────────────────────────────────────
    # Diffusion helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _abar(self, t):
        return self.alpha_bar[t].view(-1, 1, 1, 1)

    def _q_sample(self, z0, t, noise):
        ab = self._abar(t)
        return torch.sqrt(ab) * z0 + torch.sqrt(1.0 - ab) * noise

    def _target(self, z0, noise, t):
        ab = self._abar(t)
        return torch.sqrt(ab) * noise - torch.sqrt(1.0 - ab) * z0 \
               if self.pred_type == "v" else noise

    def _z0_from_pred(self, zt, pred, t):
        ab = self._abar(t)
        return torch.sqrt(ab) * zt - torch.sqrt(1.0 - ab) * pred \
               if self.pred_type == "v" \
               else (zt - torch.sqrt(1.0 - ab) * pred) / torch.sqrt(ab + 1e-12)

    def _eps_from_pred(self, zt, pred, t):
        ab = self._abar(t)
        return torch.sqrt(1.0 - ab) * zt + torch.sqrt(ab) * pred \
               if self.pred_type == "v" else pred

    def _snr_weights(self, t):
        if self.cfg.min_snr_gamma <= 0:
            return None
        ab  = self._abar(t)
        snr = ab / torch.clamp(1.0 - ab, min=1e-12)
        g   = torch.tensor(self.cfg.min_snr_gamma, device=snr.device, dtype=snr.dtype)
        w   = torch.minimum(snr, g) / (snr + 1.0) if self.pred_type == "v" \
              else torch.minimum(snr, g) / torch.clamp(snr, min=1e-12)
        return w.view(-1)

    def _get_lr(self):
        if self.global_step < self.cfg.warmup_steps:
            return self.cfg.lr * (self.global_step + 1) / self.cfg.warmup_steps
        return self.cfg.lr

    # ─────────────────────────────────────────────────────────────────────────
    # CFG dropout
    # ─────────────────────────────────────────────────────────────────────────
    def _project(self, feat):
        return self.feat_proj(feat)

    def _cfg_drop(self, cond):
        if self.cfg.cfg_drop_prob <= 0:
            return cond
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cfg.cfg_drop_prob
        out  = cond.clone()
        out[mask] = 0.0
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # DDIM sampler — matches latent_sampler.py exactly
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ddim(self, feat, x_init, steps, eta, cfg_scale, seed):
        """
        Deterministic DDIM sampler.
        x_init: pre-seeded noise tensor (B, latent_ch, H, W)
        All stochasticity comes from x_init — eta=0 by default.
        If eta > 0, extra noise is seeded per-step from a generator derived
        from seed to ensure reproducibility.
        """
        self.net.eval(); self.feat_proj.eval()
        B      = feat.shape[0]
        g      = torch.Generator(device=feat.device)
        g.manual_seed(int(seed) & 0x7FFFFFFF)
        cond   = self._project(feat)
        null   = torch.zeros_like(cond)
        z      = x_init
        ts     = _make_ts(self.T, steps, feat.device)

        for i, t_val in enumerate(ts):
            t     = t_val.expand(B)
            t_emb = sinusoidal_embedding(t, 128)
            ab_t  = self._abar(t)
            x_in  = torch.cat([z, cond], dim=1)
            if cfg_scale > 0.0:
                pred_u = self.net(torch.cat([z, null], 1), t_emb)
                pred_c = self.net(x_in, t_emb)
                pred   = pred_u + cfg_scale * (pred_c - pred_u)
            else:
                pred = self.net(x_in, t_emb)
            z0  = self._z0_from_pred(z, pred, t)
            eps = self._eps_from_pred(z, pred, t)
            if i == len(ts) - 1:
                z = z0; break
            t_next  = ts[i+1].expand(B)
            ab_next = self._abar(t_next)
            sigma   = eta * torch.sqrt(
                (1 - ab_next) / (1 - ab_t + 1e-12) *
                torch.clamp(1 - ab_t / (ab_next + 1e-12), min=0.0))
            # Seeded noise for reproducibility when eta > 0
            noise = torch.randn(z.shape, device=z.device, generator=g) \
                    if eta > 0 else 0.0
            z = (torch.sqrt(ab_next) * z0 +
                 torch.sqrt(torch.clamp(1 - ab_next - sigma**2, min=0.0)) * eps +
                 sigma * noise)
        return z

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    def _save(self, path: str, epoch: int, extra: dict | None = None):
        """
        Saves current model weights (may be EMA or raw depending on context).
        See docstring for checkpoint policy.
        """
        ck = dict(
            # Model weights — may be EMA (best_*.pt) or raw (latest.pt)
            net             = self.net.state_dict(),
            feat_proj       = self.feat_proj.state_dict(),
            # Optimizer — only meaningful in latest.pt for resuming
            opt             = self.optimizer.state_dict(),
            # Training state
            epoch           = epoch,
            global_step     = self.global_step,
            best_val_loss   = self.best_val_loss,
            best_val_epoch  = self.best_val_epoch,
            best_gen_score  = self.best_gen_score,
            best_gen_epoch  = self.best_gen_epoch,
            # Everything sampler needs — no external config required
            vae_type        = self.vae_type,
            task            = self.task,
            pred_type       = self.pred_type,
            beta_schedule   = self.cfg.beta_schedule,
            diffusion_steps = self.T,
            latent_ch       = self.cfg.latent_ch,
            H_latent        = self.cfg.H_latent,
            W_latent        = self.cfg.W_latent,
            C_feat          = self.C_feat,
            drop_feat_ch    = self.cfg.drop_feat_channels,
            feat_proj_ch    = self.cfg.feat_proj_ch,
            proj_type       = self.proj_type,
            in_ch           = self.in_ch,
            base_channels   = self.cfg.base_channels,
            cfg_drop_prob   = self.cfg.cfg_drop_prob,
            ae_ckpt         = self.cfg.ae_ckpt,
            vae_dir         = self.cfg.vae_dir,
            z_mean          = self.z_mean.squeeze().tolist(),
            z_std           = self.z_std.squeeze().tolist(),
        )
        if extra:
            ck.update(extra)
        if self.ema is not None:
            ck["ema"] = self.ema.state_dict()
        torch.save(ck, path)

    def _resume(self, path: str):
        """Resume from latest.pt — restores full training state."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"], strict=True)
        self.feat_proj.load_state_dict(ck["feat_proj"], strict=True)
        self.optimizer.load_state_dict(ck["opt"])
        if self.ema and "ema" in ck:
            self.ema.load_state_dict(ck["ema"])
        self.start_epoch    = int(ck.get("epoch", 0)) + 1
        self.global_step    = int(ck.get("global_step", 0))
        self.best_val_loss  = float(ck.get("best_val_loss",  float("inf")))
        self.best_val_epoch = int(  ck.get("best_val_epoch", -1))
        self.best_gen_score = float(ck.get("best_gen_score", -float("inf")))
        self.best_gen_epoch = int(  ck.get("best_gen_epoch", -1))
        print(f"[Resume] epoch {self.start_epoch} | "
              f"best_val={self.best_val_loss:.6f} @ E{self.best_val_epoch} | "
              f"best_gen={self.best_gen_score:.4f} @ E{self.best_gen_epoch}")

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        cfg = self.cfg
        print(f"[Train] E{self.start_epoch}→{cfg.epochs} | "
              f"{len(self.ds_train)} train | {len(self.ds_val)} val\n")

        for epoch in range(self.start_epoch, cfg.epochs + 1):
            self.net.train(); self.feat_proj.train()
            meter = AvgMeter()
            t0    = time.time()

            for feat, lbl, _ in self.loader_train:
                feat = feat.float().to(self.device)
                lbl  = lbl.float().to(self.device)
                B    = lbl.size(0)
                lr   = self._get_lr()
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                z0    = self._encode(lbl)
                t     = torch.randint(0, self.T, (B,), device=self.device)
                noise = torch.randn_like(z0)
                zt    = self._q_sample(z0, t, noise)
                t_emb = sinusoidal_embedding(t, 128)
                cond  = self._cfg_drop(self._project(feat))
                pred  = self.net(torch.cat([zt, cond], dim=1), t_emb)
                tgt   = self._target(z0, noise, t)
                mse   = ((pred - tgt) ** 2).mean(dim=(1, 2, 3))
                w     = self._snr_weights(t)
                loss  = (mse * w).mean() if w is not None else mse.mean()

                if cfg.aux_weight > 0:
                    mask = t < cfg.aux_t_cutoff
                    if mask.any():
                        z0hat = self._z0_from_pred(zt[mask], pred[mask], t[mask])
                        loss  = loss + cfg.aux_weight * F.l1_loss(z0hat, z0[mask])

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters()) + list(self.feat_proj.parameters()),
                    cfg.grad_clip)
                self.optimizer.step()
                self.global_step += 1
                if self.ema:
                    self.ema.update(self._combined)
                meter.update(loss.item(), B)

            elapsed = time.time() - t0
            print(f"[E{epoch:03d}] loss={meter.avg:.5f} "
                  f"lr={self._get_lr():.2e} "
                  f"time={elapsed:.0f}s mem={gpu_mem_gb():.1f}GB")

            val_loss = None
            if epoch % cfg.eval_every == 0:
                val_loss = self._evaluate(epoch)

            self.csv_log.log(dict(
                epoch=epoch, train_loss=f"{meter.avg:.6f}",
                val_loss=f"{val_loss:.6f}" if val_loss else "",
                lr=f"{self._get_lr():.2e}", time_s=f"{elapsed:.1f}"))
            self.tb_log.add_scalar("train/loss", meter.avg, epoch)
            self.tb_log.flush()
            # latest.pt always uses raw weights — use for resuming
            self._save(os.path.join(cfg.out_dir, "latest.pt"), epoch)

        print(f"\n[Done] best_val={self.best_val_loss:.6f} @ E{self.best_val_epoch}")
        print(f"[Done] best_gen={self.best_gen_score:.4f}  @ E{self.best_gen_epoch}")

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation — uses same protocol as final sampler
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, epoch: int) -> float:
        cfg = self.cfg

        with _EMASwap(self.ema, self._combined):
            self.net.eval(); self.feat_proj.eval()

            # ── Denoising val loss ─────────────────────────────────────────────
            rng   = torch.Generator(device=self.device)
            rng.manual_seed(cfg.seed)
            val_m = AvgMeter()
            for feat, lbl, _ in self.loader_val:
                feat  = feat.float().to(self.device)
                lbl   = lbl.float().to(self.device)
                z0    = self._encode(lbl)
                t     = torch.randint(0, self.T, (lbl.size(0),),
                                      device=self.device, generator=rng)
                noise = torch.randn(z0.shape, device=self.device, generator=rng)
                zt    = self._q_sample(z0, t, noise)
                t_emb = sinusoidal_embedding(t, 128)
                pred  = self.net(torch.cat([zt, self._project(feat)], 1), t_emb)
                tgt   = self._target(z0, noise, t)
                mse   = ((pred - tgt) ** 2).mean(dim=(1, 2, 3))
                w     = self._snr_weights(t)
                val_m.update(
                    ((mse*w).mean() if w is not None else mse.mean()).item(),
                    lbl.size(0))

            val_loss = val_m.avg
            print(f"\n  [E{epoch:03d}] val_loss={val_loss:.6f}", end="")
            if val_loss < self.best_val_loss:
                self.best_val_loss  = val_loss
                self.best_val_epoch = epoch
                # Saved inside EMASwap — stores EMA weights
                self._save(os.path.join(cfg.out_dir, "best_val.pt"), epoch)
                print("  ★ best_val", end="")
            print()

            # ── Generation eval — averaged over eval_seeds ─────────────────────
            # Uses same per-sample stable seeding as final sampler.
            # eval_N draws per sample, averaged — controls cost vs stability.
            cfg_scale_eff = cfg.eval_cfg_scale if cfg.cfg_drop_prob > 0 else 0.0
            seed_scores   = []

            for eval_seed in cfg.eval_seeds:
                all_m  = []
                all_tv = []
                for nb, (feat, lbl, names) in enumerate(self.loader_val):
                    if cfg.eval_gen_batches > 0 and nb >= cfg.eval_gen_batches:
                        break
                    feat = feat.float().to(self.device)
                    lbl  = lbl.float().to(self.device)
                    B    = feat.shape[0]

                    # N draws per sample, averaged
                    gens = []
                    for k in range(cfg.eval_N):
                        x_init = _make_noise(
                            B, cfg.latent_ch, cfg.H_latent, cfg.W_latent,
                            self.device, eval_seed, list(names), k)
                        z_gen = self._ddim(
                            feat, x_init,
                            steps=cfg.eval_gen_steps,
                            eta=cfg.eval_gen_eta,
                            cfg_scale=cfg_scale_eff,
                            seed=eval_seed)
                        gens.append(self._decode(z_gen))
                    mean_pred = torch.stack(gens, 0).mean(0)

                    gt_np = lbl.clamp(0,1).cpu().numpy()[:,0]
                    pr_np = mean_pred.clamp(0,1).cpu().numpy()[:,0]
                    for i in range(gt_np.shape[0]):
                        all_m.append(_compute_metrics(gt_np[i], pr_np[i], self.task))
                        all_tv.append(_trivial_metrics(gt_np[i], self.task, self.train_mean))

                def mn(lst, k):
                    v = [x[k] for x in lst if not np.isnan(x.get(k, float("nan")))]
                    return float(np.mean(v)) if v else float("nan")

                seed_scores.append(_gen_score({k: mn(all_m, k) for k in all_m[0]}, self.task))

            # Average score over eval_seeds before checkpoint decision
            gen_score = float(np.nanmean(seed_scores))
            n_gen     = len(all_m)

            if self.task == "DRC":
                topk = float(np.nanmean([m["topk_1pct"] for m in all_m]))
                nmae = float(np.nanmean([m["nz_mae"]    for m in all_m]))
                t_topk = float(np.nanmean([m["topk_1pct"] for m in all_tv]))
                print(f"  [E{epoch:03d}] gen({n_gen} samples) "
                      f"TopK@1%={topk:.4f} (trivial={t_topk:.4f}) "
                      f"nz_MAE={nmae:.5f} "
                      f"[avg over {len(cfg.eval_seeds)} seeds]")
            else:
                mae  = float(np.nanmean([m["mae"]     for m in all_m]))
                ssim = float(np.nanmean([m["ssim"]    for m in all_m]))
                t_mae= float(np.nanmean([m["mae"]     for m in all_tv]))
                print(f"  [E{epoch:03d}] gen({n_gen} samples) "
                      f"MAE={mae:.5f} (trivial={t_mae:.5f}) "
                      f"SSIM={ssim:.4f} "
                      f"[avg over {len(cfg.eval_seeds)} seeds]")

            if gen_score > self.best_gen_score:
                self.best_gen_score = gen_score
                self.best_gen_epoch = epoch
                # Saved inside EMASwap — stores EMA weights
                self._save(os.path.join(cfg.out_dir, "best_gen.pt"), epoch,
                           extra=dict(gen_score=gen_score))
                crit = "TopK@1%" if self.task == "DRC" else "-MAE"
                print(f"  [E{epoch:03d}] ★ best_gen ({crit}={gen_score:.4f})")

        self.tb_log.add_scalar("eval/val_loss",  val_loss,  epoch)
        self.tb_log.add_scalar("eval/gen_score", gen_score, epoch)
        self.tb_log.flush()
        return val_loss


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser("LDM Trainer")
    # Required — no defaults
    p.add_argument("--ae_ckpt",  required=True,
                   help="Path to VAE/AE checkpoint (e.g. best_ldm.pt)")
    p.add_argument("--vae_dir",  required=True,
                   help="Directory containing VAE model Python file")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory for this LDM run")
    # Optional overrides
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--cfg_drop_prob",   type=float, default=None,
                   help="0.0 for Congestion, 0.1 for DRC")
    p.add_argument("--eval_every",      type=int,   default=None)
    p.add_argument("--eval_gen_steps",  type=int,   default=None,
                   help="DDIM steps for validation eval (should match sampler --steps)")
    p.add_argument("--eval_gen_batches",type=int,   default=None)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--resume",          default=None,
                   help="Path to latest.pt to resume from")
    return p.parse_args()


def main():
    args = build_args()
    cfg  = LatentConfig()

    # Required fields
    cfg.ae_ckpt = args.ae_ckpt
    cfg.vae_dir = args.vae_dir
    cfg.out_dir = args.out_dir

    # Optional overrides
    for attr in ["epochs", "batch_size", "lr", "cfg_drop_prob",
                 "eval_every", "eval_gen_steps", "eval_gen_batches", "seed"]:
        val = getattr(args, attr)
        if val is not None:
            setattr(cfg, attr, val)

    trainer = LatentTrainer(cfg, resume_ckpt=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()