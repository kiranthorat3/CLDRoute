#!/usr/bin/env python3
"""
latent_trainer.py — LDM trainer for ldm_control (ControlNet conditioning).

Key differences from ldm_unified/latent_trainer.py:
  - feat_proj replaced by MultiScaleConditioner
  - UNet in_ch = latent_ch only (no feature concatenation)
  - Conditioning injected at three scales: 64×64, 32×32, 16×16
  - CFG dropout zeros all three conditioning scales together
  - DDIM: conditional = net(z, c64, c32, c16)
           unconditional = net(z, None, None, None)
  - Checkpoint stores "conditioner" key, not "feat_proj"
  - EMA covers net + conditioner via _Combined module

Usage:
  # DRC
  python latent_trainer.py \
    --ae_ckpt  .../drc_vae/runs/vae_DRC_expanded/best_ldm.pt \
    --vae_dir  .../drc_vae \
    --out_dir  ./runs/ldm_DRC_control \
    --cfg_drop_prob 0.1 \
    --epochs 200

  # Resume
  python latent_trainer.py ... --resume ./runs/ldm_DRC_control/latest.pt
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
from models        import LatentUNet, MultiScaleConditioner, sinusoidal_embedding
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
# Module import — sanitized name
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

    stored_vae_dir = ck.get("vae_dir", None)
    if stored_vae_dir and \
       os.path.realpath(stored_vae_dir) != os.path.realpath(vae_dir):
        print(f"  [VAE] WARNING: supplied vae_dir differs from stored:")
        print(f"    supplied: {vae_dir}")
        print(f"    stored:   {stored_vae_dir}")

    if model_type == "LabelVAE_v2":
        mod   = _import_from_file(os.path.join(vae_dir, "vae_model.py"))
        model = mod.LabelVAE(
            C_label   = int(ck["C_label"]),
            latent_ch = int(ck["latent_ch"]),
            base_ch   = int(ck.get("base_ch", 64)),
            log_scale = float(ck.get("log_scale", 10.0)),
        ).to(device)
    elif model_type in ("CongestionVAE_v2", "CongestionVAE"):
        mod   = _import_from_file(
            os.path.join(vae_dir, "vae_model_congestion.py"))
        model = mod.CongestionVAE(
            C_label    = int(ck["C_label"]),
            latent_ch  = int(ck["latent_ch"]),
            base_ch    = int(ck.get("base_ch", 64)),
            logvar_min = float(ck.get("logvar_min", -4.0)),
            logvar_max = float(ck.get("logvar_max",  4.0)),
        ).to(device)
    elif model_type in ("CongestionAE_v2", "CongestionAE"):
        mod   = _import_from_file(
            os.path.join(vae_dir, "ae_model_congestion.py"))
        model = mod.CongestionAE(
            C_label   = int(ck["C_label"]),
            latent_ch = int(ck["latent_ch"]),
            base_ch   = int(ck.get("base_ch", 64)),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type '{model_type}' in {ckpt_path}")

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
            f"Run vae_latent_stats.py --ckpt {ckpt_path} first.")
    with open(stats_path) as f:
        stats = json.load(f)
    z_mean_l = stats["z_mean"]
    z_std_l  = stats["z_std"]
    assert len(z_mean_l) == latent_ch
    if min(z_std_l) < 0.05:
        raise ValueError(
            f"Latent std too small (min={min(z_std_l):.4f}). "
            f"Retrain VAE with stronger KL.")
    z_mean = torch.tensor(
        z_mean_l, dtype=torch.float32, device=device).view(1, latent_ch, 1, 1)
    z_std  = torch.tensor(
        z_std_l, dtype=torch.float32, device=device).view(1, latent_ch, 1, 1)
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

def _safe_pearson(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _metrics_drc(gt, pr):
    k      = max(1, int(0.01 * gt.size))
    gt_idx = np.argsort(gt.ravel())[-k:]
    pr_idx = np.argsort(pr.ravel())[-k:]
    topk   = float(len(np.intersect1d(gt_idx, pr_idx)) / k)
    nz     = gt.ravel() > _NZ_THRESH
    nz_mae = float(np.mean(np.abs(gt.ravel()[nz] - pr.ravel()[nz]))) \
             if nz.any() else float("nan")
    return dict(topk_1pct=topk, nz_mae=nz_mae,
                mae=float(np.mean(np.abs(gt - pr))),
                ssim=float(structural_similarity(gt, pr, data_range=1.0)))

def _metrics_congestion(gt, pr):
    return dict(
        mae=float(np.mean(np.abs(gt - pr))),
        ssim=float(structural_similarity(gt, pr, data_range=1.0)),
        pearson=_safe_pearson(gt, pr))

def _compute_metrics(gt, pr, task):
    return _metrics_drc(gt, pr) if task == "DRC" else _metrics_congestion(gt, pr)

def _trivial_metrics(gt, task, train_mean):
    pr = np.zeros_like(gt) if task == "DRC" else np.full_like(gt, train_mean)
    return _compute_metrics(gt, pr, task)

def _gen_score(m, task):
    return m["topk_1pct"] if task == "DRC" else -m["mae"]


# ─────────────────────────────────────────────────────────────────────────────
# EMA context manager
# ─────────────────────────────────────────────────────────────────────────────
class _EMASwap:
    def __init__(self, ema, module):
        self.ema = ema; self.module = module; self._stored = None

    def __enter__(self):
        if self.ema is not None:
            self._stored = {k: v.clone()
                            for k, v in self.module.state_dict().items()}
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
        assert self.pred_type in ("v", "eps")

        # ── Frozen VAE ────────────────────────────────────────────────────────
        print("[Init] Loading VAE/AE...")
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
        print(f"  z_std: [{min(z_std_l):.3f}, {max(z_std_l):.3f}]")

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

            assert cfg.H_latent == 64 and cfg.W_latent == 64, (
            f"ldm_control requires 64×64 latents, got "
            f"{cfg.H_latent}×{cfg.W_latent}. "
            f"Use a 64×64 VAE/AE or switch to ldm_unified."
        )

        # ── Training mean for Congestion trivial baseline ──────────────────────
        if self.task == "Congestion":
            ds_full  = LatentDataset(
                cfg.csv_train, cfg.feature_dir, cfg.label_dir,
                drop_channels=cfg.drop_feat_channels,
                split="train", verify=False)
            ldr_full = DataLoader(
                ds_full, batch_size=64, shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=_collate, drop_last=False)
            s, n = 0.0, 0
            for _, lb, _ in ldr_full:
                s += float(lb.sum()); n += lb.numel()
            self.train_mean = s / max(n, 1)
            print(f"  Congestion train mean={self.train_mean:.4f}")
        else:
            self.train_mean = 0.0

        # ── ControlNet-style multi-scale conditioner ───────────────────────────
        self.conditioner = MultiScaleConditioner(
            in_ch = self.C_feat,
            base  = cfg.base_channels,
        ).to(self.device)
        self.proj_type = "multiscale_controlnet"

        # UNet input is latent only — no feature concatenation
        self.net = LatentUNet(
            in_ch     = cfg.latent_ch,
            out_ch    = cfg.latent_ch,
            base      = cfg.base_channels,
            t_emb_dim = 128,
            dropout   = cfg.dropout,
        ).to(self.device)

        n_net  = sum(p.numel() for p in self.net.parameters())
        n_cond = sum(p.numel() for p in self.conditioner.parameters())
        print(f"  UNet={n_net/1e6:.1f}M | Conditioner={n_cond/1e6:.2f}M | "
              f"Total={( n_net+n_cond)/1e6:.1f}M | VAE=frozen")

        # ── Optimizer + EMA ───────────────────────────────────────────────────
        params = (list(self.net.parameters()) +
                  list(self.conditioner.parameters()))
        self.optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        class _Combined(torch.nn.Module):
            def __init__(self, net, cond):
                super().__init__()
                self.net  = net
                self.cond = cond

        self._combined = _Combined(self.net, self.conditioner)
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

        if resume_ckpt:
            self._resume(resume_ckpt)

        # ── Loggers ───────────────────────────────────────────────────────────
        self.csv_log = CSVLogger(
            os.path.join(cfg.out_dir, "train_log.csv"),
            fieldnames=["epoch", "train_loss", "val_loss", "lr", "time_s"])
        self.tb_log = TBLogger(os.path.join(cfg.out_dir, "tb"))

        # ── Save invariants ────────────────────────────────────────────────────
        inv = dict(
            ae_ckpt           = cfg.ae_ckpt,
            vae_dir           = cfg.vae_dir,
            vae_type          = self.vae_type,
            task              = self.task,
            latent_ch         = cfg.latent_ch,
            H_latent          = cfg.H_latent,
            W_latent          = cfg.W_latent,
            C_feat            = self.C_feat,
            drop_feat_channels= cfg.drop_feat_channels,
            proj_type         = self.proj_type,
            base_channels     = cfg.base_channels,
            pred_type         = self.pred_type,
            beta_schedule     = cfg.beta_schedule,
            diffusion_steps   = self.T,
            cfg_drop_prob     = cfg.cfg_drop_prob,
            eval_seeds        = cfg.eval_seeds,
            eval_N            = cfg.eval_N,
            eval_gen_steps    = cfg.eval_gen_steps,
            z_mean            = z_mean_l,
            z_std             = z_std_l,
        )
        with open(os.path.join(cfg.out_dir, "invariants.json"), "w") as f:
            json.dump(inv, f, indent=2)
        cfg.save(os.path.join(cfg.out_dir, "latent_config.json"))

        # ── Header ────────────────────────────────────────────────────────────
        gen_metric = "TopK@1%" if self.task == "DRC" else "MAE"
        print("=" * 70)
        print(f"  LDM-Control | task={self.task} | device={self.device}")
        print(f"  VAE: {self.vae_type} epoch={ae_ck['epoch']} | "
              f"latent {cfg.latent_ch}×{cfg.H_latent}×{cfg.W_latent}")
        print(f"  Features: C={self.C_feat} | drop={cfg.drop_feat_channels}")
        print(f"  Conditioner: MultiScale ControlNet-style "
              f"(64×64, 32×32, 16×16) | ZeroConv initialized")
        print(f"  UNet in_ch={cfg.latent_ch} (latent only — no concatenation)")
        print(f"  T={self.T} {cfg.beta_schedule} | pred={self.pred_type} | "
              f"CFG drop={cfg.cfg_drop_prob}")
        print(f"  Best gen: {gen_metric} avg over {len(cfg.eval_seeds)} seeds")
        print("=" * 70)

        # Sanity
        with torch.no_grad():
            z0    = self._encode(_l[:2].float().to(self.device))
            f0    = _f[:2].float().to(self.device)
            c64, c32, c16 = self.conditioner(f0)
            out   = self.net(z0, torch.zeros(2, 128, device=self.device),
                             c64=c64, c32=c32, c16=c16)
            out_u = self.net(z0, torch.zeros(2, 128, device=self.device),
                             c64=None, c32=None, c16=None)
        print(f"[Sanity] z_norm mean={z0.mean():.3f} std={z0.std():.3f}")
        print(f"[Sanity] c64={tuple(c64.shape)} "
              f"c32={tuple(c32.shape)} c16={tuple(c16.shape)}")
        print(f"[Sanity] cond out={tuple(out.shape)} "
              f"null out={tuple(out_u.shape)}")
        print(f"[Sanity] ZeroConv check: "
              f"c64_norm={c64.norm():.4f} (should be ~0 at init)")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # Normalization
    # ─────────────────────────────────────────────────────────────────────────
    def _norm(self, z):   return (z - self.z_mean) / (self.z_std + 1e-8)
    def _denorm(self, z): return z * self.z_std + self.z_mean

    @torch.no_grad()
    #def _encode(self, lbl): return self._norm(self.ae.encode_to_z(lbl))
    def _encode(self, lbl):
        z_norm = self._norm(self.ae.encode_to_z(lbl))
        return z_norm.clamp(-3.0, 3.0)

    @torch.no_grad()
    def _decode(self, z):   return self.ae.decode_from_z(self._denorm(z))

    # ─────────────────────────────────────────────────────────────────────────
    # Conditioning
    # ─────────────────────────────────────────────────────────────────────────
    def _condition(self, feat):
        """feat → (c64, c32, c16)"""
        return self.conditioner(feat)

    def _cfg_drop(self, c64, c32, c16):
        """
        Zero out ALL conditioning scales for the same samples.
        Consistent null conditioning is critical for CFG to work correctly.
        A sample is either fully conditioned or fully null — no mixing.
        """
        if self.cfg.cfg_drop_prob <= 0:
            return c64, c32, c16
        mask = torch.rand(c64.shape[0], device=c64.device) < self.cfg.cfg_drop_prob
        c64 = c64.clone(); c64[mask] = 0.0
        c32 = c32.clone(); c32[mask] = 0.0
        c16 = c16.clone(); c16[mask] = 0.0
        return c64, c32, c16

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
        g   = torch.tensor(self.cfg.min_snr_gamma,
                            device=snr.device, dtype=snr.dtype)
        w   = torch.minimum(snr, g) / (snr + 1.0) if self.pred_type == "v" \
              else torch.minimum(snr, g) / torch.clamp(snr, min=1e-12)
        return w.view(-1)

    def _get_lr(self):
        if self.global_step < self.cfg.warmup_steps:
            return self.cfg.lr * (self.global_step + 1) / self.cfg.warmup_steps
        return self.cfg.lr

    # ─────────────────────────────────────────────────────────────────────────
    # DDIM sampler
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ddim(self, feat, x_init, steps, eta, cfg_scale, seed):
        self.net.eval(); self.conditioner.eval()
        B      = feat.shape[0]
        g      = torch.Generator(device=feat.device)
        g.manual_seed(int(seed) & 0x7FFFFFFF)
        c64, c32, c16 = self._condition(feat)
        z      = x_init
        ts     = _make_ts(self.T, steps, feat.device)

        for i, t_val in enumerate(ts):
            t     = t_val.expand(B)
            t_emb = sinusoidal_embedding(t, 128)
            ab_t  = self._abar(t)

            if cfg_scale > 0.0:
                # Conditional forward
                pred_c = self.net(z, t_emb, c64=c64, c32=c32, c16=c16)
                # Unconditional forward — pass None, UNet uses raw skips
                pred_u = self.net(z, t_emb, c64=None, c32=None, c16=None)
                pred   = pred_u + cfg_scale * (pred_c - pred_u)
            else:
                pred = self.net(z, t_emb, c64=c64, c32=c32, c16=c16)

            z0  = self._z0_from_pred(z, pred, t)
            eps = self._eps_from_pred(z, pred, t)
            if i == len(ts) - 1:
                z = z0; break

            t_next  = ts[i+1].expand(B)
            ab_next = self._abar(t_next)
            sigma   = eta * torch.sqrt(
                (1-ab_next)/(1-ab_t+1e-12) *
                torch.clamp(1 - ab_t/(ab_next+1e-12), min=0.0))
            noise = torch.randn(z.shape, device=z.device, generator=g) \
                    if eta > 0 else 0.0
            z = (torch.sqrt(ab_next) * z0 +
                 torch.sqrt(torch.clamp(1-ab_next-sigma**2, min=0.0)) * eps +
                 sigma * noise)
        return z

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    def _save(self, path: str, epoch: int, extra: dict | None = None):
        ck = dict(
            net             = self.net.state_dict(),
            conditioner     = self.conditioner.state_dict(),
            opt             = self.optimizer.state_dict(),
            epoch           = epoch,
            global_step     = self.global_step,
            best_val_loss   = self.best_val_loss,
            best_val_epoch  = self.best_val_epoch,
            best_gen_score  = self.best_gen_score,
            best_gen_epoch  = self.best_gen_epoch,
            # Everything sampler needs
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
            proj_type       = self.proj_type,
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
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"], strict=True)
        self.conditioner.load_state_dict(ck["conditioner"], strict=True)
        self.optimizer.load_state_dict(ck["opt"])
        if self.ema and "ema" in ck:
            self.ema.load_state_dict(ck["ema"])
        self.start_epoch    = int(ck.get("epoch", 0)) + 1
        self.global_step    = int(ck.get("global_step", 0))
        self.best_val_loss  = float(ck.get("best_val_loss",  float("inf")))
        self.best_val_epoch = int(  ck.get("best_val_epoch", -1))
        self.best_gen_score = float(ck.get("best_gen_score", -float("inf")))
        self.best_gen_epoch = int(  ck.get("best_gen_epoch", -1))
        print(f"[Resume] E{self.start_epoch} | "
              f"best_val={self.best_val_loss:.6f}@E{self.best_val_epoch} | "
              f"best_gen={self.best_gen_score:.4f}@E{self.best_gen_epoch}")

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        cfg = self.cfg
        print(f"[Train] E{self.start_epoch}→{cfg.epochs} | "
              f"{len(self.ds_train)} train | {len(self.ds_val)} val\n")

        for epoch in range(self.start_epoch, cfg.epochs + 1):
            self.net.train(); self.conditioner.train()
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

                # Conditioning with CFG dropout
                c64, c32, c16 = self._cfg_drop(*self._condition(feat))

                pred = self.net(zt, t_emb, c64=c64, c32=c32, c16=c16)
                tgt  = self._target(z0, noise, t)
                mse  = ((pred - tgt) ** 2).mean(dim=(1, 2, 3))
                w    = self._snr_weights(t)
                loss = (mse * w).mean() if w is not None else mse.mean()

                if cfg.aux_weight > 0:
                    mask = t < cfg.aux_t_cutoff
                    if mask.any():
                        z0hat = self._z0_from_pred(zt[mask], pred[mask], t[mask])
                        loss  = loss + cfg.aux_weight * F.l1_loss(z0hat, z0[mask])

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters()) +
                    list(self.conditioner.parameters()),
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
            self._save(os.path.join(cfg.out_dir, "latest.pt"), epoch)

        print(f"\n[Done] best_val={self.best_val_loss:.6f} "
              f"@ E{self.best_val_epoch}")
        print(f"[Done] best_gen={self.best_gen_score:.4f}  "
              f"@ E{self.best_gen_epoch}")

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, epoch: int) -> float:
        cfg = self.cfg
        with _EMASwap(self.ema, self._combined):
            self.net.eval(); self.conditioner.eval()

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
                c64, c32, c16 = self._condition(feat)
                pred  = self.net(zt, t_emb, c64=c64, c32=c32, c16=c16)
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
                self._save(os.path.join(cfg.out_dir, "best_val.pt"), epoch)
                print("  ★ best_val", end="")
            print()

            # ── Generation eval — averaged over eval_seeds ─────────────────────
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
                        all_tv.append(_trivial_metrics(gt_np[i], self.task,
                                                       self.train_mean))

                def mn(lst, k):
                    v = [x[k] for x in lst
                         if not np.isnan(x.get(k, float("nan")))]
                    return float(np.mean(v)) if v else float("nan")

                seed_scores.append(
                    _gen_score({k: mn(all_m, k) for k in all_m[0]}, self.task))

            gen_score = float(np.nanmean(seed_scores))
            n_gen     = len(all_m)

            if self.task == "DRC":
                topk = float(np.nanmean([m["topk_1pct"] for m in all_m]))
                nmae = float(np.nanmean([m["nz_mae"]    for m in all_m]))
                t_topk = float(np.nanmean([m["topk_1pct"] for m in all_tv]))
                print(f"  [E{epoch:03d}] gen({n_gen}) "
                      f"TopK@1%={topk:.4f} (trivial={t_topk:.4f}) "
                      f"nz_MAE={nmae:.5f} "
                      f"[{len(cfg.eval_seeds)} seeds]")
            else:
                mae  = float(np.nanmean([m["mae"]  for m in all_m]))
                ssim = float(np.nanmean([m["ssim"] for m in all_m]))
                t_mae= float(np.nanmean([m["mae"]  for m in all_tv]))
                print(f"  [E{epoch:03d}] gen({n_gen}) "
                      f"MAE={mae:.5f} (trivial={t_mae:.5f}) "
                      f"SSIM={ssim:.4f} "
                      f"[{len(cfg.eval_seeds)} seeds]")

            if gen_score > self.best_gen_score:
                self.best_gen_score = gen_score
                self.best_gen_epoch = epoch
                self._save(os.path.join(cfg.out_dir, "best_gen.pt"), epoch,
                           extra=dict(gen_score=gen_score))
                crit = "TopK@1%" if self.task == "DRC" else "-MAE"
                print(f"  [E{epoch:03d}] ★ best_gen ({crit}={gen_score:.4f})")

        self.tb_log.add_scalar("eval/val_loss",  val_loss,  epoch)
        self.tb_log.add_scalar("eval/gen_score", gen_score, epoch)
        self.tb_log.flush()
        print()
        return val_loss


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser("LDM-Control Trainer")
    p.add_argument("--ae_ckpt",         required=True)
    p.add_argument("--vae_dir",         required=True)
    p.add_argument("--out_dir",         required=True)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--cfg_drop_prob",   type=float, default=None)
    p.add_argument("--eval_every",      type=int,   default=None)
    p.add_argument("--eval_gen_steps",  type=int,   default=None)
    p.add_argument("--eval_gen_batches",type=int,   default=None)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--resume",          default=None)
    return p.parse_args()


def main():
    args = build_args()
    cfg  = LatentConfig()
    cfg.ae_ckpt = args.ae_ckpt
    cfg.vae_dir = args.vae_dir
    cfg.out_dir = args.out_dir
    for attr in ["epochs", "batch_size", "lr", "cfg_drop_prob",
                 "eval_every", "eval_gen_steps", "eval_gen_batches", "seed"]:
        val = getattr(args, attr)
        if val is not None:
            setattr(cfg, attr, val)
    trainer = LatentTrainer(cfg, resume_ckpt=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()