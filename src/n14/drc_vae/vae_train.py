#!/usr/bin/env python3
"""
vae_train.py — DRC VAE training for CircuitNet-N14.

N14 update:
  - same DRC VAE baseline family as N28
  - hotspot metric/printout follows cfg.hotspot_q
  - latent readiness thresholds moved into config
  - explicit LDM handoff fallback:
      best_ldm.pt if available, otherwise best_topk.pt
  - writes ldm_handoff.txt for downstream use

Checkpoint policy:
  best_topk.pt   — best hotspot-overlap metric on val
  best_focal.pt  — best focal reconstruction on val
  best_ldm.pt    — best TopK among latent-ready epochs (preferred for LDM)
"""

import os
import json
import time
import math
import random
import argparse
import numpy as np
import torch
from torch.optim import AdamW

from vae_config import VAEConfig
from vae_model import (
    LabelVAE,
    focal_recon_loss,
    hotspot_loss,
    free_bits_kl,
    _LOGVAR_MIN,
    _LOGVAR_MAX,
)
from vae_data import make_loaders


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict({
            k: v.to(next(model.parameters()).device)
            for k, v in self.shadow.items()
        })

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, s):
        self.decay = float(s.get("decay", self.decay))
        shadow = s.get("shadow", {})
        self.shadow = {k: v.clone().float() for k, v in shadow.items()}


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule — cosine decay after warmup
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def hotspot_pct(q: float) -> float:
    return 100.0 * (1.0 - q)

def hotspot_pct_str(q: float) -> str:
    pct = hotspot_pct(q)
    if pct >= 1.0:
        return f"{pct:.0f}%"
    if pct >= 0.1:
        return f"{pct:.1f}%"
    if pct >= 0.01:
        return f"{pct:.2f}%"
    return f"{pct:.4f}%"

def latent_ready(min_ch_std, mean_ch_std, clamp_frac_low, cfg):
    return (
        min_ch_std > cfg.ldm_min_ch_std and
        mean_ch_std > cfg.ldm_mean_ch_std and
        clamp_frac_low < cfg.ldm_max_clamp_frac
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def save_ckpt(path, model, ema, optimizer, epoch, cfg, meta, global_step):
    ck = {
        "model_type": "LabelVAE_v2",
        "C_label": cfg.C_label,
        "latent_ch": cfg.latent_ch,
        "base_ch": cfg.base_ch,
        "log_scale": cfg.log_scale,
        "task": cfg.task,
        "tech": cfg.tech,
        "net": model.state_dict(),
        "opt": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "focal_gamma": cfg.focal_gamma,
        "hotspot_weight": cfg.hotspot_weight,
        "hotspot_q": cfg.hotspot_q,
        "free_bits": cfg.free_bits,
        "beta_target": cfg.beta_target,
    }
    ck.update(meta)
    if ema is not None:
        ck["ema"] = ema.state_dict()
    torch.save(ck, path)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser("LabelVAE training — N14 DRC")
    p.add_argument("--task", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--beta_target", type=float, default=None)
    p.add_argument("--hotspot_weight", type=float, default=None)
    p.add_argument("--hotspot_q", type=float, default=None)
    p.add_argument("--free_bits", type=float, default=None)
    p.add_argument("--latent_ch", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--label_dir", default=None)
    p.add_argument("--csv_train", default=None)
    p.add_argument("--csv_val", default=None)
    p.add_argument("--csv_test", default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_args()
    cfg = VAEConfig()
    for attr in [
        "task", "out_dir", "epochs", "batch_size", "lr",
        "beta_target", "hotspot_weight", "hotspot_q", "free_bits",
        "latent_ch", "seed", "eval_every",
        "label_dir", "csv_train", "csv_val", "csv_test"
    ]:
        val = getattr(args, attr)
        if val is not None:
            setattr(cfg, attr, val)

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train, ds_val, loader_train, loader_val = make_loaders(cfg)

    model = LabelVAE(
        C_label=cfg.C_label,
        latent_ch=cfg.latent_ch,
        base_ch=cfg.base_ch,
        log_scale=cfg.log_scale,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema else None

    steps_per_epoch = len(loader_train)
    total_steps = cfg.epochs * steps_per_epoch
    topk_tag = hotspot_pct_str(cfg.hotspot_q)

    print("=" * 72)
    print(f"  LabelVAE | tech={cfg.tech} | task={cfg.task} | device={device}")
    print(f"  train={len(ds_train)} | val={len(ds_val)}")
    print(f"  arch: {cfg.H}×{cfg.W} → latent {cfg.latent_ch}×64×64")
    print(f"  params: {n_params/1e6:.2f}M")
    print(f"  loss: focal(γ={cfg.focal_gamma}) + "
          f"{cfg.hotspot_weight}×hotspot@top {topk_tag}")
    print(f"  KL:   free_bits={cfg.free_bits} | beta 0→{cfg.beta_target} "
          f"over {cfg.beta_warmup_epochs} epochs")
    print(f"  LR:   warmup {cfg.warmup_steps} steps then cosine to "
          f"{cfg.lr * cfg.min_lr_ratio:.2e}")
    print(f"  checkpoints: best_topk | best_focal | best_ldm (LDM-ready)")
    print(f"  LDM gate: min_ch_std>{cfg.ldm_min_ch_std} "
          f"mean_ch_std>{cfg.ldm_mean_ch_std} "
          f"clamp_frac<{cfg.ldm_max_clamp_frac}")
    print("=" * 72)

    # Sanity
    lbl0, _ = next(iter(loader_train))
    lbl0 = lbl0.float().to(device)
    with torch.no_grad():
        r, mu, lv, z = model(lbl0, sample=True)
    print(f"[Sanity] input={tuple(lbl0.shape)} z={tuple(z.shape)} "
          f"recon=[{r.min():.3f},{r.max():.3f}]")
    print(f"[Sanity] mu: mean={mu.mean():.4f} std={mu.std():.4f} | "
          f"logvar: mean={lv.mean():.4f} min={lv.min():.4f}")
    print()

    best_topk = -1.0
    best_focal = float("inf")
    best_ldm = -1.0
    best_topk_epoch = -1
    best_focal_epoch = -1
    best_ldm_epoch = -1
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        beta = cfg.beta_at_epoch(epoch)
        sum_focal = 0.0
        sum_hot = 0.0
        sum_kl = 0.0
        n_batches = 0
        t0 = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for lbl, _ in loader_train:
            lbl = lbl.float().to(device)

            lr_now = get_lr(
                global_step, cfg.warmup_steps, total_steps, cfg.lr,
                min_lr_ratio=cfg.min_lr_ratio
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            recon, mu, logvar, z = model(lbl, sample=True)
            loss_focal = focal_recon_loss(recon, lbl, gamma=cfg.focal_gamma)
            loss_hot = hotspot_loss(recon, lbl, q=cfg.hotspot_q)
            loss_kl = free_bits_kl(mu, logvar, free_bits=cfg.free_bits)
            loss = loss_focal + cfg.hotspot_weight * loss_hot + beta * loss_kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            global_step += 1

            if ema is not None:
                ema.update(model)

            sum_focal += loss_focal.item()
            sum_hot += loss_hot.item()
            sum_kl += loss_kl.item()
            n_batches += 1

        avg_focal = sum_focal / max(n_batches, 1)
        avg_hot = sum_hot / max(n_batches, 1)
        avg_kl = sum_kl / max(n_batches, 1)
        elapsed = time.time() - t0
        mem_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0
        )
        lr_now = get_lr(
            global_step, cfg.warmup_steps, total_steps, cfg.lr,
            min_lr_ratio=cfg.min_lr_ratio
        )

        print(f"[E{epoch:03d}] focal={avg_focal:.5f} hot={avg_hot:.5f} "
              f"kl={avg_kl:.4f} beta={beta:.4f} "
              f"lr={lr_now:.2e} time={elapsed:.0f}s mem={mem_gb:.1f}GB")

        # ── Evaluation ────────────────────────────────────────────────────────
        if epoch % cfg.eval_every == 0:
            model.eval()

            if ema is not None:
                saved = {k: v.clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)

            val_focal = 0.0
            val_kl = 0.0
            val_n = 0.0
            mae_list = []
            topk_list = []
            ch_mu_stds = []
            logvar_all = []

            with torch.no_grad():
                for lbl, _ in loader_val:
                    lbl = lbl.float().to(device)
                    recon, mu, logvar, z = model(lbl, sample=False)

                    val_focal += focal_recon_loss(
                        recon, lbl, gamma=cfg.focal_gamma
                    ).item() * lbl.shape[0]
                    val_kl += free_bits_kl(
                        mu, logvar, free_bits=cfg.free_bits
                    ).item() * lbl.shape[0]
                    val_n += lbl.shape[0]

                    ch_mu_stds.append(mu.std(dim=(0, 2, 3)).cpu().numpy())
                    logvar_all.append(logvar.detach().cpu().float())

                    gt_np = lbl.cpu().numpy()[:, 0]
                    pr_np = recon.cpu().numpy()[:, 0]
                    for i in range(gt_np.shape[0]):
                        mae_list.append(float(np.mean(np.abs(gt_np[i] - pr_np[i]))))
                        k = max(1, int((1.0 - cfg.hotspot_q) * gt_np[i].size))
                        gt_idx = np.argsort(gt_np[i].ravel())[-k:]
                        pr_idx = np.argsort(pr_np[i].ravel())[-k:]
                        topk_list.append(float(len(np.intersect1d(gt_idx, pr_idx)) / k))

            val_focal /= max(val_n, 1)
            val_kl /= max(val_n, 1)
            mean_topk = float(np.mean(topk_list))

            ch_stds_avg = np.mean(ch_mu_stds, axis=0)
            mean_ch_std = float(np.mean(ch_stds_avg))
            min_ch_std = float(np.min(ch_stds_avg))

            lv_cat = torch.cat([lv.reshape(-1) for lv in logvar_all])
            frac_low = float((lv_cat <= _LOGVAR_MIN + 1e-4).float().mean())
            frac_high = float((lv_cat >= _LOGVAR_MAX - 1e-4).float().mean())

            ready = latent_ready(min_ch_std, mean_ch_std, frac_low, cfg)

            print(f"  [EVAL  E{epoch:03d}] "
                  f"val_focal={val_focal:.5f} val_kl={val_kl:.4f} | "
                  f"MAE={np.mean(mae_list):.5f} TopK@{topk_tag}={mean_topk:.4f}")
            print(f"  [LATENT E{epoch:03d}] "
                  f"ch_std_mean={mean_ch_std:.4f} "
                  f"ch_std_min={min_ch_std:.4f} — "
                  f"{'✓ LDM-ready' if ready else '✗ not LDM-ready'}")
            print(f"  [LOGVAR E{epoch:03d}] "
                  f"clamp_low={frac_low:.3f} clamp_high={frac_high:.3f} "
                  f"({'clamp dominant' if frac_low > cfg.ldm_max_clamp_frac else 'clamp OK'})")
            ch_str = " ".join(f"{s:.2f}" for s in ch_stds_avg)
            print(f"  [CH_STD E{epoch:03d}] [{ch_str}]")

            if ema is not None:
                model.load_state_dict(saved)

            if mean_topk > best_topk:
                best_topk = mean_topk
                best_topk_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_topk.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {"best_topk": best_topk, "best_topk_epoch": best_topk_epoch},
                    global_step,
                )
                print(f"  [BEST_TOPK  E{epoch:03d}] "
                      f"TopK@{topk_tag}={best_topk:.4f} — saved")

            if val_focal < best_focal:
                best_focal = val_focal
                best_focal_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_focal.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {"best_focal": best_focal, "best_focal_epoch": best_focal_epoch},
                    global_step,
                )
                print(f"  [BEST_FOCAL E{epoch:03d}] "
                      f"val_focal={best_focal:.5f} — saved")

            if ready and mean_topk > best_ldm:
                best_ldm = mean_topk
                best_ldm_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_ldm.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {
                        "best_ldm": best_ldm,
                        "best_ldm_epoch": best_ldm_epoch,
                        "latent_state": {
                            "ch_std_mean": mean_ch_std,
                            "ch_std_min": min_ch_std,
                            "clamp_frac_low": frac_low,
                        }
                    },
                    global_step,
                )
                print(f"  [BEST_LDM   E{epoch:03d}] "
                      f"TopK@{topk_tag}={best_ldm:.4f} "
                      f"ch_std_min={min_ch_std:.4f} "
                      f"clamp_low={frac_low:.3f} — saved best_ldm.pt")

            with open(os.path.join(cfg.out_dir, "eval_last.json"), "w") as f:
                json.dump({
                    "epoch": epoch,
                    "beta": beta,
                    "val_focal": val_focal,
                    "val_kl": val_kl,
                    "mae": float(np.mean(mae_list)),
                    "topk": mean_topk,
                    "topk_pct": topk_tag,
                    "latent": {
                        "ch_std_mean": mean_ch_std,
                        "ch_std_min": min_ch_std,
                        "per_channel": ch_stds_avg.tolist(),
                        "clamp_frac_low": frac_low,
                        "clamp_frac_high": frac_high,
                        "ldm_ready": ready,
                    },
                    "best": {
                        "topk": {"val": best_topk, "epoch": best_topk_epoch},
                        "focal": {"val": best_focal, "epoch": best_focal_epoch},
                        "ldm": {"val": best_ldm, "epoch": best_ldm_epoch},
                    },
                }, f, indent=2)
            print()

        save_ckpt(
            os.path.join(cfg.out_dir, "latest.pt"),
            model, ema, optimizer, epoch, cfg,
            {
                "best_topk": best_topk,
                "best_focal": best_focal,
                "best_ldm": best_ldm,
            },
            global_step,
        )

    print(f"\n[DONE] best_topk={best_topk:.4f} @ epoch {best_topk_epoch}")
    print(f"[DONE] best_focal={best_focal:.5f} @ epoch {best_focal_epoch}")

    if best_ldm > -1.0:
        handoff_ckpt = os.path.join(cfg.out_dir, "best_ldm.pt")
        print(f"[DONE] best_ldm={best_ldm:.4f} @ epoch {best_ldm_epoch}")
        print(f"[DONE] Recommended LDM handoff: {handoff_ckpt}")
    else:
        handoff_ckpt = os.path.join(cfg.out_dir, "best_topk.pt")
        print("[DONE] best_ldm: never achieved LDM-ready latent.")
        print("[DONE] Falling back to best_topk.pt for downstream experiments.")
        print("[DONE] This fallback is practical, but not latent-gated.")

    with open(os.path.join(cfg.out_dir, "ldm_handoff.txt"), "w") as f:
        f.write(handoff_ckpt + "\n")

    print(f"\n[DONE] Next:")
    print(f"  python vae_latent_stats.py --ckpt {handoff_ckpt}")
    print(f"  # then evaluate / hand off that checkpoint")


if __name__ == "__main__":
    main()