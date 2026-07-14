#!/usr/bin/env python3
"""
vae_train.py — DRC VAE training (expanded dataset).

Changes from gen_auto/vae2/vae_train.py:
  - Cosine LR decay after warmup
  - Per-channel latent std for latent_ok (replaces batch mu.std())
  - Clamp fraction logging (fraction of logvar at lower/upper bound)
  - best_ldm.pt: best TopK@1% among epochs where latent is ready
  - Fixed redundant EMA copy/restore pair
  - beta_target=0.05 (unchanged from original — one change at a time)
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
#from vae_model  import LabelVAE, focal_recon_loss, hotspot_loss, free_bits_kl
from vae_model  import (LabelVAE, focal_recon_loss, hotspot_loss,
                        free_bits_kl, _LOGVAR_MIN, _LOGVAR_MAX)
from vae_data   import make_loaders

# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device)
             for k, v in self.shadow.items()}
        )

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, s):
        self.decay  = float(s.get("decay", self.decay))
        shadow      = s.get("shadow", {})
        self.shadow = {k: v.clone().float() for k, v in shadow.items()}

# ─────────────────────────────────────────────────────────────────────────────
# LR schedule — cosine decay after warmup
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t   = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)

# ─────────────────────────────────────────────────────────────────────────────
# Latent readiness — gating criterion for best_ldm.pt
# ─────────────────────────────────────────────────────────────────────────────
_LDM_MIN_CH_STD  = 0.30   # every channel must exceed this
_LDM_MEAN_CH_STD = 0.45   # mean across channels must exceed this
_LDM_MAX_CLAMP_FRAC = 0.30  # at most 30% of logvar elements at lower clamp

def latent_ready(min_ch_std, mean_ch_std, clamp_frac_low):
    """
    Returns True if the latent is ready for LDM training.
    All three conditions must hold:
      - Every channel has std > 0.30 (no dead channels)
      - Mean channel std > 0.45 (sufficient global spread)
      - <30% of logvar at lower clamp (clamp not dominating)
    """
    return (min_ch_std   > _LDM_MIN_CH_STD and
            mean_ch_std  > _LDM_MEAN_CH_STD and
            clamp_frac_low < _LDM_MAX_CLAMP_FRAC)

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def save_ckpt(path, model, ema, optimizer, epoch, cfg, meta, global_step):
    ck = {
        "model_type":     "LabelVAE_v2",
        "C_label":        cfg.C_label,
        "latent_ch":      cfg.latent_ch,
        "base_ch":        cfg.base_ch,
        "log_scale":      cfg.log_scale,
        "task":           cfg.task,
        "tech":           cfg.tech,
        "net":            model.state_dict(),
        "opt":            optimizer.state_dict(),
        "epoch":          epoch,
        "global_step":    global_step,
        "focal_gamma":    cfg.focal_gamma,
        "hotspot_weight": cfg.hotspot_weight,
        "free_bits":      cfg.free_bits,
        "beta_target":    cfg.beta_target,
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
    p = argparse.ArgumentParser("LabelVAE training — expanded DRC")
    p.add_argument("--task",           default=None)
    p.add_argument("--out_dir",        default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch_size",     type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--beta_target",    type=float, default=None)
    p.add_argument("--hotspot_weight", type=float, default=None)
    p.add_argument("--free_bits",      type=float, default=None)
    p.add_argument("--latent_ch",      type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=None)
    p.add_argument("--eval_every",     type=int,   default=None)
    p.add_argument("--label_dir",      default=None)
    p.add_argument("--csv_train",      default=None)
    p.add_argument("--csv_val",        default=None)
    p.add_argument("--csv_test",       default=None)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_args()
    cfg  = VAEConfig()
    for attr in ["task", "out_dir", "epochs", "batch_size", "lr",
                 "beta_target", "hotspot_weight", "free_bits", "latent_ch",
                 "seed", "eval_every", "label_dir",
                 "csv_train", "csv_val", "csv_test"]:
        val = getattr(args, attr)
        if val is not None:
            setattr(cfg, attr, val)

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train, ds_val, loader_train, loader_val = make_loaders(cfg)

    model = LabelVAE(
        C_label   = cfg.C_label,
        latent_ch = cfg.latent_ch,
        base_ch   = cfg.base_ch,
        log_scale = cfg.log_scale,
    ).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW(model.parameters(), lr=cfg.lr,
                      weight_decay=cfg.weight_decay)
    ema       = EMA(model, decay=cfg.ema_decay) if cfg.ema else None

    steps_per_epoch = len(loader_train)
    total_steps     = cfg.epochs * steps_per_epoch

    print("=" * 65)
    print(f"  LabelVAE | task={cfg.task} | device={device}")
    print(f"  train={len(ds_train)} | val={len(ds_val)}")
    print(f"  arch: {cfg.H}×{cfg.W} → latent {cfg.latent_ch}×64×64")
    print(f"  params: {n_params/1e6:.2f}M")
    print(f"  loss: focal(γ={cfg.focal_gamma}) + "
          f"{cfg.hotspot_weight}×hotspot@1%")
    print(f"  KL:   free_bits={cfg.free_bits} | beta 0→{cfg.beta_target} "
          f"over {cfg.beta_warmup_epochs} epochs")
    print(f"  LR:   warmup {cfg.warmup_steps} steps then cosine to "
          f"{cfg.lr*0.1:.2e}")
    print(f"  checkpoints: best_topk | best_focal | best_ldm (LDM-ready)")
    print(f"  LDM gate: min_ch_std>{_LDM_MIN_CH_STD} "
          f"mean_ch_std>{_LDM_MEAN_CH_STD} "
          f"clamp_frac<{_LDM_MAX_CLAMP_FRAC}")
    print("=" * 65)

    # Sanity
    lbl0, _ = next(iter(loader_train))
    lbl0    = lbl0.float().to(device)
    with torch.no_grad():
        r, mu, lv, z = model(lbl0, sample=True)
    print(f"[Sanity] input={tuple(lbl0.shape)} z={tuple(z.shape)} "
          f"recon=[{r.min():.3f},{r.max():.3f}]")
    print(f"[Sanity] mu: mean={mu.mean():.4f} std={mu.std():.4f} | "
          f"logvar: mean={lv.mean():.4f} min={lv.min():.4f}")
    print()

    best_topk        = -1.0
    best_focal       = float("inf")
    best_ldm         = -1.0     # best TopK among latent-ready epochs
    best_topk_epoch  = -1
    best_focal_epoch = -1
    best_ldm_epoch   = -1
    global_step      = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        beta = cfg.beta_at_epoch(epoch)
        sum_focal = sum_hot = sum_kl = n_batches = 0
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for lbl, _ in loader_train:
            lbl    = lbl.float().to(device)
            lr_now = get_lr(global_step, cfg.warmup_steps,
                            total_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            recon, mu, logvar, z = model(lbl, sample=True)
            loss_focal = focal_recon_loss(recon, lbl, gamma=cfg.focal_gamma)
            loss_hot   = hotspot_loss(recon, lbl, q=cfg.hotspot_q)
            loss_kl    = free_bits_kl(mu, logvar, free_bits=cfg.free_bits)
            loss       = (loss_focal
                          + cfg.hotspot_weight * loss_hot
                          + beta * loss_kl)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            global_step += 1
            if ema is not None:
                ema.update(model)

            sum_focal += loss_focal.item()
            sum_hot   += loss_hot.item()
            sum_kl    += loss_kl.item()
            n_batches += 1

        avg_focal = sum_focal / max(n_batches, 1)
        avg_hot   = sum_hot   / max(n_batches, 1)
        avg_kl    = sum_kl    / max(n_batches, 1)
        elapsed   = time.time() - t0
        mem_gb    = (torch.cuda.max_memory_allocated() / 1e9
                     if torch.cuda.is_available() else 0)
        lr_now    = get_lr(global_step, cfg.warmup_steps,
                           total_steps, cfg.lr)
        print(
            f"[E{epoch:03d}] focal={avg_focal:.5f} hot={avg_hot:.5f} "
            f"kl={avg_kl:.4f} beta={beta:.4f} "
            f"lr={lr_now:.2e} time={elapsed:.0f}s mem={mem_gb:.1f}GB"
        )

        # ── Evaluation ────────────────────────────────────────────────────────
        if epoch % cfg.eval_every == 0:
            model.eval()
            if ema is not None:
                saved = {k: v.clone()
                         for k, v in model.state_dict().items()}
                ema.copy_to(model)

            val_focal = val_kl = val_n = 0.0
            mae_list, topk_list = [], []
            ch_mu_stds       = []
            logvar_all        = []   # collect all logvar values for clamp frac

            with torch.no_grad():
                for lbl, _ in loader_val:
                    lbl = lbl.float().to(device)
                    recon, mu, logvar, z = model(lbl, sample=False)

                    val_focal += focal_recon_loss(
                        recon, lbl, gamma=cfg.focal_gamma,
                    ).item() * lbl.shape[0]
                    val_kl += free_bits_kl(
                        mu, logvar, free_bits=cfg.free_bits,
                    ).item() * lbl.shape[0]
                    val_n += lbl.shape[0]

                    # Per-channel std of mu
                    ch_mu_stds.append(
                        mu.std(dim=(0, 2, 3)).cpu().numpy()
                    )

                    # Collect logvar values for clamp fraction computation
                    logvar_all.append(logvar.detach().cpu().float())

                    gt_np = lbl.cpu().numpy()[:, 0]
                    pr_np = recon.cpu().numpy()[:, 0]
                    for i in range(gt_np.shape[0]):
                        mae_list.append(
                            float(np.mean(np.abs(gt_np[i] - pr_np[i]))))
                        k      = max(1, int((1 - cfg.hotspot_q)
                                            * gt_np[i].size))
                        gt_idx = np.argsort(gt_np[i].ravel())[-k:]
                        pr_idx = np.argsort(pr_np[i].ravel())[-k:]
                        topk_list.append(
                            float(len(np.intersect1d(gt_idx, pr_idx)) / k))

            val_focal /= max(val_n, 1)
            val_kl    /= max(val_n, 1)
            mean_topk  = float(np.mean(topk_list))

            # Per-channel std
            ch_stds_avg  = np.mean(ch_mu_stds, axis=0)
            mean_ch_std  = float(np.mean(ch_stds_avg))
            min_ch_std   = float(np.min(ch_stds_avg))

            # Clamp fractions — fraction of logvar elements at each bound
            lv_cat = torch.cat([lv.reshape(-1) for lv in logvar_all])
            from vae_model import _LOGVAR_MIN, _LOGVAR_MAX
            n_total      = lv_cat.numel()
            frac_low  = float((lv_cat <= _LOGVAR_MIN + 1e-4).float().mean())
            frac_high = float((lv_cat >= _LOGVAR_MAX - 1e-4).float().mean())

            # Latent readiness
            ready = latent_ready(min_ch_std, mean_ch_std, frac_low)

            print(
                f"  [EVAL  E{epoch:03d}] "
                f"val_focal={val_focal:.5f} val_kl={val_kl:.4f} | "
                f"MAE={np.mean(mae_list):.5f} TopK@1%={mean_topk:.4f}"
            )
            print(
                f"  [LATENT E{epoch:03d}] "
                f"ch_std_mean={mean_ch_std:.4f} "
                f"ch_std_min={min_ch_std:.4f} — "
                f"{'✓ LDM-ready' if ready else '✗ not LDM-ready'}"
            )
            print(
                f"  [LOGVAR E{epoch:03d}] "
                f"clamp_low={frac_low:.3f} clamp_high={frac_high:.3f} "
                f"({'clamp dominant' if frac_low > 0.3 else 'clamp OK'})"
            )
            ch_str = " ".join(f"{s:.2f}" for s in ch_stds_avg)
            print(f"  [CH_STD E{epoch:03d}] [{ch_str}]")

            # Restore raw weights before saving
            if ema is not None:
                model.load_state_dict(saved)

            # best_topk — best reconstruction regardless of latent state
            if mean_topk > best_topk:
                best_topk       = mean_topk
                best_topk_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_topk.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {"best_topk": best_topk,
                     "best_topk_epoch": best_topk_epoch},
                    global_step,
                )
                print(f"  [BEST_TOPK  E{epoch:03d}] "
                      f"TopK@1%={best_topk:.4f} — saved")

            # best_focal — best reconstruction loss regardless of latent state
            if val_focal < best_focal:
                best_focal       = val_focal
                best_focal_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_focal.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {"best_focal": best_focal,
                     "best_focal_epoch": best_focal_epoch},
                    global_step,
                )
                print(f"  [BEST_FOCAL E{epoch:03d}] "
                      f"val_focal={best_focal:.5f} — saved")

            # best_ldm — best TopK among LDM-ready epochs
            # This is the checkpoint to hand to the LDM trainer
            if ready and mean_topk > best_ldm:
                best_ldm       = mean_topk
                best_ldm_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_ldm.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {"best_ldm":       best_ldm,
                     "best_ldm_epoch": best_ldm_epoch,
                     "latent_state": {
                         "ch_std_mean":  mean_ch_std,
                         "ch_std_min":   min_ch_std,
                         "clamp_frac_low": frac_low,
                     }},
                    global_step,
                )
                print(f"  [BEST_LDM   E{epoch:03d}] "
                      f"TopK@1%={best_ldm:.4f} "
                      f"ch_std_min={min_ch_std:.4f} "
                      f"clamp_low={frac_low:.3f} — saved best_ldm.pt")

            with open(os.path.join(cfg.out_dir, "eval_last.json"), "w") as f:
                json.dump({
                    "epoch":     epoch,
                    "beta":      beta,
                    "val_focal": val_focal,
                    "val_kl":    val_kl,
                    "mae":       float(np.mean(mae_list)),
                    "topk_1pct": mean_topk,
                    "latent": {
                        "ch_std_mean":    mean_ch_std,
                        "ch_std_min":     min_ch_std,
                        "per_channel":    ch_stds_avg.tolist(),
                        "clamp_frac_low": frac_low,
                        "clamp_frac_high":frac_high,
                        "ldm_ready":      ready,
                    },
                    "best": {
                        "topk": {"val": best_topk,  "epoch": best_topk_epoch},
                        "focal":{"val": best_focal, "epoch": best_focal_epoch},
                        "ldm":  {"val": best_ldm,   "epoch": best_ldm_epoch},
                    },
                }, f, indent=2)

            # Apply EMA for next eval block if needed — but do NOT
            # redundantly copy then immediately restore
            # (removed the copy/restore pair that cancelled itself)
            print()

        save_ckpt(
            os.path.join(cfg.out_dir, "latest.pt"),
            model, ema, optimizer, epoch, cfg,
            {"best_topk": best_topk, "best_focal": best_focal,
             "best_ldm":  best_ldm},
            global_step,
        )

    print(f"\n[DONE] best_topk={best_topk:.4f} @ epoch {best_topk_epoch}")
    print(f"[DONE] best_focal={best_focal:.5f} @ epoch {best_focal_epoch}")
    if best_ldm > -1.0:
        print(f"[DONE] best_ldm={best_ldm:.4f} @ epoch {best_ldm_epoch}"
              f"  ← USE THIS FOR LDM")
    else:
        print(f"[DONE] best_ldm: never achieved LDM-ready latent. "
              f"Consider training longer or adjusting beta_target.")
    print(f"[DONE] Next: python vae_eval.py "
          f"--ckpt {cfg.out_dir}/best_ldm.pt")
    print(f"[DONE] Then: python vae_latent_stats.py "
          f"--ckpt {cfg.out_dir}/best_ldm.pt")

if __name__ == "__main__":
    main()