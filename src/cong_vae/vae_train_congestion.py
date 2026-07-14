#!/usr/bin/env python3
"""
vae_train_congestion.py — Training for CongestionVAE (expanded dataset).

Loss: L1 reconstruction + beta * KL (with free-bits per channel)

Checkpoints:
  best_l1.pt   — lowest val L1
  best_ssim.pt — highest val SSIM
  best_ldm.pt  — lowest val L1 among latent-ready epochs — USE FOR LDM

Usage:
  python vae_train_congestion.py \
    --out_dir ./runs/vae_Cong_expanded_v2 \
    --beta_target 0.005 \
    --epochs 150
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
from skimage.metrics import structural_similarity
from vae_config_congestion import CongestionVAEConfig
from vae_model_congestion  import CongestionVAE, recon_loss_l1, kl_loss
from vae_data_congestion   import make_loaders


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().float()
                       for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(
                    v.float(), alpha=1 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device)
             for k, v in self.shadow.items()})

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, s):
        self.decay  = float(s.get("decay", self.decay))
        self.shadow = {k: v.clone().float()
                       for k, v in s.get("shadow", {}).items()}


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t   = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


# ─────────────────────────────────────────────────────────────────────────────
# Latent readiness gate
# ─────────────────────────────────────────────────────────────────────────────
def latent_ready(min_ch_std, mean_ch_std, clamp_frac_low, cfg):
    return (min_ch_std     > cfg.ldm_min_ch_std and
            mean_ch_std    > cfg.ldm_mean_ch_std and
            clamp_frac_low < cfg.ldm_max_clamp_frac)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def save_ckpt(path, model, ema, optimizer, epoch, cfg, meta, global_step):
    ck = dict(
        model_type  = "CongestionVAE_v2",
        C_label     = cfg.C_label,
        latent_ch   = cfg.latent_ch,
        base_ch     = cfg.base_ch,
        logvar_min  = cfg.logvar_min,
        logvar_max  = cfg.logvar_max,
        task        = cfg.task,
        tech        = cfg.tech,
        net         = model.state_dict(),
        opt         = optimizer.state_dict(),
        epoch       = epoch,
        global_step = global_step,
        beta_target = cfg.beta_target,
        free_bits   = cfg.free_bits,
        recon_loss  = cfg.recon_loss,
    )
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
    p = argparse.ArgumentParser("CongestionVAE v2 training")
    p.add_argument("--out_dir",     default=None)
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--beta_target", type=float, default=None)
    p.add_argument("--free_bits",   type=float, default=None)
    p.add_argument("--latent_ch",   type=int,   default=None)
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--eval_every",  type=int,   default=None)
    p.add_argument("--label_dir",   default=None)
    p.add_argument("--csv_train",   default=None)
    p.add_argument("--csv_val",     default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_args()
    cfg  = CongestionVAEConfig()
    for attr in ["out_dir", "epochs", "batch_size", "lr", "beta_target",
                 "free_bits", "latent_ch", "seed", "eval_every",
                 "label_dir", "csv_train", "csv_val"]:
        val = getattr(args, attr)
        if val is not None:
            setattr(cfg, attr, val)

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train, ds_val, loader_train, loader_val = make_loaders(cfg)

    model = CongestionVAE(
        C_label    = cfg.C_label,
        latent_ch  = cfg.latent_ch,
        base_ch    = cfg.base_ch,
        logvar_min = cfg.logvar_min,
        logvar_max = cfg.logvar_max,
    ).to(device)

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW(model.parameters(), lr=cfg.lr,
                      weight_decay=cfg.weight_decay)
    ema       = EMA(model, decay=cfg.ema_decay) if cfg.ema else None

    steps_per_epoch = len(loader_train)
    total_steps     = cfg.epochs * steps_per_epoch

    print("=" * 65)
    print(f"  CongestionVAE v2 | task={cfg.task} | device={device}")
    print(f"  train={len(ds_train)} | val={len(ds_val)}")
    print(f"  arch: {cfg.H}×{cfg.W} → latent {cfg.latent_ch}×64×64")
    print(f"  params: {n_params/1e6:.2f}M")
    print(f"  recon: {cfg.recon_loss.upper()} | "
          f"KL: beta 0→{cfg.beta_target} over {cfg.beta_warmup_epochs} epochs")
    print(f"  free_bits={cfg.free_bits} nats/channel "
          f"(prevents channel collapse)")
    print(f"  logvar clamp: ({cfg.logvar_min}, {cfg.logvar_max})")
    print(f"  LDM gate: min_ch_std>{cfg.ldm_min_ch_std} "
          f"mean_ch_std>{cfg.ldm_mean_ch_std} "
          f"clamp_frac<{cfg.ldm_max_clamp_frac}")
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

    best_l1         = float("inf")
    best_ssim       = float("-inf")
    best_ldm_l1     = float("inf")
    best_l1_epoch   = -1
    best_ssim_epoch = -1
    best_ldm_epoch  = -1
    global_step     = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        beta      = cfg.beta_at_epoch(epoch)
        sum_l1    = sum_kl = n_batches = 0
        t0        = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for lbl, _ in loader_train:
            lbl    = lbl.float().to(device)
            lr_now = get_lr(global_step, cfg.warmup_steps,
                            total_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            recon, mu, logvar, z = model(lbl, sample=True)
            loss_l1 = recon_loss_l1(recon, lbl)
            loss_kl = kl_loss(mu, logvar,
                               cfg.logvar_min, cfg.logvar_max,
                               free_bits=cfg.free_bits)
            loss    = loss_l1 + beta * loss_kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            global_step += 1
            if ema is not None:
                ema.update(model)

            sum_l1    += loss_l1.item()
            sum_kl    += loss_kl.item()
            n_batches += 1

        avg_l1  = sum_l1 / max(n_batches, 1)
        avg_kl  = sum_kl / max(n_batches, 1)
        elapsed = time.time() - t0
        mem_gb  = (torch.cuda.max_memory_allocated() / 1e9
                   if torch.cuda.is_available() else 0)
        lr_now  = get_lr(global_step, cfg.warmup_steps, total_steps, cfg.lr)
        print(f"[E{epoch:03d}] l1={avg_l1:.5f} kl={avg_kl:.4f} "
              f"beta={beta:.4f} lr={lr_now:.2e} "
              f"time={elapsed:.0f}s mem={mem_gb:.1f}GB")

        # ── Evaluation ────────────────────────────────────────────────────────
        if epoch % cfg.eval_every == 0:
            model.eval()
            if ema is not None:
                saved = {k: v.clone()
                         for k, v in model.state_dict().items()}
                ema.copy_to(model)

            val_l1     = val_kl = val_n = 0.0
            ssim_list  = []
            mae_list   = []
            ch_mu_stds = []
            logvar_all = []

            with torch.no_grad():
                for lbl, _ in loader_val:
                    lbl = lbl.float().to(device)
                    recon, mu, logvar, z = model(lbl, sample=False)
                    val_l1 += recon_loss_l1(
                        recon, lbl).item() * lbl.shape[0]
                    val_kl += kl_loss(
                        mu, logvar,
                        cfg.logvar_min, cfg.logvar_max,
                        free_bits=cfg.free_bits).item() * lbl.shape[0]
                    val_n  += lbl.shape[0]
                    ch_mu_stds.append(mu.std(dim=(0,2,3)).cpu().numpy())
                    logvar_all.append(logvar.detach().cpu().float())
                    gt_np = lbl.cpu().numpy()[:,0]
                    pr_np = recon.cpu().numpy()[:,0]
                    for i in range(gt_np.shape[0]):
                        mae_list.append(
                            float(np.mean(np.abs(gt_np[i] - pr_np[i]))))
                        ssim_list.append(float(
                            structural_similarity(
                                gt_np[i], pr_np[i], data_range=1.0)))

            val_l1  /= max(val_n, 1)
            val_kl  /= max(val_n, 1)
            val_ssim = float(np.mean(ssim_list))
            val_mae  = float(np.mean(mae_list))

            ch_stds_avg = np.mean(ch_mu_stds, axis=0)
            mean_ch_std = float(np.mean(ch_stds_avg))
            min_ch_std  = float(np.min(ch_stds_avg))

            lv_cat     = torch.cat([lv.reshape(-1) for lv in logvar_all])
            frac_low   = float((lv_cat <= cfg.logvar_min + 1e-4).float().mean())
            frac_high  = float((lv_cat >= cfg.logvar_max - 1e-4).float().mean())

            ready = latent_ready(min_ch_std, mean_ch_std, frac_low, cfg)

            print(f"  [EVAL  E{epoch:03d}] "
                  f"val_l1={val_l1:.5f} val_kl={val_kl:.4f} | "
                  f"MAE={val_mae:.5f} SSIM={val_ssim:.4f}")
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

            if val_l1 < best_l1:
                best_l1       = val_l1
                best_l1_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_l1.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {"best_l1": best_l1, "best_l1_epoch": best_l1_epoch},
                    global_step)
                print(f"  [BEST_L1    E{epoch:03d}] "
                      f"val_l1={best_l1:.5f} — saved (reference only)")

            if val_ssim > best_ssim:
                best_ssim       = val_ssim
                best_ssim_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_ssim.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {"best_ssim":       best_ssim,
                     "best_ssim_epoch": best_ssim_epoch},
                    global_step)
                print(f"  [BEST_SSIM  E{epoch:03d}] "
                      f"val_ssim={best_ssim:.4f} — saved (reference only)")

            if ready and val_l1 < best_ldm_l1:
                best_ldm_l1    = val_l1
                best_ldm_epoch = epoch
                save_ckpt(
                    os.path.join(cfg.out_dir, "best_ldm.pt"),
                    model, ema, optimizer, epoch, cfg,
                    {"best_ldm_l1":    best_ldm_l1,
                     "best_ldm_epoch": best_ldm_epoch,
                     "latent_state": {
                         "ch_std_mean":    mean_ch_std,
                         "ch_std_min":     min_ch_std,
                         "clamp_frac_low": frac_low,
                     }},
                    global_step)
                print(f"  [BEST_LDM   E{epoch:03d}] "
                      f"L1={best_ldm_l1:.5f} "
                      f"ch_std_min={min_ch_std:.4f} — saved best_ldm.pt ← USE FOR LDM")

            with open(os.path.join(cfg.out_dir, "eval_last.json"), "w") as f:
                json.dump({
                    "epoch":    epoch,
                    "beta":     beta,
                    "val_l1":   val_l1,
                    "val_kl":   val_kl,
                    "val_mae":  val_mae,
                    "val_ssim": val_ssim,
                    "latent": {
                        "ch_std_mean":     mean_ch_std,
                        "ch_std_min":      min_ch_std,
                        "per_channel":     ch_stds_avg.tolist(),
                        "clamp_frac_low":  frac_low,
                        "clamp_frac_high": frac_high,
                        "ldm_ready":       ready,
                    },
                    "best": {
                        "l1":   {"val": best_l1,      "epoch": best_l1_epoch},
                        "ssim": {"val": best_ssim,    "epoch": best_ssim_epoch},
                        "ldm":  {"val": best_ldm_l1,  "epoch": best_ldm_epoch},
                    },
                }, f, indent=2)
            print()

        save_ckpt(
            os.path.join(cfg.out_dir, "latest.pt"),
            model, ema, optimizer, epoch, cfg,
            {"best_l1":     best_l1,
             "best_ssim":   best_ssim,
             "best_ldm_l1": best_ldm_l1},
            global_step)

    print(f"\n[DONE] best_l1={best_l1:.5f} @ epoch {best_l1_epoch} "
          f"(reference only)")
    print(f"[DONE] best_ssim={best_ssim:.4f} @ epoch {best_ssim_epoch} "
          f"(reference only)")
    if best_ldm_l1 < float("inf"):
        print(f"[DONE] best_ldm L1={best_ldm_l1:.5f} "
              f"@ epoch {best_ldm_epoch}  ← USE THIS FOR LDM")
    else:
        print(f"[DONE] best_ldm: no epoch passed readiness gate.")
        print(f"[DONE] Lower ldm_min_ch_std threshold or reduce beta further.")
    print(f"\n[DONE] Next steps:")
    print(f"  python vae_eval_congestion.py "
          f"--ckpt {cfg.out_dir}/best_ldm.pt")
    print(f"  python vae_latent_stats_congestion.py "
          f"--ckpt {cfg.out_dir}/best_ldm.pt")


if __name__ == "__main__":
    main()