#!/usr/bin/env python3
"""
vae_latent_stats.py — Compute per-channel latent statistics for diffusion training.

MUST be run after VAE training, before starting the latent diffusion trainer.
The diffusion trainer reads latent_stats.json to normalize latents per channel.

Why per-channel normalization:
  Standard LDM practice (Rombach et al.) normalizes each latent channel
  independently. A global scalar mean/std fails if channels have different
  scales — e.g., channel 3 mean=2.0 and channel 7 mean=-1.0 would both
  be poorly normalized by a single global value.

Why train set only:
  Normalization statistics must be computed on training data only.
  Applying val/test statistics would leak information and is non-standard.

Saves:
  {ckpt_dir}/latent_stats.json
    z_mean: list of per-channel means  (length = latent_ch)
    z_std:  list of per-channel stds   (length = latent_ch)

Usage:
  python vae_latent_stats.py --ckpt ./runs/vae_v2_DRC_N28/best.pt
"""

import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from vae_config import VAEConfig
from vae_model  import LabelVAE
from vae_data   import LabelDataset, _collate
from vae_train  import EMA


def main():
    p = argparse.ArgumentParser("Compute per-channel latent stats for diffusion trainer")
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))

    # ── Load model ────────────────────────────────────────────────────────────
    ck = torch.load(args.ckpt, map_location=device)
    assert ck.get("model_type") == "LabelVAE_v2", \
        f"Expected LabelVAE_v2, got {ck.get('model_type')}"

    model = LabelVAE(
        C_label   = int(ck["C_label"]),
        latent_ch = int(ck["latent_ch"]),
        base_ch   = int(ck.get("base_ch", 64)),
        log_scale = float(ck.get("log_scale", 10.0)),
    ).to(device)
    model.load_state_dict(ck["net"], strict=True)

    # Apply EMA weights — same weights used at eval time
    if "ema" in ck:
        ema = EMA(model)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model)
        print(f"[LatentStats] EMA weights applied")

    model.eval()
    print(f"[LatentStats] epoch={ck['epoch']} | task={ck.get('task','?')} | "
          f"latent_ch={ck['latent_ch']}")

    # ── Load training data — stats always on train set ────────────────────────
    cfg = VAEConfig.load(os.path.join(ckpt_dir, "config.json"))
    ds  = LabelDataset(cfg.csv_train, cfg.label_dir, split="train", verify=False)
    loader = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        collate_fn  = _collate,
        drop_last   = False,
    )
    print(f"[LatentStats] Computing over {len(ds)} training samples...")

    # ── Collect all latents ───────────────────────────────────────────────────
    all_z = []
    with torch.no_grad():
        for lbl, _ in loader:
            z = model.encode_to_z(lbl.float().to(device))   # returns mu, no noise
            all_z.append(z.cpu())

    all_z = torch.cat(all_z, dim=0)   # (N, latent_ch, 64, 64)
    N, C, H, W = all_z.shape

    # Per-channel stats — reshape each channel to (N*H*W,)
    z_mean = []
    z_std  = []
    print(f"\n[LatentStats] Per-channel statistics (N={N}, H={H}, W={W}):")
    print(f"  {'Ch':<6} {'Mean':>10} {'Std':>10}  Status")
    print(f"  {'-'*40}")
    for c in range(C):
        ch_vals = all_z[:, c, :, :].reshape(-1)
        m = float(ch_vals.mean())
        s = float(ch_vals.std())
        z_mean.append(m)
        z_std.append(s)
        status = "✓" if (abs(m) < 0.5 and 0.3 < s < 2.0) else "⚠"
        print(f"  {c:<6} {m:>10.4f} {s:>10.4f}  {status}")

    # Global summary
    global_mean = float(np.mean(z_mean))
    global_std  = float(np.mean(z_std))
    print(f"\n[LatentStats] Global mean={global_mean:.4f}  std={global_std:.4f}")

    if global_std < 0.45:
        print("[LatentStats] WARNING: global std < 0.45 — latent under-regularized.")
        print("              Diffusion samples from N(0,1) at inference → mismatch.")
        print("              Consider increasing beta_target and retraining.")
    elif global_std > 1.5:
        print("[LatentStats] WARNING: global std > 1.5 — latent over-dispersed.")
    else:
        print("[LatentStats] ✓ Global std in acceptable range [0.5, 1.5]")

    # ── Save ──────────────────────────────────────────────────────────────────
    stats = {
        "z_mean":      z_mean,        # list of length latent_ch
        "z_std":       z_std,         # list of length latent_ch
        "latent_ch":   C,
        "n_samples":   N,
        "ckpt_epoch":  int(ck["epoch"]),
        "task":        ck.get("task", "?"),
        "note": (
            "Per-channel normalization. Diffusion trainer normalizes each "
            "channel c as: z_norm[:,c] = (z[:,c] - z_mean[c]) / z_std[c]"
        ),
    }
    out_path = os.path.join(ckpt_dir, "latent_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[LatentStats] Saved → {out_path}")


if __name__ == "__main__":
    main()