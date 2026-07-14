#!/usr/bin/env python3
"""
vae_latent_stats_congestion.py — Per-channel latent stats for CongestionVAE.
MUST be run after VAE training, before LDM training.
LDM trainer reads latent_stats.json to normalize latents per channel.

Usage:
  python vae_latent_stats_congestion.py \
    --ckpt ./runs/vae_Cong_expanded/best_ldm.pt
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from vae_config_congestion import CongestionVAEConfig
from vae_model_congestion  import CongestionVAE
from vae_data_congestion   import CongestionLabelDataset, _collate
from vae_train_congestion  import EMA


def main():
    p = argparse.ArgumentParser("CongestionVAE latent stats")
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))

    ck = torch.load(args.ckpt, map_location=device)
    assert ck.get("model_type") == "CongestionVAE_v2", \
        f"Expected CongestionVAE_v2, got {ck.get('model_type')}"

    model = CongestionVAE(
        C_label    = int(ck["C_label"]),
        latent_ch  = int(ck["latent_ch"]),
        base_ch    = int(ck.get("base_ch", 64)),
        logvar_min = float(ck.get("logvar_min", -4.0)),
        logvar_max = float(ck.get("logvar_max",  4.0)),
    ).to(device)
    model.load_state_dict(ck["net"], strict=True)

    if "ema" in ck:
        ema = EMA(model)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model)
        print("[LatentStats] EMA weights applied")
    model.eval()

    print(f"[LatentStats] epoch={ck['epoch']} | "
          f"latent_ch={ck['latent_ch']} | task={ck.get('task','?')}")

    cfg    = CongestionVAEConfig.load(os.path.join(ckpt_dir, "config.json"))
    ds     = CongestionLabelDataset(
        cfg.csv_train, cfg.label_dir, split="train", verify=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate, drop_last=False)

    print(f"[LatentStats] Computing over {len(ds)} training samples...")

    all_z = []
    with torch.no_grad():
        for lbl, _ in loader:
            z = model.encode_to_z(lbl.float().to(device))
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)   # (N, latent_ch, 64, 64)
    N, C, H, W = all_z.shape

    z_mean = []
    z_std  = []

    print(f"\n[LatentStats] Per-channel statistics (N={N}, H={H}, W={W}):")
    print(f"  {'Ch':<6} {'Mean':>10} {'Std':>10}  Status")
    print(f"  {'-'*42}")

    for c in range(C):
        ch_vals = all_z[:, c, :, :].reshape(-1)
        m = float(ch_vals.mean())
        s = float(ch_vals.std())
        z_mean.append(m)
        z_std.append(max(s, 1e-6))   # protect downstream division

        if s < 0.05:
            status = "⚠ very weak"
        elif s < 0.1:
            status = "◆ weak"
        elif s > 3.0:
            status = "◆ high"
        else:
            status = "✓"
        print(f"  {c:<6} {m:>10.4f} {s:>10.4f}  {status}")

    global_mean = float(np.mean(z_mean))
    global_std  = float(np.mean(z_std))

    print(f"\n[LatentStats] Global mean={global_mean:.4f}  "
          f"std={global_std:.4f}")
    print(f"[LatentStats] Diagnostic: global std={global_std:.4f}")
    if global_std < 0.3:
        print("[LatentStats]   ⚠ Very low — diffusion noise schedule "
              "likely mismatched.")
        print("                 Consider increasing beta_target.")
    elif global_std < 0.6:
        print("[LatentStats]   ◆ Moderate — per-channel normalization "
              "will correct this.")
        print("                 Monitor first LDM generation eval carefully.")
    elif global_std > 2.5:
        print("[LatentStats]   ◆ High — per-channel normalization will "
              "compress range.")
        print("                 Check that individual channels are consistent.")
    else:
        print("[LatentStats]   ✓ In typical range [0.6, 2.5]")

    stats = {
        "z_mean":      z_mean,
        "z_std":       z_std,
        "latent_ch":   C,
        "n_samples":   N,
        "ckpt_epoch":  int(ck["epoch"]),
        "task":        ck.get("task", "?"),
        "model_type":  "CongestionVAE_v2",
        "global_mean": global_mean,
        "global_std":  global_std,
        "note": (
            "Per-channel normalization. LDM normalizes each channel c as: "
            "z_norm[:,c] = (z[:,c] - z_mean[c]) / z_std[c]. "
            "z_std clamped to min 1e-6 to protect division."
        ),
    }
    out_path = os.path.join(ckpt_dir, "latent_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[LatentStats] Saved → {out_path}")

if __name__ == "__main__":
    main()