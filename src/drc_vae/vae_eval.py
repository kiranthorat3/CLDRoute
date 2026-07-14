#!/usr/bin/env python3
"""
vae_eval.py — Evaluate trained LabelVAE v2 on test split.

Reports:
  - Reconstruction: MAE, SSIM, NRMSE
  - Hotspot: Recall@1%, topk overlap
  - Latent regularity: mu_mean, mu_std, KL — tells you if diffusion will work
  - Trivial baseline (predict all zeros) for comparison

Usage:
  python vae_eval.py --ckpt ./runs/vae_v2_DRC_N28/best.pt
  python vae_eval.py --ckpt ./runs/vae_v2_DRC_N28/best.pt --split val
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from vae_config import VAEConfig
from vae_model  import LabelVAE, free_bits_kl
from vae_data   import LabelDataset, _collate


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics(gt: np.ndarray, pr: np.ndarray, hotspot_q: float = 0.99):
    mae  = float(np.mean(np.abs(gt - pr)))
    rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
    rng  = float(gt.max()) - float(gt.min())
    nrms = (rmse / rng) if rng > 1e-8 else float("nan")
    ssim = float(structural_similarity(gt, pr, data_range=1.0))

    # Hotspot recall — what fraction of top-q% GT pixels are predicted in top-q%
    thr_gt = float(np.quantile(gt, hotspot_q))
    thr_pr = float(np.quantile(pr, hotspot_q))
    gt_hot = gt >= thr_gt
    pr_hot = pr >= thr_pr
    recall = float(np.sum(gt_hot & pr_hot) / max(np.sum(gt_hot), 1))

    # topk overlap (stricter — exact top-k pixels)
    k      = max(1, int((1 - hotspot_q) * gt.size))
    gt_idx = np.argsort(gt.ravel())[-k:]
    pr_idx = np.argsort(pr.ravel())[-k:]
    topk   = float(len(np.intersect1d(gt_idx, pr_idx)) / k)

    return dict(mae=mae, nrms=nrms, ssim=ssim,
                hotspot_recall=recall, topk=topk)


def trivial(gt: np.ndarray):
    return metrics(gt, np.zeros_like(gt))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser("LabelVAE v2 evaluation")
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--hotspot_q",   type=float, default=0.99)
    args = p.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ck = torch.load(args.ckpt, map_location=device)
    assert ck.get("model_type") == "LabelVAE_v2", (
        f"Wrong checkpoint type: {ck.get('model_type')}. Expected LabelVAE_v2."
    )

    model = LabelVAE(
        C_label   = int(ck["C_label"]),
        latent_ch = int(ck["latent_ch"]),
        base_ch   = int(ck.get("base_ch", 64)),
        log_scale = float(ck.get("log_scale", 10.0)),
    ).to(device)
    model.load_state_dict(ck["net"], strict=True)

    # Apply EMA weights if available
    if "ema" in ck:
        from vae_train import EMA
        ema = EMA(model)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model)
        print("[Eval] EMA weights applied")

    model.eval()
    print(
        f"[Eval] ckpt epoch={ck['epoch']} | "
        f"task={ck.get('task','?')} | "
        f"latent={ck['latent_ch']}×64×64 | "
        f"beta_target={ck.get('beta_target','?')}"
    )

    # ── Load config and data ──────────────────────────────────────────────────
    cfg      = VAEConfig.load(os.path.join(ckpt_dir, "config.json"))
    csv_path = cfg.csv_test if args.split == "test" else cfg.csv_val

    ds = LabelDataset(csv_path, cfg.label_dir, split=args.split, verify=False)
    loader = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        collate_fn  = _collate,
        drop_last   = False,
    )
    print(f"[Eval] split={args.split} | n={len(ds)} | hotspot_q={args.hotspot_q}")

    # ── Collect metrics ───────────────────────────────────────────────────────
    # TopK@1% is the primary metric for DRC hotspot quality.
    # SSIM and MAE are reported as secondary metrics for comparison with baselines.
    mae_l, ssim_l, nrms_l, topk_l   = [], [], [], []
    triv_mae_l, triv_topk_l         = [], []
    kl_vals, mu_means, mu_stds      = [], [], []

    with torch.no_grad():
        for lbl, _ in loader:
            lbl = lbl.float().to(device)
            recon, mu, logvar, z = model(lbl, sample=False)

            kl_vals.append(float(free_bits_kl(mu, logvar).item()))
            mu_means.append(float(mu.mean()))
            mu_stds.append(float(mu.std()))

            gt_np = lbl.cpu().numpy()[:, 0]
            pr_np = recon.cpu().numpy()[:, 0]
            for i in range(gt_np.shape[0]):
                m  = metrics(gt_np[i], pr_np[i], args.hotspot_q)
                tv = trivial(gt_np[i])
                mae_l.append(m["mae"])
                ssim_l.append(m["ssim"])
                nrms_l.append(m["nrms"])
                topk_l.append(m["topk"])
                triv_mae_l.append(tv["mae"])
                triv_topk_l.append(tv["topk"])

    # ── Print results ─────────────────────────────────────────────────────────
    mu_mean_agg = float(np.mean(mu_means))
    mu_std_agg  = float(np.mean(mu_stds))
    #latent_ok   = abs(mu_mean_agg) < 0.2 and mu_std_agg > 0.5
    latent_ok = mu_std_agg > 0.45

    print("\n" + "=" * 60)
    print(f"  VAE v2 EVAL — {ck.get('task','?')} / {args.split}")
    print("=" * 60)
    print(f"\n  Primary metric (hotspot quality):")
    print(f"  {'TopK@1% ↑':<22} {np.mean(topk_l):>10.4f}  {np.mean(triv_topk_l):>10.4f}  (trivial)")
    print(f"\n  Secondary metrics (for baseline comparison):")
    print(f"  {'MAE ↓':<22} {np.mean(mae_l):>10.5f}  {np.mean(triv_mae_l):>10.5f}  (trivial)")
    print(f"  {'NRMSE ↓':<22} {np.nanmean(nrms_l):>10.4f}  {'—':>10}")
    print(f"  {'SSIM ↑':<22} {np.mean(ssim_l):>10.4f}  {'—':>10}")

    print(f"\n  Latent regularity (diffusion readiness):")
    print(f"    mu_mean = {mu_mean_agg:.4f}  (target: ~0)")
    print(f"    mu_std  = {mu_std_agg:.4f}  (target: ~1)")
    print(f"    mean KL = {np.mean(kl_vals):.4f}")
   
    print(f"    Status  : {'✓ latent spread acceptable' if latent_ok else '✗ run vae_latent_stats.py to verify'}")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = os.path.join(ckpt_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "ckpt":  args.ckpt,
        "epoch": ck["epoch"],
        "split": args.split,
        "n":     len(mae_l),
        "metrics": {
            "topk_1pct": float(np.mean(topk_l)),   # primary
            "mae":       float(np.mean(mae_l)),
            "nrms":      float(np.nanmean(nrms_l)),
            "ssim":      float(np.mean(ssim_l)),
        },
        "trivial": {
            "topk_1pct": float(np.mean(triv_topk_l)),
            "mae":       float(np.mean(triv_mae_l)),
        },
        "latent": {
            "mu_mean": mu_mean_agg,
            "mu_std":  mu_std_agg,
            "mean_kl": float(np.mean(kl_vals)),
            "ready":   latent_ok,
        },
    }
    out_path = os.path.join(out_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()