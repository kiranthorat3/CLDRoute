#!/usr/bin/env python3
"""
vae_eval.py — Evaluate trained LabelVAE on val or test split.

Why this version is stricter than the old one:
  - Default hotspot_q matches DRC training: 0.999 = top 0.1%
  - Reports both TopK@0.1% and TopK@1%
  - Latent readiness uses per-channel std + clamp fraction,
    not only a single global mu_std
  - Reads dataset paths from the checkpoint's config.json,
    so it works for both N14 and N28 without hardcoding

Usage:
  python vae_eval.py --ckpt ./runs/vae_DRC_xxx/best_ldm.pt
  python vae_eval.py --ckpt ./runs/vae_DRC_xxx/best_topk.pt --split val
"""

import os
import json
import argparse
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from vae_config import VAEConfig
from vae_model import LabelVAE, free_bits_kl, _LOGVAR_MIN, _LOGVAR_MAX
from vae_data import LabelDataset, _collate


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_NZ_THRESH = 0.01


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _topk_overlap(gt: np.ndarray, pr: np.ndarray, frac: float) -> float:
    k = max(1, int(frac * gt.size))
    gt_idx = np.argsort(gt.ravel())[-k:]
    pr_idx = np.argsort(pr.ravel())[-k:]
    return float(len(np.intersect1d(gt_idx, pr_idx)) / k)


def _hotspot_recall(gt: np.ndarray, pr: np.ndarray, q: float) -> float:
    thr_gt = float(np.quantile(gt, q))
    thr_pr = float(np.quantile(pr, q))
    gt_hot = gt >= thr_gt
    pr_hot = pr >= thr_pr
    return float(np.sum(gt_hot & pr_hot) / max(np.sum(gt_hot), 1))


def metrics(gt: np.ndarray, pr: np.ndarray, hotspot_q: float = 0.999) -> Dict[str, float]:
    gt = np.asarray(gt, dtype=np.float32)
    pr = np.asarray(pr, dtype=np.float32)

    mae  = float(np.mean(np.abs(gt - pr)))
    rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
    rng  = float(gt.max() - gt.min())
    nrms = (rmse / rng) if rng > 1e-8 else float("nan")
    ssim = float(structural_similarity(gt, pr, data_range=1.0))

    # Match training hotspot definition by default: q=0.999 → top 0.1%
    topk_q = _topk_overlap(gt, pr, frac=(1.0 - hotspot_q))
    rec_q  = _hotspot_recall(gt, pr, q=hotspot_q)

    # Also report top-1% for comparison with some LDM eval code
    topk_1pct = _topk_overlap(gt, pr, frac=0.01)

    gt_flat = gt.ravel()
    pr_flat = pr.ravel()
    nz = gt_flat > _NZ_THRESH
    nz_mae  = float(np.mean(np.abs(gt_flat[nz] - pr_flat[nz]))) if nz.any() else float("nan")
    nz_pear = _safe_pearson(gt_flat[nz], pr_flat[nz]) if np.sum(nz) >= 2 else float("nan")

    k_hot = max(1, int((1.0 - hotspot_q) * gt_flat.size))
    hot_idx = np.argsort(gt_flat)[-k_hot:]
    hotspot_mae = float(np.mean(np.abs(gt_flat[hot_idx] - pr_flat[hot_idx])))

    return {
        "mae": mae,
        "nrms": nrms,
        "ssim": ssim,
        "topk_at_q": topk_q,
        "recall_at_q": rec_q,
        "topk_1pct": topk_1pct,
        "nz_mae": nz_mae,
        "nz_pearson": nz_pear,
        "hotspot_mae": hotspot_mae,
    }


def trivial(gt: np.ndarray, hotspot_q: float = 0.999) -> Dict[str, float]:
    return metrics(gt, np.zeros_like(gt), hotspot_q=hotspot_q)


def latent_ready(min_ch_std: float, mean_ch_std: float, clamp_frac_low: float, cfg: VAEConfig) -> bool:
    return (
        min_ch_std > cfg.ldm_min_ch_std and
        mean_ch_std > cfg.ldm_mean_ch_std and
        clamp_frac_low < cfg.ldm_max_clamp_frac
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser("LabelVAE evaluation")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--hotspot_q", type=float, default=0.999,
                   help="Default 0.999 = top 0.1%%, matching training")
    args = p.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))

    # ── Load checkpoint ───────────────────────────────────────────────────────
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
        f"tech={ck.get('tech','?')} | "
        f"latent={ck['latent_ch']}×64×64 | "
        f"beta_target={ck.get('beta_target','?')}"
    )

    # ── Load config and data ─────────────────────────────────────────────────
    cfg      = VAEConfig.load(os.path.join(ckpt_dir, "config.json"))
    csv_path = cfg.csv_test if args.split == "test" else cfg.csv_val

    ds = LabelDataset(csv_path, cfg.label_dir, split=args.split, verify=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )
    print(
        f"[Eval] split={args.split} | n={len(ds)} | "
        f"hotspot_q={args.hotspot_q} (top {(1.0 - args.hotspot_q)*100:.3f}%)"
    )

    # ── Collect metrics ───────────────────────────────────────────────────────
    m_list: List[Dict[str, float]] = []
    t_list: List[Dict[str, float]] = []

    kl_vals     = []
    ch_mu_stds  = []
    logvar_all  = []

    with torch.no_grad():
        for lbl, _ in loader:
            lbl = lbl.float().to(device)
            recon, mu, logvar, z = model(lbl, sample=False)

            kl_vals.append(
                float(free_bits_kl(
                    mu, logvar,
                    free_bits=float(ck.get("free_bits", 0.5))
                ).item())
            )
            ch_mu_stds.append(mu.std(dim=(0, 2, 3)).cpu().numpy())
            logvar_all.append(logvar.detach().cpu().float())

            gt_np = lbl.cpu().numpy()[:, 0]
            pr_np = recon.cpu().numpy()[:, 0]

            for i in range(gt_np.shape[0]):
                m_list.append(metrics(gt_np[i], pr_np[i], hotspot_q=args.hotspot_q))
                t_list.append(trivial(gt_np[i], hotspot_q=args.hotspot_q))

    # ── Latent diagnostics ────────────────────────────────────────────────────
    ch_stds_avg = np.mean(ch_mu_stds, axis=0)
    mean_ch_std = float(np.mean(ch_stds_avg))
    min_ch_std  = float(np.min(ch_stds_avg))

    lv_cat    = torch.cat([lv.reshape(-1) for lv in logvar_all])
    frac_low  = float((lv_cat <= _LOGVAR_MIN + 1e-4).float().mean())
    frac_high = float((lv_cat >= _LOGVAR_MAX - 1e-4).float().mean())

    ready = latent_ready(min_ch_std, mean_ch_std, frac_low, cfg)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def mean_key(lst, key):
        vals = [x[key] for x in lst if not np.isnan(x[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 68)
    print(f"  LabelVAE EVAL — {ck.get('task','?')} / {args.split}")
    print("=" * 68)

    print("\n  Primary hotspot metrics:")
    print(f"  {'TopK@q ↑':<24} {mean_key(m_list, 'topk_at_q'):>10.4f}  {mean_key(t_list, 'topk_at_q'):>10.4f}  (trivial)")
    print(f"  {'Recall@q ↑':<24} {mean_key(m_list, 'recall_at_q'):>10.4f}  {mean_key(t_list, 'recall_at_q'):>10.4f}  (trivial)")
    print(f"  {'TopK@1% ↑':<24} {mean_key(m_list, 'topk_1pct'):>10.4f}  {mean_key(t_list, 'topk_1pct'):>10.4f}  (trivial)")

    print("\n  Secondary reconstruction metrics:")
    print(f"  {'MAE ↓':<24} {mean_key(m_list, 'mae'):>10.5f}  {mean_key(t_list, 'mae'):>10.5f}  (trivial)")
    print(f"  {'NRMSE ↓':<24} {mean_key(m_list, 'nrms'):>10.4f}")
    print(f"  {'SSIM ↑':<24} {mean_key(m_list, 'ssim'):>10.4f}")
    print(f"  {'NZ-MAE ↓':<24} {mean_key(m_list, 'nz_mae'):>10.5f}")
    print(f"  {'NZ-Pearson ↑':<24} {mean_key(m_list, 'nz_pearson'):>10.4f}")
    print(f"  {'Hotspot-MAE ↓':<24} {mean_key(m_list, 'hotspot_mae'):>10.5f}")

    print("\n  Latent regularity:")
    print(f"    mean KL            = {float(np.mean(kl_vals)):.4f}")
    print(f"    ch_std_mean        = {mean_ch_std:.4f}")
    print(f"    ch_std_min         = {min_ch_std:.4f}")
    print(f"    clamp_frac_low     = {frac_low:.4f}")
    print(f"    clamp_frac_high    = {frac_high:.4f}")
    print(f"    readiness gate     = {'✓ LDM-ready' if ready else '✗ not LDM-ready'}")
    print(f"    thresholds         = min_ch_std>{cfg.ldm_min_ch_std}, "
          f"mean_ch_std>{cfg.ldm_mean_ch_std}, clamp_low<{cfg.ldm_max_clamp_frac}")
    print(f"    per-channel std    = [{' '.join(f'{x:.2f}' for x in ch_stds_avg)}]")

    print("=" * 68)

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = os.path.join(ckpt_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "ckpt": args.ckpt,
        "epoch": int(ck["epoch"]),
        "split": args.split,
        "n": len(m_list),
        "hotspot_q": float(args.hotspot_q),
        "metrics": {
            "topk_at_q":    mean_key(m_list, "topk_at_q"),
            "recall_at_q":  mean_key(m_list, "recall_at_q"),
            "topk_1pct":    mean_key(m_list, "topk_1pct"),
            "mae":          mean_key(m_list, "mae"),
            "nrms":         mean_key(m_list, "nrms"),
            "ssim":         mean_key(m_list, "ssim"),
            "nz_mae":       mean_key(m_list, "nz_mae"),
            "nz_pearson":   mean_key(m_list, "nz_pearson"),
            "hotspot_mae":  mean_key(m_list, "hotspot_mae"),
        },
        "trivial": {
            "topk_at_q":   mean_key(t_list, "topk_at_q"),
            "recall_at_q": mean_key(t_list, "recall_at_q"),
            "topk_1pct":   mean_key(t_list, "topk_1pct"),
            "mae":         mean_key(t_list, "mae"),
        },
        "latent": {
            "mean_kl":         float(np.mean(kl_vals)),
            "per_channel_std": ch_stds_avg.tolist(),
            "ch_std_mean":     mean_ch_std,
            "ch_std_min":      min_ch_std,
            "clamp_frac_low":  frac_low,
            "clamp_frac_high": frac_high,
            "ready":           ready,
            "thresholds": {
                "ldm_min_ch_std":     float(cfg.ldm_min_ch_std),
                "ldm_mean_ch_std":    float(cfg.ldm_mean_ch_std),
                "ldm_max_clamp_frac": float(cfg.ldm_max_clamp_frac),
            },
        },
    }

    out_path = os.path.join(out_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()