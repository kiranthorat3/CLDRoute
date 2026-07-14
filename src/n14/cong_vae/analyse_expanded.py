#!/usr/bin/env python3
"""
analyse_expanded.py — Fast full-dataset analysis for expanded-feature CircuitNet-N14.

Kept useful for VAE design, without heavy metrics.

Usage:
  python analyse_expanded.py --task DRC
  python analyse_expanded.py --task congestion
  python analyse_expanded.py --task congestion --n_samples 0
"""

import os
import argparse
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Channel names — must match N14 expanded generation order exactly
# ─────────────────────────────────────────────────────────────────────────────
DRC_CHANNEL_NAMES = [
    "macro_region",
    "cell_density",
    "RUDY_long",
    "RUDY_short",
    "RUDY_pin_long",
    "eGR_overflow_H",
    "eGR_overflow_V",
    "GR_overflow_H",
    "GR_overflow_V",
    "GR_util_H",
    "GR_util_V",
    "RUDY",
    "RUDY_pin",
    "macro_boundary_distance",
    "eGR_util_H",
    "eGR_util_V",
]

CONG_CHANNEL_NAMES = [
    "macro_region",
    "RUDY",
    "RUDY_pin",
    "RUDY_long",
    "RUDY_short",
    "cell_density",
    "macro_boundary_distance",
    "GR_util_H",
    "GR_util_V",
    "eGR_overflow_H",
    "eGR_overflow_V",
]

DATASET_ROOT = Path("/data2/kgt22001/CircuitNet-N14/training_set_expanded")

TASK_CONFIGS = {
    "DRC": {
        "feat_dir": DATASET_ROOT / "DRC" / "feature",
        "label_dir": DATASET_ROOT / "DRC" / "label",
        "ch_names": DRC_CHANNEL_NAMES,
        "n_ch": 16,
    },
    "congestion": {
        "feat_dir": DATASET_ROOT / "congestion" / "feature",
        "label_dir": DATASET_ROOT / "congestion" / "label",
        "ch_names": CONG_CHANNEL_NAMES,
        "n_ch": 11,
    },
}


def load_sample(feat_dir, label_dir, fname):
    feat = np.load(feat_dir / fname).astype(np.float32)
    label = np.load(label_dir / fname).astype(np.float32)
    return feat, label


def to_chw_feat(arr):
    """(H,W,C) or (C,H,W) → (C,H,W)"""
    if arr.ndim == 3 and arr.shape[2] < arr.shape[0]:
        return arr.transpose(2, 0, 1)
    return arr


def to_hw_label(arr):
    """Any shape → (H,W)"""
    arr = arr.squeeze()
    assert arr.ndim == 2, f"Label squeeze failed: {arr.shape}"
    return arr


def corr_safe(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def avg_pool_2d(arr, factor):
    h, w = arr.shape
    hh = (h // factor) * factor
    ww = (w // factor) * factor
    arr = arr[:hh, :ww]
    return arr.reshape(hh // factor, factor, ww // factor, factor).mean(axis=(1, 3))


def repeat_upsample(arr, factor):
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["DRC", "congestion"])
    p.add_argument("--n_samples", type=int, default=200,
                   help="Samples to analyse (default 200, use 0 for all)")
    p.add_argument("--corr_pixels", type=int, default=8192,
                   help="Pixels sampled per sample for correlation (default 8192)")
    p.add_argument("--resolution_probe", type=int, default=24,
                   help="Number of samples for resolution analysis (default 24)")
    args = p.parse_args()

    cfg = TASK_CONFIGS[args.task]
    feat_dir = cfg["feat_dir"]
    label_dir = cfg["label_dir"]
    ch_names = cfg["ch_names"]
    n_ch_expected = cfg["n_ch"]

    print("=" * 70)
    print(f"  EXPANDED DATASET ANALYSIS — N14 — {args.task}")
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Feature dir  : {feat_dir}")
    print(f"  Label dir    : {label_dir}")
    print("=" * 70)

    if not feat_dir.is_dir():
        print(f"[ERROR] Feature dir not found: {feat_dir}")
        return
    if not label_dir.is_dir():
        print(f"[ERROR] Label dir not found: {label_dir}")
        return

    feat_files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".npy"))
    label_files = sorted(f for f in os.listdir(label_dir) if f.endswith(".npy"))
    common = sorted(set(feat_files) & set(label_files))

    print(f"  Feature files : {len(feat_files)}")
    print(f"  Label files   : {len(label_files)}")
    print(f"  Matched pairs : {len(common)}")

    if not common:
        print("[ERROR] No matched pairs found. Check directories.")
        return

    n = len(common) if args.n_samples == 0 else min(args.n_samples, len(common))
    files = common[:n]
    print(f"  Analysing    : {n} samples\n")

    # ── Probe first sample ───────────────────────────────────────────────────
    feat0, lbl0 = load_sample(feat_dir, label_dir, files[0])
    feat0_chw = to_chw_feat(feat0)
    lbl0_hw = to_hw_label(lbl0)

    print(f"  Raw feature shape : {feat0.shape}")
    print(f"  Raw label shape   : {lbl0.shape}")
    print(f"  Feature (C,H,W)   : {feat0_chw.shape}")
    print(f"  Label (H,W)       : {lbl0_hw.shape}")
    actual_ch = feat0_chw.shape[0]
    if actual_ch != n_ch_expected:
        print(f"  [WARN] Expected {n_ch_expected} channels, got {actual_ch}")
    print()

    # ── Aggregates ───────────────────────────────────────────────────────────
    total_pixels = 0
    total_zero = 0
    total_nonzero = 0
    label_sum = 0.0
    label_sq_sum = 0.0
    label_max = -np.inf
    label_values_for_percentiles = []

    ch_means = np.zeros(actual_ch, dtype=np.float64)
    ch_stds = np.zeros(actual_ch, dtype=np.float64)
    ch_mins = np.full(actual_ch, np.inf, dtype=np.float64)
    ch_maxs = np.full(actual_ch, -np.inf, dtype=np.float64)
    ch_corr_g = np.zeros(actual_ch, dtype=np.float64)
    ch_corr_nz = np.zeros(actual_ch, dtype=np.float64)
    n_valid_g = np.zeros(actual_ch, dtype=np.float64)
    n_valid_nz = np.zeros(actual_ch, dtype=np.float64)

    per_max = []
    per_mean = []
    per_nz_frac = []

    rng = np.random.default_rng(1234)

    # ── Main pass ────────────────────────────────────────────────────────────
    for fname in files:
        feat, lbl = load_sample(feat_dir, label_dir, fname)
        feat_chw = to_chw_feat(feat)
        lbl_hw = to_hw_label(lbl)
        lbl_flat = lbl_hw.ravel()

        # label stats
        nz_mask = lbl_flat > 0.01
        total_pixels += lbl_flat.size
        total_zero += np.sum(~nz_mask)
        total_nonzero += np.sum(nz_mask)
        label_sum += float(lbl_flat.sum())
        label_sq_sum += float((lbl_flat ** 2).sum())
        label_max = max(label_max, float(lbl_flat.max()))

        # collect only a subsample for percentiles when full dataset is huge
        if args.n_samples == 0:
            take = min(4096, lbl_flat.size)
            idx = rng.choice(lbl_flat.size, size=take, replace=False)
            label_values_for_percentiles.append(lbl_flat[idx])
        else:
            label_values_for_percentiles.append(lbl_flat)

        # per-sample label heterogeneity
        per_max.append(float(lbl_hw.max()))
        per_mean.append(float(lbl_hw.mean()))
        per_nz_frac.append(float(np.mean(lbl_hw > 0.01)))

        # sampled pixels for correlation
        take_corr = min(args.corr_pixels, lbl_flat.size)
        corr_idx = rng.choice(lbl_flat.size, size=take_corr, replace=False)
        lbl_corr = lbl_flat[corr_idx]
        nz_corr = lbl_corr > 0.01

        for c in range(actual_ch):
            fc = feat_chw[c].ravel()

            ch_means[c] += float(fc.mean())
            ch_stds[c] += float(fc.std())
            ch_mins[c] = min(ch_mins[c], float(fc.min()))
            ch_maxs[c] = max(ch_maxs[c], float(fc.max()))

            fc_corr = fc[corr_idx]
            cg = corr_safe(fc_corr, lbl_corr)
            if not np.isnan(cg):
                ch_corr_g[c] += cg
                n_valid_g[c] += 1

            if np.sum(nz_corr) >= 10:
                cnz = corr_safe(fc_corr[nz_corr], lbl_corr[nz_corr])
                if not np.isnan(cnz):
                    ch_corr_nz[c] += cnz
                    n_valid_nz[c] += 1

    # finalize
    ch_means /= n
    ch_stds /= n
    ch_corr_g = ch_corr_g / np.maximum(n_valid_g, 1)
    ch_corr_nz = ch_corr_nz / np.maximum(n_valid_nz, 1)

    per_max = np.array(per_max)
    per_mean = np.array(per_mean)
    per_nz_frac = np.array(per_nz_frac)

    label_mean = label_sum / total_pixels
    label_std = np.sqrt(max(label_sq_sum / total_pixels - label_mean ** 2, 0.0))
    label_values_for_percentiles = np.concatenate(label_values_for_percentiles)

    # ── Label distribution ───────────────────────────────────────────────────
    print("─" * 70)
    print("  LABEL DISTRIBUTION")
    print("─" * 70)
    print(f"  Total pixels     : {total_pixels:,}")
    print(f"  Zero (<=0.01)    : {total_zero:,}  = {100*total_zero/total_pixels:.2f}%")
    print(f"  Nonzero (>0.01)  : {total_nonzero:,}   = {100*total_nonzero/total_pixels:.2f}%")
    print(f"  mean={label_mean:.4f}  std={label_std:.4f}  max={label_max:.4f}")
    for p_val in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    p{p_val:>2}: {np.percentile(label_values_for_percentiles, p_val):.4f}", end="")
    print()

    if total_nonzero > 0:
        nz_vals = label_values_for_percentiles[label_values_for_percentiles > 0.01]
        ratio = total_zero / max(total_nonzero, 1)
        if nz_vals.size > 0:
            print(f"\n  Nonzero pixels: mean={nz_vals.mean():.4f}  std={nz_vals.std():.4f}")
        print(f"  Zero:nonzero ratio = {ratio:.1f}:1")
        print(f"  → Suggested focal_gamma ≈ {ratio:.0f}")
    print()

    # ── Feature channel stats ────────────────────────────────────────────────
    print("─" * 70)
    print("  FEATURE CHANNEL STATISTICS & LABEL CORRELATION")
    print("─" * 70)
    print(f"  {'Ch':>3}  {'Name':<26} {'Mean':>7} {'Std':>7} "
          f"{'Min':>7} {'Max':>7}  {'Corr_global':>12}  {'Corr_nz':>9}")
    print(f"  {'─'*3}  {'─'*26} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7}  {'─'*12}  {'─'*9}")

    for c in range(actual_ch):
        name = ch_names[c] if c < len(ch_names) else f"ch{c}"
        tag = "★" if abs(ch_corr_g[c]) > 0.3 else "◆" if abs(ch_corr_g[c]) > 0.1 else " "
        print(f"  {c:>3}  {name:<26} {ch_means[c]:>7.3f} {ch_stds[c]:>7.3f} "
              f"{ch_mins[c]:>7.3f} {ch_maxs[c]:>7.3f}  "
              f"{ch_corr_g[c]:>+12.4f}  {ch_corr_nz[c]:>+9.4f}  {tag}")

    print()
    high_corr = [ch_names[c] for c in range(actual_ch)
                 if abs(ch_corr_g[c]) > 0.3 and c < len(ch_names)]
    mod_corr = [ch_names[c] for c in range(actual_ch)
                if 0.1 < abs(ch_corr_g[c]) <= 0.3 and c < len(ch_names)]
    print(f"  High correlation (|r|>0.3): {high_corr}")
    print(f"  Moderate (0.1<|r|<=0.3):   {mod_corr}")
    print()

    # ── Per-sample label heterogeneity ───────────────────────────────────────
    print("─" * 70)
    print("  PER-SAMPLE LABEL HETEROGENEITY")
    print("─" * 70)
    print(f"  Per-sample max:  mean={per_max.mean():.4f}  "
          f"std={per_max.std():.4f}  min={per_max.min():.4f}")
    print(f"  Per-sample mean: mean={per_mean.mean():.4f}  "
          f"std={per_mean.std():.4f}")
    print(f"  Nonzero fraction: mean={per_nz_frac.mean():.4f}  "
          f"median={np.median(per_nz_frac):.4f}")
    clean = np.sum(per_nz_frac < 0.01)
    print(f"  Clean samples (nz_frac<1%): {clean}/{n} = {100*clean/n:.1f}%")
    print()

    # ── Resolution / latent-size check ───────────────────────────────────────
    print("─" * 70)
    print("  LATENT RESOLUTION ANALYSIS")
    print("─" * 70)
    H, W = lbl0_hw.shape
    print(f"  Image size: {H}×{W}")

    probe_files = files[:min(args.resolution_probe, len(files))]
    for factor in [2, 4, 8]:
        Hl, Wl = H // factor, W // factor
        lost = 0
        mass_retained = []
        corr_recon = []

        for fname in probe_files:
            _, lbl = load_sample(feat_dir, label_dir, fname)
            lbl_hw = to_hw_label(lbl)

            orig = lbl_hw[:Hl * factor, :Wl * factor]
            orig_mass = float(orig.sum())
            orig_nz = float(np.sum(orig > 0.01))

            if orig_nz == 0:
                continue

            ds = avg_pool_2d(orig, factor)
            up = repeat_upsample(ds, factor)

            ds_nz = float(np.sum(ds > (0.01 / factor)))
            if ds_nz == 0:
                lost += 1

            if orig_mass > 1e-8:
                mass_retained.append(float(up.sum() / orig_mass))

            rc = corr_safe(orig.ravel(), up.ravel())
            if not np.isnan(rc):
                corr_recon.append(rc)

        mr = np.mean(mass_retained) if mass_retained else float("nan")
        cr = np.mean(corr_recon) if corr_recon else float("nan")

        print(f"  Factor {factor}x → {Hl}×{Wl}: "
              f"lost-all-signal={lost}/{len(probe_files)}  "
              f"mean_mass_retained={mr:.3f}  "
              f"mean_recon_corr={cr:.3f}")

    print()
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()