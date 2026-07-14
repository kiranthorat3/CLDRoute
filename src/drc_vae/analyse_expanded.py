#!/usr/bin/env python3
"""
analyse_expanded.py — Dataset analysis for expanded-feature CircuitNet-N28.
Run before designing VAE/AE for the expanded dataset.

Usage:
  python analyse_expanded.py --task DRC
  python analyse_expanded.py --task congestion
"""
import os
import argparse
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Channel names — must match generate_training_set_expanded.py order
# ─────────────────────────────────────────────────────────────────────────────
DRC_CHANNEL_NAMES = [
    # original 9
    "macro_region",
    "cell_density",
    "RUDY_long",
    "RUDY_short",
    "RUDY_pin_long",
    "eGR_overflow_H",
    "eGR_overflow_V",
    "GR_overflow_H",
    "GR_overflow_V",
    # new 7
    "GR_util_H",
    "GR_util_V",
    "RUDY",
    "RUDY_pin",
    "macro_boundary_distance",
    "eGR_util_H",
    "eGR_util_V",
]

CONG_CHANNEL_NAMES = [
    # original 3
    "macro_region",
    "RUDY",
    "RUDY_pin",
    # new 8
    "RUDY_long",
    "RUDY_short",
    "cell_density",
    "macro_boundary_distance",
    "GR_util_H",
    "GR_util_V",
    "eGR_overflow_H",
    "eGR_overflow_V",
]

TASK_CONFIGS = {
    "DRC": {
        "feat_dir":   "/data2/kgt22001/CircuitNet-N28/training_set_expanded/DRC/feature",
        "label_dir":  "/data2/kgt22001/CircuitNet-N28/training_set_expanded/DRC/label",
        "ch_names":   DRC_CHANNEL_NAMES,
        "n_ch":       16,
    },
    "congestion": {
        "feat_dir":   "/data2/kgt22001/CircuitNet-N28/training_set_expanded/congestion/feature",
        "label_dir":  "/data2/kgt22001/CircuitNet-N28/training_set_expanded/congestion/label",
        "ch_names":   CONG_CHANNEL_NAMES,
        "n_ch":       11,
    },
}

def load_sample(feat_dir, label_dir, fname):
    feat  = np.load(os.path.join(feat_dir,  fname)).astype(np.float32)
    label = np.load(os.path.join(label_dir, fname)).astype(np.float32)
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["DRC", "congestion"])
    p.add_argument("--n_samples", type=int, default=200,
                   help="Samples to analyse (default 200, use 0 for all)")
    args = p.parse_args()

    cfg       = TASK_CONFIGS[args.task]
    feat_dir  = cfg["feat_dir"]
    label_dir = cfg["label_dir"]
    ch_names  = cfg["ch_names"]
    n_ch      = cfg["n_ch"]

    print("=" * 70)
    print(f"  EXPANDED DATASET ANALYSIS — {args.task}")
    print(f"  Feature dir : {feat_dir}")
    print(f"  Label dir   : {label_dir}")
    print("=" * 70)

    if not os.path.isdir(feat_dir):
        print(f"[ERROR] Feature dir not found: {feat_dir}")
        return
    if not os.path.isdir(label_dir):
        print(f"[ERROR] Label dir not found: {label_dir}")
        return

    feat_files  = sorted(f for f in os.listdir(feat_dir)  if f.endswith(".npy"))
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

    # ── Probe shape ──────────────────────────────────────────────────────────
    feat0, lbl0 = load_sample(feat_dir, label_dir, files[0])
    print(f"  Raw feature shape : {feat0.shape}")
    print(f"  Raw label shape   : {lbl0.shape}")
    feat0_chw = to_chw_feat(feat0)
    lbl0_hw   = to_hw_label(lbl0)
    print(f"  Feature (C,H,W)   : {feat0_chw.shape}")
    print(f"  Label (H,W)       : {lbl0_hw.shape}")
    actual_ch = feat0_chw.shape[0]
    if actual_ch != n_ch:
        print(f"  [WARN] Expected {n_ch} channels, got {actual_ch}")
    print()

    # ── Label distribution ────────────────────────────────────────────────────
    print("─" * 70)
    print("  LABEL DISTRIBUTION")
    print("─" * 70)
    all_labels = []
    for fname in files:
        _, lbl = load_sample(feat_dir, label_dir, fname)
        all_labels.append(to_hw_label(lbl).ravel())
    all_labels = np.concatenate(all_labels)
    nz_mask    = all_labels > 0.01
    print(f"  Total pixels     : {len(all_labels):,}")
    print(f"  Zero (<=0.01)    : {np.sum(~nz_mask):,}  = {100*np.mean(~nz_mask):.2f}%")
    print(f"  Nonzero (>0.01)  : {np.sum(nz_mask):,}   = {100*np.mean(nz_mask):.2f}%")
    print(f"  mean={all_labels.mean():.4f}  std={all_labels.std():.4f}  "
          f"max={all_labels.max():.4f}")
    for p_val in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    p{p_val:>2}: {np.percentile(all_labels, p_val):.4f}", end="")
    print()
    if nz_mask.any():
        nz = all_labels[nz_mask]
        print(f"\n  Nonzero pixels: mean={nz.mean():.4f}  std={nz.std():.4f}")
        ratio = np.sum(~nz_mask) / max(np.sum(nz_mask), 1)
        print(f"  Zero:nonzero ratio = {ratio:.1f}:1")
        print(f"  → Suggested focal_gamma ≈ {ratio:.0f}")
    print()

    # ── Per-channel feature stats ─────────────────────────────────────────────
    print("─" * 70)
    print("  FEATURE CHANNEL STATISTICS & LABEL CORRELATION")
    print("─" * 70)
    print(f"  {'Ch':>3}  {'Name':<26} {'Mean':>7} {'Std':>7} "
          f"{'Min':>7} {'Max':>7}  {'Corr_global':>12}  {'Corr_nz':>9}")
    print(f"  {'─'*3}  {'─'*26} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7}  {'─'*12}  {'─'*9}")

    ch_means  = np.zeros(actual_ch)
    ch_stds   = np.zeros(actual_ch)
    ch_mins   = np.zeros(actual_ch)
    ch_maxs   = np.zeros(actual_ch)
    ch_corr_g = np.zeros(actual_ch)   # global pearson with label
    ch_corr_nz= np.zeros(actual_ch)   # nonzero-label-pixel pearson
    n_valid_g  = np.zeros(actual_ch)
    n_valid_nz = np.zeros(actual_ch)

    for fname in files:
        feat, lbl = load_sample(feat_dir, label_dir, fname)
        feat_chw  = to_chw_feat(feat)
        lbl_flat  = to_hw_label(lbl).ravel()
        nz        = lbl_flat > 0.01

        for c in range(actual_ch):
            fc = feat_chw[c].ravel()
            ch_means[c] += fc.mean()
            ch_stds[c]  += fc.std()
            ch_mins[c]   = min(ch_mins[c], fc.min()) if fname != files[0] else fc.min()
            ch_maxs[c]   = max(ch_maxs[c], fc.max()) if fname != files[0] else fc.max()

            if np.std(fc) > 1e-8 and np.std(lbl_flat) > 1e-8:
                ch_corr_g[c]  += float(np.corrcoef(fc, lbl_flat)[0, 1])
                n_valid_g[c]  += 1

            if nz.sum() >= 10 and np.std(fc[nz]) > 1e-8 and np.std(lbl_flat[nz]) > 1e-8:
                ch_corr_nz[c]  += float(np.corrcoef(fc[nz], lbl_flat[nz])[0, 1])
                n_valid_nz[c]  += 1

    ch_means  /= n
    ch_stds   /= n
    ch_corr_g  = ch_corr_g  / np.maximum(n_valid_g,  1)
    ch_corr_nz = ch_corr_nz / np.maximum(n_valid_nz, 1)

    for c in range(actual_ch):
        name = ch_names[c] if c < len(ch_names) else f"ch{c}"
        tag  = "★" if abs(ch_corr_g[c]) > 0.3 else \
               "◆" if abs(ch_corr_g[c]) > 0.1 else " "
        print(f"  {c:>3}  {name:<26} {ch_means[c]:>7.3f} {ch_stds[c]:>7.3f} "
              f"{ch_mins[c]:>7.3f} {ch_maxs[c]:>7.3f}  "
              f"{ch_corr_g[c]:>+12.4f}  {ch_corr_nz[c]:>+9.4f}  {tag}")

    print()
    high_corr = [ch_names[c] for c in range(actual_ch)
                 if abs(ch_corr_g[c]) > 0.3 and c < len(ch_names)]
    mod_corr  = [ch_names[c] for c in range(actual_ch)
                 if 0.1 < abs(ch_corr_g[c]) <= 0.3 and c < len(ch_names)]
    print(f"  High correlation (|r|>0.3): {high_corr}")
    print(f"  Moderate (0.1<|r|<=0.3):   {mod_corr}")

    # ── Per-sample label stats ────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("  PER-SAMPLE LABEL HETEROGENEITY")
    print("─" * 70)
    per_max  = []
    per_mean = []
    per_nz_frac = []
    for fname in files:
        _, lbl = load_sample(feat_dir, label_dir, fname)
        lbl_hw = to_hw_label(lbl)
        per_max.append(float(lbl_hw.max()))
        per_mean.append(float(lbl_hw.mean()))
        per_nz_frac.append(float(np.mean(lbl_hw > 0.01)))

    per_max  = np.array(per_max)
    per_mean = np.array(per_mean)
    per_nz_frac = np.array(per_nz_frac)

    print(f"  Per-sample max:  mean={per_max.mean():.4f}  "
          f"std={per_max.std():.4f}  min={per_max.min():.4f}")
    print(f"  Per-sample mean: mean={per_mean.mean():.4f}  "
          f"std={per_mean.std():.4f}")
    print(f"  Nonzero fraction: mean={per_nz_frac.mean():.4f}  "
          f"median={np.median(per_nz_frac):.4f}")
    clean = np.sum(per_nz_frac < 0.01)
    print(f"  Clean samples (nz_frac<1%): {clean}/{n} = {100*clean/n:.1f}%")

    # ── Resolution check ─────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("  LATENT RESOLUTION ANALYSIS")
    print("─" * 70)
    H, W = lbl0_hw.shape
    print(f"  Image size: {H}×{W}")
    for factor in [4, 8]:
        Hl, Wl = H // factor, W // factor
        lost = 0
        for fname in files[:50]:
            _, lbl = load_sample(feat_dir, label_dir, fname)
            lbl_hw = to_hw_label(lbl)
            orig_nz = float(np.sum(lbl_hw > 0.01))
            if orig_nz == 0:
                continue
            lbl_ds = lbl_hw[:Hl*factor, :Wl*factor].reshape(
                Hl, factor, Wl, factor).mean(axis=(1,3))
            ds_nz = float(np.sum(lbl_ds > (0.01/factor)))
            if ds_nz == 0:
                lost += 1
        print(f"  Factor {factor}x → {Hl}×{Wl}: "
              f"samples losing all signal = {lost}/50")

    print()
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()