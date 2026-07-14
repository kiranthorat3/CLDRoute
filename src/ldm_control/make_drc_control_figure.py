#!/usr/bin/env python3
"""
make_drc_controls_only_figure.py

Create a clean paper figure with 4 representative DRC routing controls
for one test sample.

Default panels:
  1) cell_density
  2) RUDY_pin
  3) GR_overflow_V
  4) eGR_util_V

Example:
  python make_drc_controls_only_figure.py \
    --out ./results/figures/drc_controls_only.pdf \
    --sample_seed 42

Choose a specific design:
  python make_drc_controls_only_figure.py \
    --out ./results/figures/drc_controls_only.pdf \
    --sample_name riscy_a
"""

from __future__ import annotations

import os
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = "/data2/kgt22001/CircuitNet-N28/training_set_expanded/DRC"
FEATURE_DIR = f"{ROOT}/feature"
LABEL_DIR   = f"{ROOT}/label"
CSV_TEST    = f"{ROOT}/files_design/test_N28.csv"

# ---------------------------------------------------------------------
# Channel definitions: must match expanded DRC feature order
# ---------------------------------------------------------------------
DRC_CHANNEL_NAMES = [
    "macro_region",            # 0
    "cell_density",            # 1
    "RUDY_long",               # 2
    "RUDY_short",              # 3
    "RUDY_pin_long",           # 4
    "eGR_overflow_H",          # 5
    "eGR_overflow_V",          # 6
    "GR_overflow_H",           # 7
    "GR_overflow_V",           # 8
    "GR_util_H",               # 9
    "GR_util_V",               # 10
    "RUDY",                    # 11
    "RUDY_pin",                # 12
    "macro_boundary_distance", # 13
    "eGR_util_H",              # 14
    "eGR_util_V",              # 15
]

DEFAULT_CHANNELS = [
    "cell_density",
    "RUDY_pin",
    "GR_overflow_V",
    "eGR_util_V",
]

PRETTY_NAME = {
    "macro_region": "Macro region",
    "cell_density": "Cell density",
    "RUDY_long": "RUDY long",
    "RUDY_short": "RUDY short",
    "RUDY_pin_long": "RUDY pin long",
    "eGR_overflow_H": "eGR overflow H",
    "eGR_overflow_V": "eGR overflow V",
    "GR_overflow_H": "GR overflow H",
    "GR_overflow_V": "GR overflow V",
    "GR_util_H": "GR util H",
    "GR_util_V": "GR util V",
    "RUDY": "RUDY",
    "RUDY_pin": "RUDY pin",
    "macro_boundary_distance": "Macro boundary dist.",
    "eGR_util_H": "eGR util H",
    "eGR_util_V": "eGR util V",
}

CHANNEL_TO_INDEX = {name: i for i, name in enumerate(DRC_CHANNEL_NAMES)}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_csv(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) >= 2:
                feat_path = row[0].strip()
                label_path = row[1].strip()
                name = Path(feat_path).stem
            else:
                name = Path(row[0].strip()).stem
                feat_path = os.path.join(FEATURE_DIR, name + ".npy")
                label_path = os.path.join(LABEL_DIR, name + ".npy")

            if os.path.isfile(feat_path) and os.path.isfile(label_path):
                rows.append((feat_path, label_path, name))
    return rows


def to_chw(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        if arr.shape[2] < arr.shape[0]:   # HWC -> CHW
            return arr.transpose(2, 0, 1)
        return arr
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def robust_norm(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    x = np.asarray(x, np.float32)
    v_lo = np.percentile(x, lo)
    v_hi = np.percentile(x, hi)
    if v_hi <= v_lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - v_lo) / (v_hi - v_lo), 0.0, 1.0)


def norm_channel(name: str, x: np.ndarray) -> np.ndarray:
    if name in {
        "macro_region", "cell_density",
        "eGR_overflow_H", "eGR_overflow_V",
        "GR_overflow_H", "GR_overflow_V",
        "GR_util_H", "GR_util_V",
        "eGR_util_H", "eGR_util_V"
    }:
        return np.clip(x, 0.0, 1.0)
    return robust_norm(x, lo=1.0, hi=99.0)


def pick_sample(rows, sample_seed: int | None, sample_name: str | None):
    if sample_name is not None:
        for item in rows:
            if item[2] == sample_name:
                return item
        raise ValueError(f"sample_name='{sample_name}' not found in test CSV")

    rng = np.random.default_rng(sample_seed if sample_seed is not None else 42)
    idx = int(rng.integers(0, len(rows)))
    return rows[idx]


# ---------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------
def build_figure(
    feat_path: str,
    design_name: str,
    channels: list[str],
    out_path: str,
):
    feat = np.clip(to_chw(np.load(feat_path)), 0.0, 1.0)   # (C,H,W)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))

    for ax, ch_name in zip(axes, channels):
        ch_idx = CHANNEL_TO_INDEX[ch_name]
        img = norm_channel(ch_name, feat[ch_idx])

        ax.imshow(img, cmap="viridis", interpolation="nearest")
        ax.set_title(PRETTY_NAME[ch_name], pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.3)

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {out_path}")
    print(f"[Design] {design_name}")
    print("[Channels]")
    for ch_name in channels:
        print(f"  - {ch_name} (ch {CHANNEL_TO_INDEX[ch_name]})")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_args():
    p = argparse.ArgumentParser(
        description="Make a representative DRC routing-controls figure."
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output PDF/PNG path",
    )
    p.add_argument(
        "--channels",
        nargs=4,
        default=DEFAULT_CHANNELS,
        help="Exactly 4 DRC channel names to visualize",
    )
    p.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Random seed for choosing one sample from the test set",
    )
    p.add_argument(
        "--sample_name",
        type=str,
        default=None,
        help="Optional exact sample name; overrides --sample_seed",
    )
    return p.parse_args()


def main():
    args = build_args()

    for ch in args.channels:
        if ch not in CHANNEL_TO_INDEX:
            raise ValueError(
                f"Unknown channel '{ch}'. Valid names are:\n" +
                ", ".join(DRC_CHANNEL_NAMES)
            )

    rows = parse_csv(CSV_TEST)
    if not rows:
        raise RuntimeError(f"No valid test samples found in {CSV_TEST}")

    feat_path, _, design_name = pick_sample(
        rows,
        sample_seed=args.sample_seed,
        sample_name=args.sample_name,
    )

    build_figure(
        feat_path=feat_path,
        design_name=design_name,
        channels=args.channels,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()