#!/usr/bin/env python3
"""
latent_data.py — Data pipeline for LDM training and sampling.

Returns (feat, label, name) per sample.
  feat:  (C_feat_effective, 256, 256) float32 in [0, 1]
         Dead channels dropped per cfg.drop_feat_channels.
  label: (1, 256, 256) float32 in [0, 1]
  name:  str — sample stem, used for stable per-sample seeding in sampler.

Label leaking prevention:
  - Normalization stats (z_mean, z_std) computed on train split only
    (enforced in latent_trainer.py, not here).
  - This loader enforces split-correct CSV usage by accepting split-specific
    csv_path as an argument — no implicit fallback to training data.
  - verify=True on train split checks label range [0, 1].

CSV format (two columns):
  feature_path, label_path
Both paths are absolute. Loader reads feature from feature_path and label
from label_path directly — no path reconstruction from basenames.
"""
from __future__ import annotations
import os
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from latent_config import LatentConfig


# ─────────────────────────────────────────────────────────────────────────────
# CSV parsing
# ─────────────────────────────────────────────────────────────────────────────
def _parse_csv(csv_path: str) -> List[Tuple[str, str]]:
    """
    Parse two-column CSV: [feature_path, label_path].
    Returns list of (feat_path, label_path) tuples.
    Raises clearly if format is wrong.
    """
    pairs = []
    with open(csv_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                raise ValueError(
                    f"{csv_path} line {i+1}: expected 2 columns, "
                    f"got {len(parts)}. Content: {line!r}"
                )
            feat_path  = parts[0].strip()
            label_path = parts[1].strip()
            pairs.append((feat_path, label_path))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Array helpers
# ─────────────────────────────────────────────────────────────────────────────
def _feat_to_chw(arr: np.ndarray) -> np.ndarray:
    """(H,W,C) → (C,H,W). Validates 3D input."""
    if arr.ndim == 3 and arr.shape[2] < arr.shape[0]:
        return arr.transpose(2, 0, 1)
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unexpected feature shape: {arr.shape}")

def _label_to_chw(arr: np.ndarray) -> np.ndarray:
    """Any label shape → (1, H, W)."""
    arr = arr.squeeze()
    if arr.ndim == 2:
        return arr[np.newaxis]
    raise ValueError(f"Cannot convert label shape {arr.shape} to (1,H,W)")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class LatentDataset(Dataset):
    """
    Dataset for LDM training/evaluation.
    Returns (feat_tensor, label_tensor, name).

    Args:
        csv_path:           path to split-specific CSV file
        feature_dir:        root directory for features (used only if CSV
                            contains bare basenames rather than full paths)
        label_dir:          root directory for labels (same caveat)
        drop_channels:      list of channel indices to drop from features
        split:              "train", "val", or "test" — for logging only
        verify:             if True, checks feat/label ranges on first 10 samples
    """
    def __init__(
        self,
        csv_path:      str,
        feature_dir:   str,
        label_dir:     str,
        drop_channels: List[int] = [],
        split:         str = "train",
        verify:        bool = True,
    ):
        self.split         = split
        self.drop_channels = drop_channels

        raw_pairs = _parse_csv(csv_path)
        self.samples: List[Tuple[str, str, str]] = []   # (feat_path, label_path, name)
        missing = []

        for feat_path, label_path in raw_pairs:
            # Resolve to absolute paths if bare basenames
            if not os.path.isabs(feat_path):
                feat_path = os.path.join(feature_dir, feat_path)
            if not os.path.isabs(label_path):
                label_path = os.path.join(label_dir, label_path)
            # Append .npy if missing
            if not feat_path.endswith(".npy"):
                feat_path += ".npy"
            if not label_path.endswith(".npy"):
                label_path += ".npy"

            if os.path.isfile(feat_path) and os.path.isfile(label_path):
                name = os.path.splitext(os.path.basename(label_path))[0]
                self.samples.append((feat_path, label_path, name))
            else:
                missing.append((feat_path, label_path))

        if not self.samples:
            raise ValueError(
                f"[{split}] No valid samples found in {csv_path}.\n"
                f"First missing: {missing[:2]}"
            )

        # Probe shapes and derive C_feat
        feat0  = np.load(self.samples[0][0]).astype(np.float32)
        label0 = np.load(self.samples[0][1]).astype(np.float32)
        feat0_chw  = _feat_to_chw(feat0)
        label0_chw = _label_to_chw(label0)

        raw_ch     = feat0_chw.shape[0]
        self.C_feat = raw_ch - len(drop_channels)
        self.H      = feat0_chw.shape[1]
        self.W      = feat0_chw.shape[2]
        self._keep  = [c for c in range(raw_ch) if c not in drop_channels]

        status = "OK" if not missing else f"WARNING: {len(missing)} missing"
        drop_str = f" drop_ch={drop_channels}" if drop_channels else ""
        print(f"[Data:{split}] {len(self.samples)} samples | "
              f"feat={self.C_feat}ch (raw={raw_ch}{drop_str}) | "
              f"label={label0_chw.shape[0]}ch | "
              f"{self.H}×{self.W} | {status}")

        if verify:
            self._verify()

    def _verify(self):
        """Check feat and label ranges on first 10 samples."""
        f_lo = f_hi = l_lo = l_hi = None
        for fp, lp, _ in self.samples[:10]:
            f = np.load(fp).astype(np.float32)
            l = np.load(lp).astype(np.float32)
            f_lo = f.min() if f_lo is None else min(f_lo, f.min())
            f_hi = f.max() if f_hi is None else max(f_hi, f.max())
            l_lo = l.min() if l_lo is None else min(l_lo, l.min())
            l_hi = l.max() if l_hi is None else max(l_hi, l.max())
        tol = 1e-3
        f_ok = (f_lo >= -tol) and (f_hi <= 1.0 + tol)
        l_ok = (l_lo >= -tol) and (l_hi <= 1.0 + tol)
        print(f"[Data:{self.split}] feat  range {'PASS' if f_ok else 'WARN'}: "
              f"[{f_lo:.4f}, {f_hi:.4f}]")
        print(f"[Data:{self.split}] label range {'PASS' if l_ok else 'WARN'}: "
              f"[{l_lo:.4f}, {l_hi:.4f}]")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat_path, label_path, name = self.samples[idx]

        feat  = np.load(feat_path).astype(np.float32)
        label = np.load(label_path).astype(np.float32)

        feat_chw  = _feat_to_chw(feat)[self._keep]   # drop dead channels
        label_chw = _label_to_chw(label).clip(0.0, 1.0)

        return (
            torch.from_numpy(np.ascontiguousarray(feat_chw)),
            torch.from_numpy(np.ascontiguousarray(label_chw)),
            name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────
def _collate(batch):
    feats, labels, names = zip(*batch)
    return torch.stack(feats, 0), torch.stack(labels, 0), list(names)


# ─────────────────────────────────────────────────────────────────────────────
# Loader factory
# ─────────────────────────────────────────────────────────────────────────────
def make_latent_loaders(cfg: LatentConfig):
    """
    Create train and val DataLoaders from cfg.
    Both use cfg.drop_feat_channels to exclude dead channels.
    """
    ds_train = LatentDataset(
        csv_path      = cfg.csv_train,
        feature_dir   = cfg.feature_dir,
        label_dir     = cfg.label_dir,
        drop_channels = cfg.drop_feat_channels,
        split         = "train",
        verify        = True,
    )
    ds_val = LatentDataset(
        csv_path      = cfg.csv_val,
        feature_dir   = cfg.feature_dir,
        label_dir     = cfg.label_dir,
        drop_channels = cfg.drop_feat_channels,
        split         = "val",
        verify        = False,
    )
    loader_train = DataLoader(
        ds_train,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,
        collate_fn  = _collate,
        drop_last   = True,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = max(1, cfg.num_workers // 2),
        pin_memory  = True,
        collate_fn  = _collate,
        drop_last   = False,
    )
    return ds_train, ds_val, loader_train, loader_val