#!/usr/bin/env python3
"""
vae_data.py — Data pipeline for unconditional label VAE.

Returns (label, name) per sample.
No feature channels — VAE is unconditional.
Supports both:
  1) old CSV format:  one column  -> [path_or_name]
  2) new CSV format: two columns -> [feature_path, label_path]

For the new expanded dataset, the VAE uses the label path from column 1.
That is the correct source of truth.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from vae_config import VAEConfig


# ─────────────────────────────────────────────────────────────────────────────
# Array helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_chw(arr: np.ndarray) -> np.ndarray:
    """
    Convert label array to (C, H, W). Accepted inputs:
      (H, W)     -> (1, H, W)
      (1, H, W)  -> (1, H, W)
      (H, W, 1)  -> (1, H, W)
    """
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr
        if arr.shape[2] == 1:
            return arr.transpose(2, 0, 1)
    raise ValueError(
        f"Unexpected label shape {arr.shape}. "
        f"Expected (H,W), (1,H,W), or (H,W,1)."
    )


def _resolve_label_entries(csv_path: str, label_dir: str) -> List[Tuple[str, str]]:
    """
    Return a list of (label_path, sample_name).

    Supported CSV formats:
      - 1 column: [path_or_name]
          Interpreted as basename/path; label_path resolved through label_dir.
      - 2+ columns: [feature_path, label_path, ...]
          Uses column 1 as the label path directly.

    sample_name is always derived from the resolved label path basename.
    """
    df = pd.read_csv(csv_path, header=None)

    entries: List[Tuple[str, str]] = []

    if df.shape[1] >= 2:
        # New format: [feature_path, label_path]
        for raw_lp in df.iloc[:, 1].astype(str).tolist():
            raw_lp = raw_lp.strip()
            name = os.path.splitext(os.path.basename(raw_lp))[0]

            # If CSV stores a bare basename instead of a full path, resolve via label_dir
            if os.path.isabs(raw_lp) or os.path.isfile(raw_lp):
                lp = raw_lp
            else:
                if raw_lp.endswith(".npy"):
                    lp = os.path.join(label_dir, raw_lp)
                else:
                    lp = os.path.join(label_dir, raw_lp + ".npy")

            entries.append((lp, name))
    else:
        # Old format: one column only
        for raw_p in df.iloc[:, 0].astype(str).tolist():
            raw_p = raw_p.strip()
            base = os.path.splitext(os.path.basename(raw_p))[0]
            lp = os.path.join(label_dir, base + ".npy")
            entries.append((lp, base))

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LabelDataset(Dataset):
    """
    Dataset returning (label_tensor, name) per sample.
    Label: (1, H, W) float32 in [0, 1].
    """

    def __init__(
        self,
        csv_path: str,
        label_dir: str,
        split: str = "train",
        verify: bool = True,
    ):
        self.label_dir = label_dir
        self.split = split

        all_entries = _resolve_label_entries(csv_path, label_dir)

        kept: List[Tuple[str, str]] = []
        dropped: List[Tuple[str, str]] = []

        for lp, name in all_entries:
            if os.path.isfile(lp):
                kept.append((lp, name))
            else:
                dropped.append((lp, name))

        if not kept:
            raise ValueError(
                f"No valid samples in {csv_path}. "
                f"First dropped entries: {dropped[:3]}"
            )

        self.samples = kept
        self.ids = [name for _, name in kept]

        # Probe shape
        probe = np.load(self.samples[0][0]).astype(np.float32)
        probe = _to_chw(probe)
        self.C = int(probe.shape[0])
        self.H = int(probe.shape[1])
        self.W = int(probe.shape[2])

        if verify:
            self._verify()

        status = "OK" if not dropped else f"WARNING: {len(dropped)} missing"
        print(
            f"[Data:{split}] {len(kept)} samples | "
            f"label={self.C}ch {self.H}×{self.W} | "
            f"dropped={len(dropped)} | {status}"
        )

    def _verify(self):
        """Check label range on first 10 samples."""
        lo, hi = float("inf"), float("-inf")
        for lp, _ in self.samples[:10]:
            arr = np.load(lp).astype(np.float32)
            lo = min(lo, float(arr.min()))
            hi = max(hi, float(arr.max()))
        tol = 1e-3
        ok = (lo >= -tol) and (hi <= 1.0 + tol)
        print(
            f"[Data:{self.split}] Range check "
            f"{'PASS' if ok else 'WARN'}: [{lo:.4f}, {hi:.4f}]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        lp, name = self.samples[idx]
        label = np.load(lp).astype(np.float32)
        label = _to_chw(label).clip(0.0, 1.0)
        return torch.from_numpy(np.ascontiguousarray(label)), name


# ─────────────────────────────────────────────────────────────────────────────
# Collate + loaders
# ─────────────────────────────────────────────────────────────────────────────

def _collate(batch):
    labels, names = zip(*batch)
    return torch.stack(list(labels), 0), list(names)


def make_loaders(cfg: VAEConfig):
    ds_train = LabelDataset(cfg.csv_train, cfg.label_dir, split="train", verify=True)
    ds_val   = LabelDataset(cfg.csv_val,   cfg.label_dir, split="val",   verify=False)

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )
    return ds_train, ds_val, loader_train, loader_val