#!/usr/bin/env python3
"""
vae_data_congestion.py — Data pipeline for congestion VAE.
Returns (label, name) per sample. No features — VAE is unconditional.
Handles both CSV formats:
  1) old: one column  [path_or_name]
  2) new: two columns [feature_path, label_path]
"""
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from vae_config_congestion import CongestionVAEConfig


def _to_chw(arr: np.ndarray) -> np.ndarray:
    """Any label array → (1, H, W)."""
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr
        if arr.shape[2] == 1:
            return arr.transpose(2, 0, 1)
    raise ValueError(f"Unexpected label shape {arr.shape}.")


def _resolve_label_entries(csv_path: str, label_dir: str) -> List[Tuple[str, str]]:
    df      = pd.read_csv(csv_path, header=None)
    entries = []
    if df.shape[1] >= 2:
        for raw_lp in df.iloc[:, 1].astype(str).tolist():
            raw_lp = raw_lp.strip()
            name   = os.path.splitext(os.path.basename(raw_lp))[0]
            if os.path.isabs(raw_lp) or os.path.isfile(raw_lp):
                lp = raw_lp
            else:
                lp = os.path.join(label_dir,
                                  raw_lp if raw_lp.endswith(".npy")
                                  else raw_lp + ".npy")
            entries.append((lp, name))
    else:
        for raw_p in df.iloc[:, 0].astype(str).tolist():
            raw_p = raw_p.strip()
            base  = os.path.splitext(os.path.basename(raw_p))[0]
            entries.append((os.path.join(label_dir, base + ".npy"), base))
    return entries


class CongestionLabelDataset(Dataset):
    """Returns (label_tensor, name). Label: (1,H,W) float32 in [0,1]."""

    def __init__(
        self,
        csv_path:  str,
        label_dir: str,
        split:     str  = "train",
        verify:    bool = True,
    ):
        self.split = split
        all_entries = _resolve_label_entries(csv_path, label_dir)
        kept, dropped = [], []
        for lp, name in all_entries:
            (kept if os.path.isfile(lp) else dropped).append((lp, name))
        if not kept:
            raise ValueError(f"No valid samples in {csv_path}. "
                             f"First dropped: {dropped[:3]}")
        self.samples = kept
        self.ids     = [n for _, n in kept]

        probe    = _to_chw(np.load(kept[0][0]).astype(np.float32))
        self.C   = int(probe.shape[0])
        self.H   = int(probe.shape[1])
        self.W   = int(probe.shape[2])

        if verify:
            self._verify()

        status = "OK" if not dropped else f"WARNING: {len(dropped)} missing"
        print(f"[Data:{split}] {len(kept)} samples | "
              f"label={self.C}ch {self.H}×{self.W} | "
              f"dropped={len(dropped)} | {status}")

    def _verify(self):
        lo, hi = float("inf"), float("-inf")
        for lp, _ in self.samples[:10]:
            arr = np.load(lp).astype(np.float32)
            lo  = min(lo, float(arr.min()))
            hi  = max(hi, float(arr.max()))
        ok = (lo >= -1e-3) and (hi <= 1.0 + 1e-3)
        print(f"[Data:{self.split}] Range check "
              f"{'PASS' if ok else 'WARN'}: [{lo:.4f}, {hi:.4f}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lp, name = self.samples[idx]
        label    = _to_chw(np.load(lp).astype(np.float32)).clip(0.0, 1.0)
        return torch.from_numpy(np.ascontiguousarray(label)), name


def _collate(batch):
    labels, names = zip(*batch)
    return torch.stack(list(labels), 0), list(names)


def make_loaders(cfg: CongestionVAEConfig):
    ds_train = CongestionLabelDataset(
        cfg.csv_train, cfg.label_dir, split="train", verify=True)
    ds_val   = CongestionLabelDataset(
        cfg.csv_val,   cfg.label_dir, split="val",   verify=False)

    loader_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=_collate, drop_last=True,
    )
    loader_val = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(1, cfg.num_workers // 2), pin_memory=True,
        collate_fn=_collate, drop_last=False,
    )
    return ds_train, ds_val, loader_train, loader_val