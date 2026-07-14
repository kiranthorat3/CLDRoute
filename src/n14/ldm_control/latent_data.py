#!/usr/bin/env python3
"""
latent_data.py — Data pipeline for LDM training and sampling.

Returns (feat, label, name) per sample.
  feat:  (C_feat_effective, 256, 256) float32 in [0, 1]
  label: (1, 256, 256) float32 in [0, 1]
  name:  sample stem, used for stable per-sample seeding

Notes:
  - Compatible with both N14 and N28
  - No label leakage here: split CSV is supplied explicitly
  - Supports dropping selected feature channels via cfg.drop_feat_channels
"""

from __future__ import annotations
import os
import csv
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from latent_config import LatentConfig


def _parse_csv(csv_path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{csv_path} line {i}: expected at least 2 columns, got {len(row)}"
                )
            feat_path = row[0].strip()
            label_path = row[1].strip()
            pairs.append((feat_path, label_path))
    return pairs


def _feat_to_chw(arr: np.ndarray) -> np.ndarray:
    """Convert feature array to (C,H,W)."""
    if arr.ndim != 3:
        raise ValueError(f"Unexpected feature shape: {arr.shape}")
    # HWC -> CHW
    if arr.shape[2] < arr.shape[0]:
        return arr.transpose(2, 0, 1)
    # Already CHW
    return arr


def _label_to_chw(arr: np.ndarray) -> np.ndarray:
    """Convert label array to (1,H,W)."""
    arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Cannot convert label shape {arr.shape} to (1,H,W)")
    return arr[np.newaxis]


class LatentDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        feature_dir: str,
        label_dir: str,
        drop_channels: List[int] | None = None,
        split: str = "train",
        verify: bool = True,
    ):
        self.split = split
        self.drop_channels = sorted(set(drop_channels or []))

        raw_pairs = _parse_csv(csv_path)
        self.samples: List[Tuple[str, str, str]] = []
        missing: List[Tuple[str, str]] = []

        for feat_path, label_path in raw_pairs:
            if not os.path.isabs(feat_path):
                feat_path = os.path.join(feature_dir, feat_path)
            if not os.path.isabs(label_path):
                label_path = os.path.join(label_dir, label_path)

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

        # Probe shapes
        try:
            feat0 = np.load(self.samples[0][0]).astype(np.float32)
            label0 = np.load(self.samples[0][1]).astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"[{split}] Failed to load probe sample:\n"
                f"  feat={self.samples[0][0]}\n"
                f"  label={self.samples[0][1]}\n"
                f"  error={e}"
            )

        feat0_chw = _feat_to_chw(feat0)
        label0_chw = _label_to_chw(label0)

        raw_ch = feat0_chw.shape[0]
        bad_drop = [c for c in self.drop_channels if c < 0 or c >= raw_ch]
        if bad_drop:
            raise ValueError(
                f"[{split}] Invalid drop_channels {bad_drop} for raw_ch={raw_ch}"
            )

        self._keep = [c for c in range(raw_ch) if c not in self.drop_channels]
        self.C_feat = len(self._keep)
        self.H = feat0_chw.shape[1]
        self.W = feat0_chw.shape[2]

        status = "OK" if not missing else f"WARNING: {len(missing)} missing"
        drop_str = f" drop_ch={self.drop_channels}" if self.drop_channels else ""
        print(
            f"[Data:{split}] {len(self.samples)} samples | "
            f"feat={self.C_feat}ch (raw={raw_ch}{drop_str}) | "
            f"label={label0_chw.shape[0]}ch | {self.H}×{self.W} | {status}"
        )

        if verify:
            self._verify()

    def _verify(self):
        """Check feat/label ranges and readability on first 10 samples."""
        f_lo = f_hi = l_lo = l_hi = None

        for fp, lp, _ in self.samples[:10]:
            try:
                f = np.load(fp).astype(np.float32)
                l = np.load(lp).astype(np.float32)
            except Exception as e:
                raise RuntimeError(
                    f"[Data:{self.split}] Corrupt/unreadable sample:\n"
                    f"  feat={fp}\n"
                    f"  label={lp}\n"
                    f"  error={e}"
                )

            f_lo = f.min() if f_lo is None else min(f_lo, f.min())
            f_hi = f.max() if f_hi is None else max(f_hi, f.max())
            l_lo = l.min() if l_lo is None else min(l_lo, l.min())
            l_hi = l.max() if l_hi is None else max(l_hi, l.max())

        tol = 1e-3
        f_ok = (f_lo >= -tol) and (f_hi <= 1.0 + tol)
        l_ok = (l_lo >= -tol) and (l_hi <= 1.0 + tol)

        print(f"[Data:{self.split}] feat  range {'PASS' if f_ok else 'WARN'}: [{f_lo:.4f}, {f_hi:.4f}]")
        print(f"[Data:{self.split}] label range {'PASS' if l_ok else 'WARN'}: [{l_lo:.4f}, {l_hi:.4f}]")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat_path, label_path, name = self.samples[idx]

        feat = np.load(feat_path).astype(np.float32)
        label = np.load(label_path).astype(np.float32)

        feat_chw = _feat_to_chw(feat)[self._keep]
        label_chw = _label_to_chw(label).clip(0.0, 1.0)

        return (
            torch.from_numpy(np.ascontiguousarray(feat_chw)),
            torch.from_numpy(np.ascontiguousarray(label_chw)),
            name,
        )


def _collate(batch):
    feats, labels, names = zip(*batch)
    return torch.stack(feats, 0), torch.stack(labels, 0), list(names)


def make_latent_loaders(cfg: LatentConfig):
    ds_train = LatentDataset(
        csv_path=cfg.csv_train,
        feature_dir=cfg.feature_dir,
        label_dir=cfg.label_dir,
        drop_channels=cfg.drop_feat_channels,
        split="train",
        verify=True,
    )
    ds_val = LatentDataset(
        csv_path=cfg.csv_val,
        feature_dir=cfg.feature_dir,
        label_dir=cfg.label_dir,
        drop_channels=cfg.drop_feat_channels,
        split="val",
        verify=False,
    )

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