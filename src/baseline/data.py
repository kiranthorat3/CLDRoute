#!/usr/bin/env python3
"""
data.py — Vanilla baseline dataset (image-only conditioning, no metadata).

Key design choices:
  - No prompt_codec, no metadata parsing, no text conditioning
  - Feature images (multi-channel) = conditioning signal
  - Label images (1-channel) = generation target
  - All loaded as float32, expected in [0, 1] from CircuitNet preprocessing
  - One-time range verification on first load
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_chw(arr: np.ndarray) -> np.ndarray:
    """
    Convert to channels-first format (C, H, W).

    Handles:
      - (H, W)       → (1, H, W)
      - (H, W, C)    → (C, H, W)
      - (C, H, W)    → (C, H, W)  [if first dim is small channel count]
    """
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        # Heuristic: if dim-0 is plausible channel count and dim-1 == dim-2,
        # it's already CHW. Otherwise treat as HWC.
        if arr.shape[1] == arr.shape[2] and arr.shape[0] <= 64:
            return arr
        return arr.transpose(2, 0, 1)
    raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")


def _base_from_path(p: str) -> str:
    """Extract filename without extension."""
    return os.path.splitext(os.path.basename(str(p)))[0]


def _read_csv_bases(csv_path: str) -> List[str]:
    """
    Read CSV and return list of base filenames.
    Supports single-column (feat_path) or two-column (feat_path, lbl_path).
    """
    import pandas as pd
    df = pd.read_csv(csv_path, header=None)
    bases = df.iloc[:, 0].astype(str).apply(_base_from_path).tolist()
    return bases


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class CircuitNetDataset(Dataset):
    """
    Simple dataset: loads (feature_image, label_image) pairs.
    No metadata encoding.
    """

    def __init__(
        self,
        csv_file: str,
        feature_dir: str,
        label_dir: str,
        check_files: bool = True,
        verify_range: bool = True,
        max_verify: int = 10,
    ):
        self.feature_dir = feature_dir
        self.label_dir = label_dir

        # Load and validate file list
        all_bases = _read_csv_bases(csv_file)
        if not all_bases:
            raise ValueError(f"Empty CSV: {csv_file}")

        kept: List[str] = []
        dropped: List[Tuple[str, str]] = []

        for base in all_bases:
            feat_path = os.path.join(feature_dir, base + ".npy")
            lbl_path = os.path.join(label_dir, base + ".npy")

            if check_files:
                if not os.path.isfile(feat_path):
                    dropped.append((base, "missing_feat"))
                    continue
                if not os.path.isfile(lbl_path):
                    dropped.append((base, "missing_lbl"))
                    continue

            kept.append(base)

        if not kept:
            raise ValueError(
                f"No valid samples in {csv_file}. "
                f"Example drops: {dropped[:5]}"
            )

        self.ids = kept

        # Probe shapes from first sample
        f0 = _to_chw(np.load(os.path.join(feature_dir, kept[0] + ".npy")).astype(np.float32))
        l0 = _to_chw(np.load(os.path.join(label_dir, kept[0] + ".npy")).astype(np.float32))
        self.C_feat = int(f0.shape[0])
        self.C_label = int(l0.shape[0])
        self.H = int(f0.shape[1])
        self.W = int(f0.shape[2])

        # One-time range verification
        if verify_range:
            self._verify_ranges(max_verify)

        csv_name = os.path.basename(csv_file)
        print(
            f"[Dataset] {csv_name}: {len(kept)} samples "
            f"(dropped {len(dropped)}) | "
            f"feat={self.C_feat}x{self.H}x{self.W} | "
            f"label={self.C_label}x{self.H}x{self.W}"
        )

    def _verify_ranges(self, n: int):
        """
        Verify that features and labels are in [0, 1].
        This is critical: the diffusion model maps labels to [-1, 1]
        via x0 = labels * 2 - 1. If labels aren't in [0, 1], the noise
        schedule assumptions break.
        """
        feat_min, feat_max = float("inf"), float("-inf")
        lbl_min, lbl_max = float("inf"), float("-inf")

        for base in self.ids[:n]:
            feat = np.load(os.path.join(self.feature_dir, base + ".npy")).astype(np.float32)
            lbl = np.load(os.path.join(self.label_dir, base + ".npy")).astype(np.float32)
            feat_min = min(feat_min, float(feat.min()))
            feat_max = max(feat_max, float(feat.max()))
            lbl_min = min(lbl_min, float(lbl.min()))
            lbl_max = max(lbl_max, float(lbl.max()))

        tolerance = 1e-3
        feat_ok = (feat_min >= -tolerance) and (feat_max <= 1.0 + tolerance)
        lbl_ok = (lbl_min >= -tolerance) and (lbl_max <= 1.0 + tolerance)

        if feat_ok and lbl_ok:
            print(
                f"[Dataset] Range check PASSED: "
                f"feat=[{feat_min:.4f}, {feat_max:.4f}] "
                f"lbl=[{lbl_min:.4f}, {lbl_max:.4f}]"
            )
        else:
            msg = (
                f"[Dataset] Range check FAILED: "
                f"feat=[{feat_min:.4f}, {feat_max:.4f}] "
                f"lbl=[{lbl_min:.4f}, {lbl_max:.4f}]. "
                f"Diffusion assumes [0,1] data. Fix normalization in preprocessing."
            )
            print(f"WARNING: {msg}")
            # Don't raise — let the user decide. But print loudly.

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        base = self.ids[idx]
        feat = np.clip(
            _to_chw(np.load(os.path.join(self.feature_dir, base + ".npy")).astype(np.float32)),
            0.0, 1.0,
        )
        lbl = np.clip(
            _to_chw(np.load(os.path.join(self.label_dir, base + ".npy")).astype(np.float32)),
            0.0, 1.0,
        )
        return (
            torch.from_numpy(np.ascontiguousarray(feat)),
            torch.from_numpy(np.ascontiguousarray(lbl)),
            base,
        )


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------

def _collate(batch):
    feats, lbls, names = zip(*batch)
    return (
        torch.stack(list(feats), 0),
        torch.stack(list(lbls), 0),
        list(names),
    )


def make_loaders(
    csv_train: str,
    csv_val: str,
    feature_dir: str,
    label_dir: str,
    batch_size: int,
    num_workers: int,
):
    ds_train = CircuitNetDataset(csv_train, feature_dir, label_dir)
    ds_val = CircuitNetDataset(
        csv_val, feature_dir, label_dir, verify_range=True
    )

    # Consistency check: channels must match
    assert ds_train.C_feat == ds_val.C_feat, (
        f"Feature channel mismatch: train={ds_train.C_feat} vs val={ds_val.C_feat}"
    )
    assert ds_train.C_label == ds_val.C_label, (
        f"Label channel mismatch: train={ds_train.C_label} vs val={ds_val.C_label}"
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )

    return ds_train, ds_val, loader_train, loader_val