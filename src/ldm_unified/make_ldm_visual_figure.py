#!/usr/bin/env python3
"""
make_ldm_visual_figure.py

Create a clean publication-style visual comparison figure for either:
  - DRC latent diffusion results
  - Congestion latent diffusion results

Figure layout:
  rows    = selected test samples (default: 3)
  columns = [initial noise, conditioning image, generated output, ground truth]

Important:
  - Design names are NOT written inside the figure.
  - Metrics are NOT written inside the figure.
  - Both names and all metrics are printed to terminal log and saved to JSON.
  - By default, random samples are chosen differently each run.
    Pass --sample_seed if you want reproducible sample selection.

Examples:

DRC:
  python make_ldm_visual_figure.py \
    --task DRC \
    --ckpt /data2/kgt22001/cong_gen/gen_auto/ldm_updated_vae/runs/ldm_v2_DRC_N28/best_gen.pt \
    --out  /data2/kgt22001/cong_gen/gen_auto/ldm_updated_vae/results/drc_visual_validation.pdf

Congestion:
  python make_ldm_visual_figure.py \
    --task Congestion \
    --ckpt /data2/kgt22001/cong_gen/gen_auto/ldm_updated_vae/runs/ldm_v2_CongAE_N28/latest.pt \
    --out  /data2/kgt22001/cong_gen/gen_auto/ldm_updated_vae/results/congestion_visual_validation.pdf
"""

from __future__ import annotations

import os
import csv
import json
import hashlib
import argparse
import inspect
import importlib.util
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DRC_AE_DIR = "/data2/kgt22001/cong_gen/gen_auto/vae2"
CONG_AE_DIR = "/data2/kgt22001/cong_gen/gen_auto/ae"

TASK_CFG = {
    "DRC": {
        "feature_dir": "/data2/kgt22001/CircuitNet-N28/training_set/DRC/feature",
        "label_dir":   "/data2/kgt22001/CircuitNet-N28/training_set/DRC/label",
        "csv_test":    "/data2/kgt22001/CircuitNet-N28/training_set/DRC/files_design/test_N28.csv",
        "default_vae_dir": DRC_AE_DIR,
        "default_cond_rgb": (0, 1, 2),
    },
    "Congestion": {
        "feature_dir": "/data2/kgt22001/CircuitNet-N28/training_set/congestion/feature",
        "label_dir":   "/data2/kgt22001/CircuitNet-N28/training_set/congestion/label",
        "csv_test":    "/data2/kgt22001/CircuitNet-N28/training_set/congestion/files_design/test_N28.csv",
        "default_vae_dir": CONG_AE_DIR,
        "default_cond_rgb": (0, 1, 2),
    },
}


# =============================================================================
# Utilities
# =============================================================================
def _import_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _to_chw(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1] and arr.shape[1] == arr.shape[2]:
            return arr
        if arr.shape[2] < arr.shape[0] and arr.shape[0] == arr.shape[1]:
            return arr.transpose(2, 0, 1)
        if arr.shape[0] == 1:
            return arr
        if arr.shape[2] == 1:
            return arr.transpose(2, 0, 1)
    raise ValueError(f"Cannot parse array with shape {arr.shape} as CHW")


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def stable_int_from_name(name: str) -> int:
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def make_initial_noise(
    B: int, C: int, H: int, W: int, device: torch.device,
    base_seed: int, names: List[str], draw_k: int
) -> torch.Tensor:
    x = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
    for i, nm in enumerate(names):
        s = int(base_seed) ^ stable_int_from_name(str(nm)) ^ (int(draw_k) * 0x9E3779B1)
        g = torch.Generator(device=device)
        g.manual_seed(s & 0x7FFFFFFF)
        x[i:i+1] = torch.randn((1, C, H, W), device=device, generator=g)
    return x


def make_ddim_ts(T: int, steps: int, device: torch.device) -> torch.Tensor:
    ts = np.rint(np.linspace(T - 1, 0, steps, dtype=np.float64)).astype(np.int64)
    for i in range(1, len(ts)):
        if ts[i] >= ts[i - 1]:
            ts[i] = ts[i - 1] - 1
    return torch.tensor(np.clip(ts, 0, T - 1), device=device, dtype=torch.long)


def robust_normalize(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, lo_q)
    hi = np.percentile(x, hi_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def latent_noise_to_img(z0: np.ndarray) -> np.ndarray:
    img = z0.mean(axis=0)
    return robust_normalize(img)


def feature_rgb(feat: np.ndarray, channels: Tuple[int, int, int]) -> np.ndarray:
    c0, c1, c2 = channels
    c0 = min(max(c0, 0), feat.shape[0] - 1)
    c1 = min(max(c1, 0), feat.shape[0] - 1)
    c2 = min(max(c2, 0), feat.shape[0] - 1)
    rgb = np.stack([
        robust_normalize(feat[c0]),
        robust_normalize(feat[c1]),
        robust_normalize(feat[c2]),
    ], axis=-1)
    return rgb


def describe_array(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    return {
        "min": float(np.min(x)),
        "p1": float(np.percentile(x, 1)),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


# =============================================================================
# Dataset
# =============================================================================
class FigureDataset(Dataset):
    """
    Handles either:
      - 2-column CSV: feature_path,label_path
      - 1-column CSV: path or basename
    """
    def __init__(self, feature_dir: str, label_dir: str, csv_test: str):
        self.rows: List[Tuple[str, str, str]] = []

        with open(csv_test, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                if len(row) >= 2:
                    feat_path = row[0].strip()
                    lbl_path  = row[1].strip()
                    base = os.path.splitext(os.path.basename(feat_path))[0]
                else:
                    item = row[0].strip()
                    base = os.path.splitext(os.path.basename(item))[0]
                    feat_path = os.path.join(feature_dir, base + ".npy")
                    lbl_path  = os.path.join(label_dir,   base + ".npy")

                if os.path.isfile(feat_path) and os.path.isfile(lbl_path):
                    self.rows.append((feat_path, lbl_path, base))

        if not self.rows:
            raise ValueError(f"No valid samples found from {csv_test}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        feat_path, lbl_path, base = self.rows[idx]
        feat = _to_chw(np.load(feat_path)).astype(np.float32)
        lbl  = _to_chw(np.load(lbl_path)).astype(np.float32)
        feat = np.clip(feat, 0.0, 1.0)
        lbl  = np.clip(lbl,  0.0, 1.0)
        return feat, lbl, base, feat_path, lbl_path


# =============================================================================
# Metrics
# =============================================================================
_DRC_NZ_THRESH = 0.01
_DRC_HOTSPOT_FRAC = 0.01
_DRC_BIN_THR = 0.10


def drc_metrics(gt: np.ndarray, pr: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt, dtype=np.float32)
    pr = np.asarray(pr, dtype=np.float32)
    gt_flat = gt.ravel()
    pr_flat = pr.ravel()

    m: Dict[str, float] = {}

    for frac, key in [(0.01, "topk_1pct"), (0.005, "topk_05pct")]:
        k = max(1, int(frac * gt_flat.size))
        gt_idx = np.argsort(gt_flat)[-k:]
        pr_idx = np.argsort(pr_flat)[-k:]
        m[key] = float(len(np.intersect1d(gt_idx, pr_idx)) / k)

    nz = gt_flat > _DRC_NZ_THRESH
    if nz.any():
        m["nz_mae"] = float(np.mean(np.abs(gt_flat[nz] - pr_flat[nz])))
        nz_rmse = float(np.sqrt(np.mean((gt_flat[nz] - pr_flat[nz]) ** 2)))
        nz_rng = float(gt_flat[nz].max() - gt_flat[nz].min())
        m["nz_nrmse"] = (nz_rmse / nz_rng) if nz_rng > 1e-8 else float("nan")
        m["nz_pearson"] = _safe_pearson(gt_flat[nz], pr_flat[nz])
    else:
        m["nz_mae"] = float("nan")
        m["nz_nrmse"] = float("nan")
        m["nz_pearson"] = float("nan")

    k_hot = max(1, int(_DRC_HOTSPOT_FRAC * gt_flat.size))
    hot_idx = np.argsort(gt_flat)[-k_hot:]
    m["hotspot_mae"] = float(np.mean(np.abs(gt_flat[hot_idx] - pr_flat[hot_idx])))

    gt_pos = gt_flat >= _DRC_BIN_THR
    pr_pos = pr_flat >= _DRC_BIN_THR
    tp = float(np.sum(gt_pos & pr_pos))
    fp = float(np.sum((~gt_pos) & pr_pos))
    fn = float(np.sum(gt_pos & (~pr_pos)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0)
          else float("nan"))

    m["precision_at_0p10"] = prec
    m["recall_at_0p10"] = rec
    m["f1_at_0p10"] = f1

    m["global_mae"] = float(np.mean(np.abs(gt - pr)))
    rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
    rng = float(gt.max() - gt.min())
    m["global_nrmse"] = (rmse / rng) if rng > 1e-8 else float("nan")
    m["ssim"] = float(structural_similarity(gt, pr, data_range=1.0))
    m["global_pearson"] = _safe_pearson(gt_flat, pr_flat)

    return m


def congestion_metrics(gt: np.ndarray, pr: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt, dtype=np.float32)
    pr = np.asarray(pr, dtype=np.float32)

    mae = float(np.mean(np.abs(gt - pr)))
    rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
    rng = float(gt.max() - gt.min())
    nrms = (rmse / rng) if rng > 1e-8 else float("nan")
    ssim = float(structural_similarity(gt, pr, data_range=1.0))
    psnr = float(peak_signal_noise_ratio(gt, pr, data_range=1.0)) if rmse > 1e-10 else float("inf")
    pear = _safe_pearson(gt, pr)
    bias = float(np.mean(pr) - np.mean(gt))

    return {
        "mae": mae,
        "rmse": rmse,
        "nrms": nrms,
        "ssim": ssim,
        "psnr": psnr,
        "pearson": pear,
        "bias": bias,
    }


# =============================================================================
# Model loading and DDIM sampling
# =============================================================================
@dataclass
class LoadedLDM:
    net: torch.nn.Module
    feat_proj: torch.nn.Module
    ae: torch.nn.Module
    ck: Dict[str, Any]
    z_mean: torch.Tensor
    z_std: torch.Tensor
    alpha_bar: torch.Tensor
    pred_type: str
    latent_ch: int
    H_lat: int
    W_lat: int
    task: str
    sinusoidal_embedding: Any


def _instantiate_feature_projector(FeatureProjector, in_ch: int, out_ch: int, H_lat: int):
    sig = inspect.signature(FeatureProjector)
    kwargs = {}
    stride = max(1, 256 // int(H_lat))

    if "in_ch" in sig.parameters:
        kwargs["in_ch"] = in_ch
    if "out_ch" in sig.parameters:
        kwargs["out_ch"] = out_ch
    if "stride" in sig.parameters:
        kwargs["stride"] = stride
    if "downsample" in sig.parameters:
        kwargs["downsample"] = stride
    if "target_hw" in sig.parameters:
        kwargs["target_hw"] = H_lat
    if "out_hw" in sig.parameters:
        kwargs["out_hw"] = H_lat

    return FeatureProjector(**kwargs)


def _instantiate_unet(LatentUNet, in_ch: int, out_ch: int, base_channels: int):
    sig = inspect.signature(LatentUNet)
    kwargs = {}

    if "in_ch" in sig.parameters:
        kwargs["in_ch"] = in_ch
    if "out_ch" in sig.parameters:
        kwargs["out_ch"] = out_ch
    if "base" in sig.parameters:
        kwargs["base"] = base_channels
    if "base_channels" in sig.parameters:
        kwargs["base_channels"] = base_channels
    if "t_emb_dim" in sig.parameters:
        kwargs["t_emb_dim"] = 128
    if "dropout" in sig.parameters:
        kwargs["dropout"] = 0.0

    return LatentUNet(**kwargs)


def _load_autoencoder_from_ckpt(ck: Dict[str, Any], vae_dir: str, device: torch.device):
    model_type = ck.get("vae_model_type", ck.get("model_type", ""))
    ae_ckpt_path = ck.get("ae_ckpt", None)
    if ae_ckpt_path is None:
        raise ValueError("Diffusion checkpoint missing 'ae_ckpt'")

    ae_ck = torch.load(ae_ckpt_path, map_location=device)

    if model_type == "LabelVAE_v2":
        mod = _import_from_file("drc_vae_model", os.path.join(vae_dir, "vae_model.py"))
        model = mod.LabelVAE(
            C_label=int(ae_ck["C_label"]),
            latent_ch=int(ae_ck["latent_ch"]),
            base_ch=int(ae_ck.get("base_ch", 64)),
            log_scale=float(ae_ck.get("log_scale", 10.0)),
        ).to(device)

    elif model_type == "CongestionAE":
        mod = _import_from_file("cong_ae_model", os.path.join(vae_dir, "ae_model_congestion.py"))
        model = mod.CongestionAE(
            C_label=int(ae_ck["C_label"]),
            latent_ch=int(ae_ck["latent_ch"]),
            base_ch=int(ae_ck.get("base_ch", 64)),
        ).to(device)

    elif model_type == "CongestionVAE":
        mod = _import_from_file("cong_vae_model", os.path.join(vae_dir, "vae_model_congestion.py"))
        model = mod.CongestionVAE(
            C_label=int(ae_ck["C_label"]),
            latent_ch=int(ae_ck["latent_ch"]),
            base_ch=int(ae_ck.get("base_ch", 64)),
        ).to(device)

    else:
        raise ValueError(f"Unsupported autoencoder type: {model_type}")

    model.load_state_dict(ae_ck["net"], strict=True)

    if "ema" in ae_ck:
        ema_sd = ae_ck["ema"]
        try:
            model.load_state_dict(ema_sd, strict=True)
            print("[AE] EMA weights applied directly")
        except Exception:
            print("[AE] WARNING: could not apply AE/VAE EMA directly; using raw weights")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _try_apply_ldm_ema(net, feat_proj, ema_sd: Dict[str, Any]) -> None:
    applied_any = False

    # Most common layout: net.* and proj.* or feat_proj.*
    net_sd = {k[len("net."):]: v for k, v in ema_sd.items() if isinstance(k, str) and k.startswith("net.")}
    proj_sd = {k[len("proj."):]: v for k, v in ema_sd.items() if isinstance(k, str) and k.startswith("proj.")}
    feat_proj_sd = {k[len("feat_proj."):]: v for k, v in ema_sd.items() if isinstance(k, str) and k.startswith("feat_proj.")}

    if net_sd:
        net.load_state_dict(net_sd, strict=True)
        print("[LDM] EMA weights applied to UNet")
        applied_any = True
    if proj_sd:
        feat_proj.load_state_dict(proj_sd, strict=True)
        print("[LDM] EMA weights applied to projector")
        applied_any = True
    elif feat_proj_sd:
        feat_proj.load_state_dict(feat_proj_sd, strict=True)
        print("[LDM] EMA weights applied to feat_proj")
        applied_any = True

    if applied_any:
        return

    # Fallback: try direct match to net
    try:
        net.load_state_dict(ema_sd, strict=False)
        print("[LDM] WARNING: EMA loaded with non-strict fallback into UNet only")
        applied_any = True
    except Exception:
        pass

    if not applied_any:
        print("[LDM] WARNING: EMA present but not applied; using raw net/proj weights")


def load_ldm_checkpoint(task: str, ckpt_path: str, vae_dir: str, device: torch.device) -> LoadedLDM:
    models_mod = _import_from_file("ldm_models", os.path.join(SCRIPT_DIR, "models.py"))
    diffusion_mod = _import_from_file("ldm_diffusion", os.path.join(SCRIPT_DIR, "diffusion.py"))

    ck = torch.load(ckpt_path, map_location=device)

    ck_task = str(ck.get("task", task))
    if ck_task != task:
        print(f"[LDM] WARNING: requested task={task}, checkpoint says task={ck_task}. Using checkpoint task for logging.")

    latent_ch = int(ck["latent_ch"])
    H_lat = int(ck["H_latent"])
    W_lat = int(ck["W_latent"])
    pred_type = str(ck["pred_type"]).lower()
    in_ch = int(ck["in_ch"])
    C_feat = int(ck["C_feat"])
    feat_proj_ch = int(ck["feat_proj_ch"])
    base_channels = int(ck.get("base_channels", 64))

    z_mean = torch.tensor(ck["z_mean"], dtype=torch.float32, device=device).view(1, latent_ch, 1, 1)
    z_std  = torch.tensor(ck["z_std"],  dtype=torch.float32, device=device).view(1, latent_ch, 1, 1)

    feat_proj = _instantiate_feature_projector(models_mod.FeatureProjector, C_feat, feat_proj_ch, H_lat).to(device)
    feat_proj.load_state_dict(ck["feat_proj"], strict=True)

    net = _instantiate_unet(models_mod.LatentUNet, in_ch, latent_ch, base_channels).to(device)
    net.load_state_dict(ck["net"], strict=True)

    if "ema" in ck and isinstance(ck["ema"], dict):
        _try_apply_ldm_ema(net, feat_proj, ck["ema"])

    net.eval()
    feat_proj.eval()
    for p in net.parameters():
        p.requires_grad = False
    for p in feat_proj.parameters():
        p.requires_grad = False

    ae = _load_autoencoder_from_ckpt(ck, vae_dir, device)

    T = int(ck["diffusion_steps"])
    betas_np = diffusion_mod.build_betas(T, str(ck["beta_schedule"]))
    alpha_bar = torch.tensor(np.cumprod(1.0 - betas_np), dtype=torch.float32, device=device)

    return LoadedLDM(
        net=net,
        feat_proj=feat_proj,
        ae=ae,
        ck=ck,
        z_mean=z_mean,
        z_std=z_std,
        alpha_bar=alpha_bar,
        pred_type=pred_type,
        latent_ch=latent_ch,
        H_lat=H_lat,
        W_lat=W_lat,
        task=ck_task,
        sinusoidal_embedding=models_mod.sinusoidal_embedding,
    )


def denormalize(z_norm: torch.Tensor, z_mean: torch.Tensor, z_std: torch.Tensor) -> torch.Tensor:
    return z_norm * z_std + z_mean


@torch.no_grad()
def ddim_sample_one(
    ldm: LoadedLDM,
    feat: torch.Tensor,
    x_init: torch.Tensor,
    steps: int,
    eta: float,
    cfg_scale: float,
) -> torch.Tensor:
    net, feat_proj, ae = ldm.net, ldm.feat_proj, ldm.ae
    alpha_bar = ldm.alpha_bar
    pred_type = ldm.pred_type
    B = feat.shape[0]
    device = feat.device
    T = alpha_bar.shape[0]

    cond = feat_proj(feat)
    null_cond = torch.zeros_like(cond)
    z = x_init
    ts = make_ddim_ts(T, steps, device)

    for i in range(steps):
        t = ts[i].expand(B)
        t_emb = ldm.sinusoidal_embedding(t, 128)
        ab_t = alpha_bar[t].view(B, 1, 1, 1)

        if cfg_scale > 0.0:
            x_in_u = torch.cat([z, null_cond], dim=1)
            x_in_c = torch.cat([z, cond], dim=1)
            pred_u = net(x_in_u, t_emb)
            pred_c = net(x_in_c, t_emb)
            pred = pred_u + cfg_scale * (pred_c - pred_u)
        else:
            x_in = torch.cat([z, cond], dim=1)
            pred = net(x_in, t_emb)

        if pred_type == "eps":
            z0 = (z - torch.sqrt(1.0 - ab_t) * pred) / torch.sqrt(ab_t + 1e-12)
            eps = pred
        else:
            z0 = torch.sqrt(ab_t) * z - torch.sqrt(1.0 - ab_t) * pred
            eps = torch.sqrt(1.0 - ab_t) * z + torch.sqrt(ab_t) * pred

        if i == steps - 1:
            z = z0
            break

        t_next = ts[i + 1].expand(B)
        ab_next = alpha_bar[t_next].view(B, 1, 1, 1)
        sigma = (
            eta
            * torch.sqrt((1 - ab_next) / (1 - ab_t + 1e-12))
            * torch.sqrt(torch.clamp(1 - ab_t / (ab_next + 1e-12), min=0.0))
        )
        noise = torch.randn_like(z) if eta > 0 else 0.0
        z = (
            torch.sqrt(ab_next) * z0
            + torch.sqrt(torch.clamp(1 - ab_next - sigma ** 2, min=0.0)) * eps
            + sigma * noise
        )

    z_raw = denormalize(z, ldm.z_mean, ldm.z_std)
    return ae.decode_from_z(z_raw)


# =============================================================================
# Pretty metric printing
# =============================================================================
def print_drc_metrics(metrics: Dict[str, float], prefix: str = "    "):
    print(f"{prefix}TopK@1%        : {metrics['topk_1pct']:.4f}")
    print(f"{prefix}TopK@0.5%      : {metrics['topk_05pct']:.4f}")
    print(f"{prefix}NZ-MAE         : {metrics['nz_mae']:.5f}")
    print(f"{prefix}NZ-NRMSE       : {metrics['nz_nrmse']:.5f}")
    print(f"{prefix}NZ-Pearson     : {metrics['nz_pearson']:.4f}")
    print(f"{prefix}Hotspot-MAE    : {metrics['hotspot_mae']:.5f}")
    print(f"{prefix}Precision@0.10 : {metrics['precision_at_0p10']:.4f}")
    print(f"{prefix}Recall@0.10    : {metrics['recall_at_0p10']:.4f}")
    print(f"{prefix}F1@0.10        : {metrics['f1_at_0p10']:.4f}")
    print(f"{prefix}Global-MAE     : {metrics['global_mae']:.5f}")
    print(f"{prefix}Global-NRMSE   : {metrics['global_nrmse']:.5f}")
    print(f"{prefix}Global-Pearson : {metrics['global_pearson']:.4f}")
    print(f"{prefix}SSIM           : {metrics['ssim']:.4f}")


def print_cong_metrics(metrics: Dict[str, float], prefix: str = "    "):
    print(f"{prefix}MAE            : {metrics['mae']:.5f}")
    print(f"{prefix}RMSE           : {metrics['rmse']:.5f}")
    print(f"{prefix}NRMSE          : {metrics['nrms']:.5f}")
    print(f"{prefix}SSIM           : {metrics['ssim']:.4f}")
    print(f"{prefix}PSNR           : {metrics['psnr']:.4f}")
    print(f"{prefix}Pearson        : {metrics['pearson']:.4f}")
    print(f"{prefix}Bias           : {metrics['bias']:+.5f}")


# =============================================================================
# Figure creation
# =============================================================================
def _display_limits(task: str, gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """
    Display-only scaling.
    Does not affect metrics.
    """
    combined = np.concatenate([gt.ravel(), pred.ravel()])

    if task == "Congestion":
        vmin = float(np.percentile(combined, 1.0))
        vmax = float(np.percentile(combined, 99.5))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.min(combined))
            vmax = float(np.max(combined))
        if vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        return vmin, vmax

    # DRC: sparse; keep zero anchored, but avoid a single outlier dominating contrast
    vmax = float(np.percentile(combined, 99.8))
    vmax = max(vmax, 0.05)
    vmax = min(vmax, 1.0)
    return 0.0, vmax


def build_figure(
    task: str,
    ckpt_path: str,
    out_pdf: str,
    vae_dir: str,
    cond_rgb: Tuple[int, int, int],
    n_rows: int,
    sample_seed: Optional[int],
    ddim_steps: int,
    eta: float,
    cfg_scale: float,
    draws: int,
    device: torch.device,
):
    task_cfg = TASK_CFG[task]
    ds = FigureDataset(task_cfg["feature_dir"], task_cfg["label_dir"], task_cfg["csv_test"])
    ldm = load_ldm_checkpoint(task, ckpt_path, vae_dir, device)

    if sample_seed is None:
        rng = np.random.default_rng()
        print("[Figure] Sample selection: random (different each run)")
    else:
        rng = np.random.default_rng(sample_seed)
        print(f"[Figure] Sample selection seed: {sample_seed}")

    idxs = rng.choice(len(ds), size=min(n_rows, len(ds)), replace=False)

    fig = plt.figure(figsize=(8.6, 6.1))
    gs = gridspec.GridSpec(len(idxs), 4, figure=fig)
    plt.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.04, wspace=0.03, hspace=0.08)

    col_titles = ["Initial noise", "Conditioning", "Generated", "Ground truth"]
    chosen: List[Dict[str, Any]] = []

    for r, ds_idx in enumerate(idxs):
        feat_np, lbl_np, name, feat_path, lbl_path = ds[int(ds_idx)]
        gt = lbl_np[0]

        feat_t = torch.from_numpy(feat_np[None]).float().to(device)
        names = [name]

        gens = []
        first_noise_np = None

        for k in range(draws):
            seed_for_draw = sample_seed if sample_seed is not None else int(np.random.SeedSequence().entropy)
            x_init = make_initial_noise(
                1, ldm.latent_ch, ldm.H_lat, ldm.W_lat, device, seed_for_draw, names, k
            )
            if k == 0:
                first_noise_np = x_init[0].detach().cpu().numpy()
            gen = ddim_sample_one(ldm, feat_t, x_init, steps=ddim_steps, eta=eta, cfg_scale=cfg_scale)
            gens.append(gen)

        G = torch.stack(gens, dim=0)
        pred = G.mean(dim=0)[0, 0].detach().cpu().numpy().clip(0.0, 1.0)

        metrics = drc_metrics(gt, pred) if task == "DRC" else congestion_metrics(gt, pred)
        gt_stats = describe_array(gt)
        pr_stats = describe_array(pred)

        chosen.append({
            "name": name,
            "feature_path": feat_path,
            "label_path": lbl_path,
            "metrics": metrics,
            "gt_stats": gt_stats,
            "pred_stats": pr_stats,
        })

        noise_img = latent_noise_to_img(first_noise_np)
        cond_img = feature_rgb(feat_np, cond_rgb)
        vmin, vmax = _display_limits(task, gt, pred)

        ax0 = fig.add_subplot(gs[r, 0])
        ax1 = fig.add_subplot(gs[r, 1])
        ax2 = fig.add_subplot(gs[r, 2])
        ax3 = fig.add_subplot(gs[r, 3])

        ax0.imshow(noise_img, cmap="gray", interpolation="nearest")
        ax1.imshow(cond_img, interpolation="nearest")
        ax2.imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax3.imshow(gt,   cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")

        for c, ax in enumerate([ax0, ax1, ax2, ax3]):
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, pad=6)

    fig.suptitle(f"{task} latent diffusion visual validation", fontsize=11, y=0.975)

    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02, format="pdf")
    plt.close(fig)

    sidecar = {
        "task": task,
        "ckpt": ckpt_path,
        "rows": chosen,
        "cond_rgb": list(cond_rgb),
        "ddim_steps": ddim_steps,
        "eta": eta,
        "cfg_scale": cfg_scale,
        "draws": draws,
        "sample_seed": sample_seed,
        "sample_selection": "random" if sample_seed is None else "seeded",
    }
    sidecar_path = os.path.splitext(out_pdf)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[Saved] figure   -> {out_pdf}")
    print(f"[Saved] metadata -> {sidecar_path}")
    print(f"[Figure] Selected samples and metrics:")

    for i, row in enumerate(chosen, 1):
        print(f"\n  Row {i}: {row['name']}")
        print(f"    feature_path: {row['feature_path']}")
        print(f"    label_path  : {row['label_path']}")

        print("    GT stats    : "
              f"min={row['gt_stats']['min']:.4f} "
              f"p1={row['gt_stats']['p1']:.4f} "
              f"p50={row['gt_stats']['p50']:.4f} "
              f"p99={row['gt_stats']['p99']:.4f} "
              f"max={row['gt_stats']['max']:.4f} "
              f"mean={row['gt_stats']['mean']:.4f}")
        print("    Pred stats  : "
              f"min={row['pred_stats']['min']:.4f} "
              f"p1={row['pred_stats']['p1']:.4f} "
              f"p50={row['pred_stats']['p50']:.4f} "
              f"p99={row['pred_stats']['p99']:.4f} "
              f"max={row['pred_stats']['max']:.4f} "
              f"mean={row['pred_stats']['mean']:.4f}")

        if task == "DRC":
            print_drc_metrics(row["metrics"], prefix="    ")
        else:
            print_cong_metrics(row["metrics"], prefix="    ")


# =============================================================================
# CLI
# =============================================================================
def build_args():
    p = argparse.ArgumentParser("Make DRC/Congestion visual comparison figure")
    p.add_argument("--task", required=True, choices=["DRC", "Congestion"])
    p.add_argument("--ckpt", required=True, help="Path to LDM checkpoint (best_gen.pt or latest.pt)")
    p.add_argument("--out", required=True, help="Output PDF path")
    p.add_argument("--vae_dir", default=None, help="Optional override for AE/VAE model directory")
    p.add_argument("--rows", type=int, default=3, help="Number of figure rows / samples")
    p.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Seed for sample selection. Default: None = different random rows each run."
    )
    p.add_argument("--steps", type=int, default=100, help="DDIM steps")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--N", type=int, default=8, help="Number of draws averaged for generated image")
    p.add_argument("--cond_rgb", nargs=3, type=int, default=None,
                   help="Three feature-channel indices to visualize as RGB")
    return p.parse_args()


def main():
    args = build_args()
    task_cfg = TASK_CFG[args.task]
    vae_dir = args.vae_dir or task_cfg["default_vae_dir"]
    cond_rgb = tuple(args.cond_rgb) if args.cond_rgb is not None else task_cfg["default_cond_rgb"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    build_figure(
        task=args.task,
        ckpt_path=args.ckpt,
        out_pdf=args.out,
        vae_dir=vae_dir,
        cond_rgb=cond_rgb,
        n_rows=args.rows,
        sample_seed=args.sample_seed,
        ddim_steps=args.steps,
        eta=args.eta,
        cfg_scale=args.cfg_scale,
        draws=args.N,
        device=device,
    )


if __name__ == "__main__":
    main()