#!/usr/bin/env python3
"""
make_ldm_visual_figure.py

4-column figure: [Initial Noise | Routing Features (RGB) | Generated | Ground Truth]
3 rows = 3 random test samples, one generation per sample.

Metrics and design names printed to terminal only — figure is clean.

Usage:
  # DRC ControlNet (default cond channels 0,1,2)
  python make_ldm_visual_figure.py \
    --task DRC \
    --ckpt ./runs/ldm_DRC_control/best_gen.pt \
    --out  ./results/figures/drc_control_visual.pdf \
    --sample_seed 42

  # Override conditioning channels
 python make_ldm_visual_figure.py \
    --task DRC \
    --ckpt ./runs/ldm_DRC_control/best_gen.pt \
    --out  ./results/figures/drc_control_visual.pdf \
    --sample_seed 42 \
    --cond_rgb 8 9 1
"""
from __future__ import annotations

import os
import csv
import json
import hashlib
import argparse
import importlib.util
from dataclasses import dataclass
from typing import List, Tuple, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded VAE paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VAE_DIRS = {
    "LabelVAE_v2":      "/data2/kgt22001/cong_gen/gen_additional/drc_vae",
    "CongestionVAE_v2": "/data2/kgt22001/cong_gen/gen_additional/conegestion_vae",
    "CongestionVAE":    "/data2/kgt22001/cong_gen/gen_additional/conegestion_vae",
}

TASK_CFG = {
    "DRC": {
        "feature_dir": ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                        "/DRC/feature"),
        "label_dir":   ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                        "/DRC/label"),
        "csv_test":    ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                        "/DRC/files_design/test_N28.csv"),
    },
    "Congestion": {
        "feature_dir": ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                        "/congestion/feature"),
        "label_dir":   ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                        "/congestion/label"),
        "csv_test":    ("/data2/kgt22001/CircuitNet-N28/training_set_expanded"
                        "/congestion/files_design/test_N28.csv"),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    8,
    "figure.dpi":   300,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_chw(arr):
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        if arr.shape[2] < arr.shape[0]:   # (H,W,C) → (C,H,W)
            return arr.transpose(2, 0, 1)
        if arr.shape[0] < arr.shape[1]:   # already (C,H,W)
            return arr
    raise ValueError(f"Cannot parse shape {arr.shape}")


def _norm(x, lo=1, hi=99):
    """Robust percentile normalisation to [0,1]."""
    x  = np.asarray(x, np.float32)
    lo = np.percentile(x, lo)
    hi = np.percentile(x, hi)
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _stable_seed(name):
    return int.from_bytes(
        hashlib.sha1(name.encode()).digest()[:4],
        byteorder="little", signed=False)


def _make_noise(B, C, H, W, device, base_seed, names, draw_k):
    x = torch.empty((B, C, H, W), device=device)
    for i, nm in enumerate(names):
        s = int(base_seed) ^ _stable_seed(str(nm)) ^ (int(draw_k) * 0x9E3779B1)
        g = torch.Generator(device=device)
        g.manual_seed(s & 0x7FFFFFFF)
        x[i:i+1] = torch.randn((1, C, H, W), device=device, generator=g)
    return x


def _make_ts(T, steps, device):
    ts = np.rint(np.linspace(T-1, 0, steps, dtype=np.float64)).astype(np.int64)
    for i in range(1, len(ts)):
        if ts[i] >= ts[i-1]:
            ts[i] = ts[i-1] - 1
    return torch.tensor(np.clip(ts, 0, T-1), device=device, dtype=torch.long)


def _drop_channels(feat_raw, drop_ch):
    """Remove drop_ch indices from (C,H,W) array."""
    keep = [c for c in range(feat_raw.shape[0]) if c not in drop_ch]
    return feat_raw[keep]


def _feature_rgb(feat_raw, cond_rgb):
    """
    Build RGB conditioning image.

    feat_raw : (C_raw, H, W) — ALL raw channels before any dropping.
    cond_rgb : tuple of 3 raw channel indices → R, G, B.
               Each channel independently normalised to [0,1].
               Clamps index to valid range silently.
    """
    rgb = []
    C   = feat_raw.shape[0]
    for c in cond_rgb:
        c = int(np.clip(c, 0, C - 1))
        rgb.append(_norm(feat_raw[c]))
    return np.stack(rgb, axis=-1)


def _display_limits(task, gt, pred):
    combined = np.concatenate([gt.ravel(), pred.ravel()])
    if task == "Congestion":
        vmin = float(np.percentile(combined, 1.0))
        vmax = float(np.percentile(combined, 99.5))
    else:
        vmin = 0.0
        vmax = float(np.percentile(combined, 99.8))
        vmax = max(vmax, 0.05)
    return vmin, vmax


def _safe_pearson(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(task, gt, pred):
    if task == "DRC":
        k      = max(1, int(0.01 * gt.size))
        gt_idx = np.argsort(gt.ravel())[-k:]
        pr_idx = np.argsort(pred.ravel())[-k:]
        topk   = float(len(np.intersect1d(gt_idx, pr_idx)) / k)
        nz     = gt.ravel() > 0.01
        nzmae  = (float(np.mean(np.abs(gt.ravel()[nz] - pred.ravel()[nz])))
                  if nz.any() else float("nan"))
        gt_pos = gt.ravel() >= 0.10
        pr_pos = pred.ravel() >= 0.10
        tp = float(np.sum(gt_pos & pr_pos))
        fp = float(np.sum((~gt_pos) & pr_pos))
        fn = float(np.sum(gt_pos & (~pr_pos)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)
                if not (np.isnan(prec) or np.isnan(rec)) and (prec + rec) > 0
                else float("nan"))
        return {"topk_1pct": topk,
                "f1_at_0.1": f1,
                "recall_at_0.1": rec,
                "nz_mae": nzmae,
                "mae": float(np.mean(np.abs(gt - pred)))}
    else:
        return {"mae":     float(np.mean(np.abs(gt - pred))),
                "ssim":    float(__import__("skimage.metrics",
                                            fromlist=["structural_similarity"])
                                 .structural_similarity(gt, pred,
                                                        data_range=1.0)),
                "pearson": _safe_pearson(gt, pred),
                "bias":    float(pred.mean() - gt.mean())}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset:
    def __init__(self, feature_dir, label_dir, csv_test):
        self.rows = []
        with open(csv_test, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                if len(row) >= 2:
                    fp   = row[0].strip()
                    lp   = row[1].strip()
                    base = os.path.splitext(os.path.basename(fp))[0]
                else:
                    base = os.path.splitext(
                        os.path.basename(row[0].strip()))[0]
                    fp   = os.path.join(feature_dir, base + ".npy")
                    lp   = os.path.join(label_dir,   base + ".npy")
                if os.path.isfile(fp) and os.path.isfile(lp):
                    self.rows.append((fp, lp, base))
        print(f"[Data] {len(self.rows)} test samples found")

    def __len__(self):
        return len(self.rows)

    def get(self, idx):
        fp, lp, base = self.rows[idx]
        feat_raw = np.clip(
            _to_chw(np.load(fp).astype(np.float32)), 0.0, 1.0)
        lbl = np.clip(
            _to_chw(np.load(lp).astype(np.float32)), 0.0, 1.0)
        return feat_raw, lbl, base


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — handles both unified and control checkpoints
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LDM:
    net:       torch.nn.Module
    cond:      torch.nn.Module
    ae:        torch.nn.Module
    z_mean:    torch.Tensor
    z_std:     torch.Tensor
    alpha_bar: torch.Tensor
    pred_type: str
    proj_type: str
    latent_ch: int
    H_lat:     int
    W_lat:     int
    task:      str
    drop_ch:   List[int]
    cfg_drop:  float
    sin_emb:   Any
    T:         int


def _load_ae(ck, device):
    vae_type = ck.get("vae_type", ck.get("model_type", ""))
    ae_path  = ck["ae_ckpt"]
    vae_dir  = VAE_DIRS.get(vae_type, ck.get("vae_dir", ""))
    ae_ck    = torch.load(ae_path, map_location=device)

    if vae_type == "LabelVAE_v2":
        mod   = _import("drc_vae",
                        os.path.join(vae_dir, "vae_model.py"))
        model = mod.LabelVAE(
            C_label   = int(ae_ck["C_label"]),
            latent_ch = int(ae_ck["latent_ch"]),
            base_ch   = int(ae_ck.get("base_ch", 64)),
            log_scale = float(ae_ck.get("log_scale", 10.0)),
        ).to(device)
    elif vae_type in ("CongestionVAE_v2", "CongestionVAE"):
        mod   = _import("cong_vae",
                        os.path.join(vae_dir, "vae_model_congestion.py"))
        model = mod.CongestionVAE(
            C_label    = int(ae_ck["C_label"]),
            latent_ch  = int(ae_ck["latent_ch"]),
            base_ch    = int(ae_ck.get("base_ch", 64)),
            logvar_min = float(ae_ck.get("logvar_min", -2.0)),
            logvar_max = float(ae_ck.get("logvar_max",  2.0)),
        ).to(device)
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")

    model.load_state_dict(ae_ck["net"], strict=True)
    if "ema" in ae_ck:
        sd = ae_ck["ema"]
        if isinstance(sd, dict) and "shadow" in sd:
            sd = sd["shadow"]
        try:
            model.load_state_dict(sd, strict=True)
            print("[AE] EMA applied")
        except Exception:
            pass
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _apply_ema(module, ema_sd, prefix):
    sd = {k[len(prefix):]: v for k, v in ema_sd.items()
          if k.startswith(prefix)}
    if sd:
        module.load_state_dict(sd, strict=True)
        print(f"[LDM] EMA applied ({prefix.rstrip('.')})")
        return True
    return False


def load_ldm(ckpt_path, device):
    ck        = torch.load(ckpt_path, map_location=device)
    proj_type = str(ck.get("proj_type", "single"))
    task      = str(ck["task"])
    latent_ch = int(ck["latent_ch"])
    H_lat     = int(ck["H_latent"])
    W_lat     = int(ck["W_latent"])
    base_ch   = int(ck.get("base_channels", 128))
    C_feat    = int(ck["C_feat"])
    drop_ch   = list(ck.get("drop_feat_ch", []))
    cfg_drop  = float(ck.get("cfg_drop_prob", 0.0))
    T         = int(ck["diffusion_steps"])

    z_mean = torch.tensor(ck["z_mean"], dtype=torch.float32,
                          device=device).view(1, latent_ch, 1, 1)
    z_std  = torch.tensor(ck["z_std"],  dtype=torch.float32,
                          device=device).view(1, latent_ch, 1, 1)

    models_mod = _import("ldm_models",
                         os.path.join(SCRIPT_DIR, "models.py"))
    diff_mod   = _import("ldm_diff",
                         os.path.join(SCRIPT_DIR, "diffusion.py"))

    # Unwrap EMA state dict
    ema_raw = ck.get("ema", {})
    ema_sd  = (ema_raw["shadow"]
               if isinstance(ema_raw, dict) and "shadow" in ema_raw
               else ema_raw)

    if proj_type == "multiscale_controlnet":
        net = models_mod.LatentUNet(
            in_ch=latent_ch, out_ch=latent_ch,
            base=base_ch, t_emb_dim=128, dropout=0.0,
        ).to(device)
        cond = models_mod.MultiScaleConditioner(
            in_ch=C_feat, base=base_ch,
        ).to(device)
        if not _apply_ema(net,  ema_sd, "net."):
            net.load_state_dict(ck["net"], strict=True)
        if not _apply_ema(cond, ema_sd, "cond."):
            cond.load_state_dict(ck["conditioner"], strict=True)

    else:  # unified / single
        feat_proj_ch = int(ck["feat_proj_ch"])
        in_ch_unet   = int(ck["in_ch"])
        net = models_mod.LatentUNet(
            in_ch=in_ch_unet, out_ch=latent_ch,
            base=base_ch, t_emb_dim=128, dropout=0.0,
        ).to(device)
        cond = models_mod.FeatureProjector(
            in_ch=C_feat, out_ch=feat_proj_ch,
            stride=256 // H_lat,
        ).to(device)
        if not _apply_ema(net, ema_sd, "net."):
            net.load_state_dict(ck["net"], strict=True)
        if (not _apply_ema(cond, ema_sd, "proj.") and
                not _apply_ema(cond, ema_sd, "feat_proj.")):
            cond.load_state_dict(ck["feat_proj"], strict=True)

    net.eval();  cond.eval()
    for p in list(net.parameters()) + list(cond.parameters()):
        p.requires_grad = False

    ae = _load_ae(ck, device)

    betas     = diff_mod.build_betas(T, str(ck["beta_schedule"]))
    alpha_bar = torch.tensor(np.cumprod(1.0 - betas),
                             dtype=torch.float32, device=device)

    print(f"[LDM] task={task}  proj={proj_type}  epoch={ck['epoch']}")
    print(f"[LDM] latent={latent_ch}x{H_lat}x{W_lat}  "
          f"C_feat={C_feat}  drop={drop_ch}  cfg_drop={cfg_drop}")

    return LDM(
        net=net, cond=cond, ae=ae,
        z_mean=z_mean, z_std=z_std, alpha_bar=alpha_bar,
        pred_type=str(ck["pred_type"]).lower(),
        proj_type=proj_type,
        latent_ch=latent_ch, H_lat=H_lat, W_lat=W_lat,
        task=task, drop_ch=drop_ch, cfg_drop=cfg_drop,
        sin_emb=models_mod.sinusoidal_embedding, T=T,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DDIM sampling — handles both unified and control
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_one(ldm, feat_dropped, x_init, steps, cfg_scale):
    """
    feat_dropped : (1, C_feat, H, W) — channels already dropped.
    cfg_scale    : only applied when ldm.cfg_drop > 0.
    """
    B      = feat_dropped.shape[0]
    device = feat_dropped.device
    ts     = _make_ts(ldm.T, steps, device)
    z      = x_init
    use_cfg = cfg_scale > 0.0 and ldm.cfg_drop > 0.0

    # Pre-compute conditioning
    if ldm.proj_type == "multiscale_controlnet":
        c64, c32, c16 = ldm.cond(feat_dropped)
    else:
        cond_feat = ldm.cond(feat_dropped)
        null_feat = torch.zeros_like(cond_feat)

    for i, t_val in enumerate(ts):
        t     = t_val.expand(B)
        t_emb = ldm.sin_emb(t, 128)
        ab_t  = ldm.alpha_bar[t].view(B, 1, 1, 1)

        if ldm.proj_type == "multiscale_controlnet":
            pred_c = ldm.net(z, t_emb, c64=c64,  c32=c32,  c16=c16)
            if use_cfg:
                pred_u = ldm.net(z, t_emb,
                                 c64=None, c32=None, c16=None)
                pred   = pred_u + cfg_scale * (pred_c - pred_u)
            else:
                pred = pred_c
        else:
            pred_c = ldm.net(torch.cat([z, cond_feat], 1), t_emb)
            if use_cfg:
                pred_u = ldm.net(torch.cat([z, null_feat], 1), t_emb)
                pred   = pred_u + cfg_scale * (pred_c - pred_u)
            else:
                pred = pred_c

        if ldm.pred_type == "v":
            z0  = (torch.sqrt(ab_t) * z
                   - torch.sqrt(1.0 - ab_t) * pred)
            eps = (torch.sqrt(1.0 - ab_t) * z
                   + torch.sqrt(ab_t) * pred)
        else:
            z0  = ((z - torch.sqrt(1.0 - ab_t) * pred)
                   / torch.sqrt(ab_t + 1e-12))
            eps = pred

        if i == len(ts) - 1:
            z = z0; break

        t_next  = ts[i+1].expand(B)
        ab_next = ldm.alpha_bar[t_next].view(B, 1, 1, 1)
        z = (torch.sqrt(ab_next) * z0
             + torch.sqrt(torch.clamp(1.0 - ab_next, min=0.0)) * eps)

    z_raw = z * ldm.z_std + ldm.z_mean
    return ldm.ae.decode_from_z(z_raw)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
def build_figure(ldm, ds, out_pdf, cond_rgb,
                 n_rows, sample_seed, steps, cfg_scale, device):

    rng   = np.random.default_rng(sample_seed)
    idxs  = rng.choice(len(ds), size=min(n_rows, len(ds)), replace=False)
    print(f"\n[Figure] sample_seed={sample_seed}  "
          f"indices={idxs.tolist()}  cond_rgb={cond_rgb}")

    cfg_eff = cfg_scale if ldm.cfg_drop > 0.0 else 0.0
    if cfg_eff != cfg_scale:
        print(f"[Figure] cfg_drop=0 → cfg_scale forced to 0.0")

    # 4 columns: Noise | Features | Generated | Ground Truth
    fig = plt.figure(figsize=(7.16, 2.2 * n_rows))
    gs  = gridspec.GridSpec(
        n_rows, 4, figure=fig,
        left=0.01, right=0.99,
        top=0.93,  bottom=0.01,
        wspace=0.03, hspace=0.08,
    )
    col_titles = ["Initial noise",
                  f"Features (ch {cond_rgb[0]},{cond_rgb[1]},{cond_rgb[2]})",
                  "Generated",
                  "Ground truth"]

    log_data = []

    for r, idx in enumerate(idxs):
        feat_raw, lbl_np, name = ds.get(int(idx))
        gt = lbl_np[0]   # (H,W)

        # Drop channels for model — AFTER building RGB image
        feat_dropped = _drop_channels(feat_raw, ldm.drop_ch)
        feat_t       = torch.from_numpy(
            feat_dropped[None]).float().to(device)

        # Initial noise — same seed as sampling so it matches generation
        x_init    = _make_noise(1, ldm.latent_ch, ldm.H_lat, ldm.W_lat,
                                device, sample_seed, [name], 0)
        noise_img = _norm(x_init[0].mean(0).cpu().numpy())

        # Generate
        gen  = sample_one(ldm, feat_t, x_init, steps, cfg_eff)
        pred = gen[0, 0].cpu().numpy().clip(0.0, 1.0)

        # Conditioning RGB from RAW features (before channel drop)
        cond_img = _feature_rgb(feat_raw, cond_rgb)

        vmin, vmax = _display_limits(ldm.task, gt, pred)

        ax0 = fig.add_subplot(gs[r, 0])
        ax1 = fig.add_subplot(gs[r, 1])
        ax2 = fig.add_subplot(gs[r, 2])
        ax3 = fig.add_subplot(gs[r, 3])

        ax0.imshow(noise_img, cmap="gray",
                   interpolation="nearest")
        ax1.imshow(cond_img,
                   interpolation="nearest")
        ax2.imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
        ax3.imshow(gt,   cmap="viridis", vmin=vmin, vmax=vmax,
                   interpolation="nearest")

        for ax in [ax0, ax1, ax2, ax3]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        if r == 0:
            for ax, title in zip([ax0, ax1, ax2, ax3], col_titles):
                ax.set_title(title, pad=4)

        # Print metrics to terminal — not in figure
        m = _metrics(ldm.task, gt, pred)
        log_data.append({"idx": int(idx), "name": name, "metrics": m})

        print(f"\n  Row {r+1}: {name}")
        print(f"    gt   — mean={gt.mean():.4f}  max={gt.max():.4f}  "
              f"nz_frac={float(np.mean(gt > 0.01)):.3f}")
        print(f"    pred — mean={pred.mean():.4f}  max={pred.max():.4f}")
        print(f"    feat — raw_channels={feat_raw.shape[0]}  "
              f"after_drop={feat_dropped.shape[0]}  "
              f"drop={ldm.drop_ch}")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"\n[Saved] {out_pdf}")

    sidecar = out_pdf.replace(".pdf", ".json")
    with open(sidecar, "w") as f:
        json.dump({
            "task":        ldm.task,
            "proj_type":   ldm.proj_type,
            "sample_seed": sample_seed,
            "cond_rgb":    list(cond_rgb),
            "indices":     idxs.tolist(),
            "rows":        log_data,
        }, f, indent=2)
    print(f"[Saved] {sidecar}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser(
        description="LDM visual figure — 4 columns, 3 rows")
    p.add_argument("--task",        required=True,
                   choices=["DRC", "Congestion"])
    p.add_argument("--ckpt",        required=True,
                   help="Path to best_gen.pt checkpoint")
    p.add_argument("--out",         required=True,
                   help="Output PDF path")
    p.add_argument("--rows",        type=int,   default=3,
                   help="Number of sample rows (default: 3)")
    p.add_argument("--sample_seed", type=int,   default=42,
                   help="Seed for sample selection (default: 42)")
    p.add_argument("--steps",       type=int,   default=100,
                   help="DDIM steps (default: 100)")
    p.add_argument("--cfg_scale",   type=float, default=1.5,
                   help="CFG scale — ignored if cfg_drop=0 (default: 1.5)")
    p.add_argument("--cond_rgb",    nargs=3,    type=int,
                   default=[0, 1, 2],
                   metavar=("R", "G", "B"),
                   help="Raw feature channel indices for RGB conditioning "
                        "image (default: 0 1 2)")
    return p.parse_args()


def main():
    args   = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg   = TASK_CFG[args.task]
    cond_rgb = tuple(args.cond_rgb)

    print(f"[Config] task={args.task}")
    print(f"[Config] ckpt={args.ckpt}")
    print(f"[Config] out={args.out}")
    print(f"[Config] rows={args.rows}  seed={args.sample_seed}  "
          f"steps={args.steps}  cfg_scale={args.cfg_scale}")
    print(f"[Config] cond_rgb={cond_rgb}  (raw feature channel indices)")

    ds  = TestDataset(
        tcfg["feature_dir"], tcfg["label_dir"], tcfg["csv_test"])
    ldm = load_ldm(args.ckpt, device)

    if ldm.task != args.task:
        print(f"[WARNING] --task={args.task} but checkpoint says "
              f"task={ldm.task}")

    build_figure(
        ldm=ldm, ds=ds,
        out_pdf=args.out,
        cond_rgb=cond_rgb,
        n_rows=args.rows,
        sample_seed=args.sample_seed,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        device=device,
    )


if __name__ == "__main__":
    main()