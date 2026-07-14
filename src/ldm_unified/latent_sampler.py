#!/usr/bin/env python3
"""
latent_sampler.py — Inference and evaluation for trained LDM.

Evaluation protocol aligned with the final paper tables.

Table 1 — Standard fidelity metrics (all tasks, all methods):
    MAE, NRMS, SSIM, Pearson

Table 2a — DRC task-aligned metrics:
    TopK@1%, TopK@0.5%, Hotspot-MAE, NZ-Pearson, F1@0.1, Unc.-Error Corr

Table 2b — Congestion task-aligned metrics:
    Hotspot-MAE, NZ-Pearson, Spatial Bias, Unc.-Error Corr

Notes:
- DRC Hotspot-MAE uses top 1% GT pixels by value.
- Congestion Hotspot-MAE uses top 5% GT pixels by value.
- NZ-Pearson is computed on active pixels with GT > 0.01.
- F1 threshold is fixed at 0.10 in normalized label space.
- Unc.-Error Corr is Pearson correlation between per-pixel variance across draws
  and per-pixel absolute error of the mean prediction.

Examples:
  python latent_sampler.py \
    --ckpt ./runs/ldm_DRC_unified_v2/best_gen.pt \
    --split test \
    --out_dir ./results/ldm_DRC_unified_v2_test \
    --steps 100 --N 8 --seeds 1234 2345 3456

  python latent_sampler.py \
    --ckpt ./runs/ldm_Cong_unified_v2/best_gen.pt \
    --split test \
    --out_dir ./results/ldm_Cong_unified_v2_test \
    --steps 100 --N 8 --seeds 1234 2345 3456
"""
from __future__ import annotations
import os
import csv
import json
import time
import hashlib
import argparse
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from latent_config import LatentConfig
from latent_data   import LatentDataset, _collate
from diffusion     import build_betas
from models        import LatentUNet, FeatureProjector, sinusoidal_embedding
from models_congestion import FeatureProjectorMultiStage
from utils_ema     import EMA


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_NZ_THRESH            = 0.01
_HOTSPOT_FRAC_DRC     = 0.01   # top 1%
_HOTSPOT_FRAC_CONG    = 0.05   # top 5%
_PREC_THRESH          = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Module import — sanitized name
# ─────────────────────────────────────────────────────────────────────────────
def _import_from_file(file_path: str):
    h = hashlib.sha1(file_path.encode()).hexdigest()[:8]
    name = f"_ldm_vae_{h}"
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample stable seeding — identical to training
# ─────────────────────────────────────────────────────────────────────────────
def _stable_seed(name: str) -> int:
    return int.from_bytes(
        hashlib.sha1(name.encode()).digest()[:4],
        byteorder="little", signed=False
    )


def _make_noise(B, C, H, W, device, base_seed, names, draw_k):
    x = torch.empty((B, C, H, W), device=device)
    for i, nm in enumerate(names):
        s = int(base_seed) ^ _stable_seed(str(nm)) ^ (int(draw_k) * 0x9E3779B1)
        g = torch.Generator(device=device)
        g.manual_seed(s & 0x7FFFFFFF)
        x[i:i+1] = torch.randn((1, C, H, W), device=device, generator=g)
    return x


def _make_ts(T, steps, device):
    ts = np.rint(np.linspace(T - 1, 0, steps, dtype=np.float64)).astype(np.int64)
    for i in range(1, len(ts)):
        if ts[i] >= ts[i - 1]:
            ts[i] = ts[i - 1] - 1
    return torch.tensor(np.clip(ts, 0, T - 1), device=device, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# VAE loader
# ─────────────────────────────────────────────────────────────────────────────
def _load_vae(vae_type, vae_dir, ae_ckpt_path, device):
    if vae_type == "LabelVAE_v2":
        mod = _import_from_file(os.path.join(vae_dir, "vae_model.py"))
        ck = torch.load(ae_ckpt_path, map_location=device)
        m = mod.LabelVAE(
            C_label=int(ck["C_label"]),
            latent_ch=int(ck["latent_ch"]),
            base_ch=int(ck.get("base_ch", 64)),
            log_scale=float(ck.get("log_scale", 10.0)),
        ).to(device)
    elif vae_type in ("CongestionVAE_v2", "CongestionVAE"):
        mod = _import_from_file(os.path.join(vae_dir, "vae_model_congestion.py"))
        ck = torch.load(ae_ckpt_path, map_location=device)
        m = mod.CongestionVAE(
            C_label=int(ck["C_label"]),
            latent_ch=int(ck["latent_ch"]),
            base_ch=int(ck.get("base_ch", 64)),
            logvar_min=float(ck.get("logvar_min", -4.0)),
            logvar_max=float(ck.get("logvar_max",  4.0)),
        ).to(device)
    elif vae_type == "CongestionAE":
        mod = _import_from_file(os.path.join(vae_dir, "ae_model_congestion.py"))
        ck = torch.load(ae_ckpt_path, map_location=device)
        m = mod.CongestionAE(
            C_label=int(ck["C_label"]),
            latent_ch=int(ck["latent_ch"]),
            base_ch=int(ck.get("base_ch", 64)),
        ).to(device)
    else:
        raise ValueError(f"Unknown vae_type: {vae_type!r}")

    m.load_state_dict(ck["net"], strict=True)
    if "ema" in ck:
        _e = EMA(m)
        _e.load_state_dict(ck["ema"])
        _e.copy_to(m)
        print("  VAE EMA applied")

    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: str, vae_dir_override: str | None, device):
    ck = torch.load(ckpt_path, map_location=device)

    stored_vae_dir = ck.get("vae_dir", "")
    if vae_dir_override:
        if os.path.realpath(vae_dir_override) != os.path.realpath(stored_vae_dir):
            print("[WARN] --vae_dir differs from checkpoint's stored vae_dir")
            print(f"  supplied: {vae_dir_override}")
            print(f"  stored:   {stored_vae_dir}")
            print("  Using supplied --vae_dir — verify this is intentional.")
        vae_dir = vae_dir_override
    else:
        if not stored_vae_dir:
            raise ValueError("Checkpoint has no stored vae_dir. Supply --vae_dir explicitly.")
        vae_dir = stored_vae_dir

    latent_ch = int(ck["latent_ch"])
    z_mean = torch.tensor(ck["z_mean"], dtype=torch.float32, device=device).view(1, latent_ch, 1, 1)
    z_std  = torch.tensor(ck["z_std"],  dtype=torch.float32, device=device).view(1, latent_ch, 1, 1)

    C_feat       = int(ck["C_feat"])
    feat_proj_ch = int(ck["feat_proj_ch"])
    proj_type    = ck.get("proj_type", "single")
    H_latent     = int(ck["H_latent"])

    if proj_type == "multistage":
        feat_proj = FeatureProjectorMultiStage(in_ch=C_feat, out_ch=feat_proj_ch).to(device)
    else:
        feat_proj = FeatureProjector(
            in_ch=C_feat,
            out_ch=feat_proj_ch,
            stride=256 // H_latent,
        ).to(device)

    feat_proj.load_state_dict(ck["feat_proj"], strict=True)
    feat_proj.eval()

    net = LatentUNet(
        in_ch=int(ck["in_ch"]),
        out_ch=latent_ch,
        base=int(ck["base_channels"]),
        t_emb_dim=128,
    ).to(device)
    net.load_state_dict(ck["net"], strict=True)

    if "ema" in ck:
        ema_raw = ck["ema"]
        ema_sd  = ema_raw.get("shadow", ema_raw) if isinstance(ema_raw, dict) else ema_raw
        net_sd  = {k[4:]: v for k, v in ema_sd.items() if k.startswith("net.")}
        proj_sd = {k[5:]: v for k, v in ema_sd.items() if k.startswith("proj.")}
        if net_sd:
            net.load_state_dict(net_sd, strict=True)
            print(f"  EMA applied: UNet ({len(net_sd)} tensors)")
        if proj_sd:
            feat_proj.load_state_dict(proj_sd, strict=True)
            print(f"  EMA applied: Projector ({len(proj_sd)} tensors)")

    net.eval()
    vae = _load_vae(ck["vae_type"], vae_dir, ck["ae_ckpt"], device)

    print(
        f"  task={ck['task']} epoch={ck['epoch']} | "
        f"latent={latent_ch}×{H_latent}×{ck['W_latent']} | "
        f"C_feat={C_feat}→{feat_proj_ch}ch ({proj_type})"
    )
    print(f"  drop_feat_ch={ck.get('drop_feat_ch', [])} | cfg_drop={ck.get('cfg_drop_prob', 0.0)}")
    print(f"  z_std: [{min(ck['z_std']):.3f}, {max(ck['z_std']):.3f}]")

    return net, feat_proj, vae, ck, z_mean, z_std


# ─────────────────────────────────────────────────────────────────────────────
# DDIM sampler
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def ddim_sample(
    net, feat_proj, vae, feat, alpha_bar,
    pred_type, steps, eta, cfg_scale,
    latent_ch, H_lat, W_lat,
    x_init, z_mean, z_std, seed
):
    B = feat.shape[0]
    device = feat.device
    T = alpha_bar.shape[0]

    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & 0x7FFFFFFF)

    cond = feat_proj(feat)
    null = torch.zeros_like(cond)
    z = x_init
    ts = _make_ts(T, steps, device)

    for i in range(steps):
        t = ts[i].expand(B)
        t_emb = sinusoidal_embedding(t, 128)
        ab_t = alpha_bar[t].view(B, 1, 1, 1)

        x_in = torch.cat([z, cond], 1)
        if cfg_scale > 0.0:
            pred_u = net(torch.cat([z, null], 1), t_emb)
            pred_c = net(x_in, t_emb)
            pred = pred_u + cfg_scale * (pred_c - pred_u)
        else:
            pred = net(x_in, t_emb)

        if pred_type == "v":
            z0  = torch.sqrt(ab_t) * z - torch.sqrt(1 - ab_t) * pred
            eps = torch.sqrt(1 - ab_t) * z + torch.sqrt(ab_t) * pred
        else:
            z0  = (z - torch.sqrt(1 - ab_t) * pred) / torch.sqrt(ab_t + 1e-12)
            eps = pred

        if i == steps - 1:
            z = z0
            break

        t_next  = ts[i + 1].expand(B)
        ab_next = alpha_bar[t_next].view(B, 1, 1, 1)
        sigma = eta * torch.sqrt(
            (1 - ab_next) / (1 - ab_t + 1e-12) *
            torch.clamp(1 - ab_t / (ab_next + 1e-12), min=0.0)
        )
        noise = torch.randn(z.shape, device=device, generator=g) if eta > 0 else 0.0
        z = (
            torch.sqrt(ab_next) * z0
            + torch.sqrt(torch.clamp(1 - ab_next - sigma**2, min=0.0)) * eps
            + sigma * noise
        )

    return vae.decode_from_z(z * z_std + z_mean)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def _safe_pearson(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _table1(gt, pr):
    gt = np.clip(gt.astype(np.float32), 0.0, 1.0)
    pr = np.clip(pr.astype(np.float32), 0.0, 1.0)
    mae  = float(np.mean(np.abs(gt - pr)))
    rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
    rng  = float(gt.max() - gt.min())
    nrms = (rmse / rng) if rng > 1e-8 else float("nan")
    ssim = float(structural_similarity(gt, pr, data_range=1.0))
    pear = _safe_pearson(gt, pr)
    return dict(mae=mae, ssim=ssim, nrms=nrms, pearson=pear)


def _table2_drc(gt, pr):
    gt = np.clip(gt.astype(np.float32), 0.0, 1.0)
    pr = np.clip(pr.astype(np.float32), 0.0, 1.0)
    gt_f = gt.ravel()
    pr_f = pr.ravel()
    m = {}

    for frac, key in [(0.01, "topk_1pct"), (0.005, "topk_05pct")]:
        k = max(1, int(frac * gt.size))
        gt_idx = np.argsort(gt_f)[-k:]
        pr_idx = np.argsort(pr_f)[-k:]
        m[key] = float(len(np.intersect1d(gt_idx, pr_idx)) / k)

    nz = gt_f > _NZ_THRESH
    m["nz_pear"] = _safe_pearson(gt_f[nz], pr_f[nz]) if nz.sum() >= 2 else float("nan")

    k_hs = max(1, int(_HOTSPOT_FRAC_DRC * gt.size))
    top_idx = np.argsort(gt_f)[-k_hs:]
    m["hs_mae"] = float(np.mean(np.abs(gt_f[top_idx] - pr_f[top_idx])))

    gt_pos = gt_f >= _PREC_THRESH
    pr_pos = pr_f >= _PREC_THRESH
    tp = float(np.sum(gt_pos & pr_pos))
    fp = float(np.sum(~gt_pos & pr_pos))
    fn = float(np.sum(gt_pos & ~pr_pos))

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (not np.isnan(prec) and prec + rec > 0) else float("nan")

    m.update(precision=prec, recall=rec, f1=f1)
    return m


def _table2_congestion(gt, pr):
    gt = np.clip(gt.astype(np.float32), 0.0, 1.0)
    pr = np.clip(pr.astype(np.float32), 0.0, 1.0)
    gt_f = gt.ravel()
    pr_f = pr.ravel()
    m = {}

    k_hs = max(1, int(_HOTSPOT_FRAC_CONG * gt.size))
    top_idx = np.argsort(gt_f)[-k_hs:]
    m["hs_mae"] = float(np.mean(np.abs(gt_f[top_idx] - pr_f[top_idx])))

    nz = gt_f > _NZ_THRESH
    m["nz_pear"] = _safe_pearson(gt_f[nz], pr_f[nz]) if nz.sum() >= 2 else float("nan")

    m["spatial_bias"] = float(np.mean(pr_f) - np.mean(gt_f))
    return m


def _trivial(gt, task, train_mean):
    pr = np.zeros_like(gt) if task == "DRC" else np.full_like(gt, train_mean)
    t1 = _table1(gt, pr)
    t2 = _table2_drc(gt, pr) if task == "DRC" else _table2_congestion(gt, pr)
    return t1, t2


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mn(lst, k):
    v = [x[k] for x in lst if not np.isnan(x.get(k, float("nan")))]
    return float(np.mean(v)) if v else float("nan")


def _sd(lst, k):
    v = [x[k] for x in lst if not np.isnan(x.get(k, float("nan")))]
    return float(np.std(v)) if v else float("nan")


def _row(label, m_lst, tv_lst, key, fmt, hi):
    arrow = "↑" if hi else "↓"
    m = _mn(m_lst, key)
    s = _sd(m_lst, key)
    t = _mn(tv_lst, key) if tv_lst else float("nan")
    tv_s = f"  trivial={t:{fmt}}" if not np.isnan(t) else ""
    print(f"  {label+' '+arrow:<30} {m:{fmt}} ± {s:{fmt}}{tv_s}")


def _print_tables(seed, task, n, t1_m, t1_tv, t2_m, t2_tv, unc_l, N):
    print(f"\n{'─'*65}")
    print(f"  SEED {seed} | {task} | N={N} draws | n={n}")
    print(f"{'─'*65}")

    print(f"\n  TABLE 1 — Standard Fidelity Metrics")
    print(f"  {'Metric':<30} {'Model':>16}  Trivial")
    print(f"  {'─'*62}")
    _row("MAE",     t1_m, t1_tv, "mae",     ".5f", False)
    _row("NRMS",    t1_m, t1_tv, "nrms",    ".4f", False)
    _row("SSIM",    t1_m, t1_tv, "ssim",    ".4f", True)
    _row("Pearson", t1_m, [],    "pearson", ".4f", True)

    print(f"\n  TABLE 2 — Task-Aligned Metrics")
    print(f"  {'Metric':<30} {'Model':>16}  Trivial")
    print(f"  {'─'*62}")

    if task == "DRC":
        _row("TopK@1%",      t2_m, t2_tv, "topk_1pct",  ".4f", True)
        _row("TopK@0.5%",    t2_m, t2_tv, "topk_05pct", ".4f", True)
        _row("Hotspot-MAE",  t2_m, t2_tv, "hs_mae",     ".5f", False)
        _row("NZ-Pearson",   t2_m, [],    "nz_pear",    ".4f", True)
        _row(f"F1@{_PREC_THRESH}", t2_m, t2_tv, "f1",   ".4f", True)
    else:
        _row("Hotspot-MAE",  t2_m, t2_tv, "hs_mae",        ".5f", False)
        _row("NZ-Pearson",   t2_m, [],    "nz_pear",       ".4f", True)
        _row("Spatial Bias", t2_m, [],    "spatial_bias",  ".5f", True)

    unc = float(np.nanmean(unc_l))
    print(f"\n  Uncertainty (var vs |error| Pearson): {unc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_args():
    p = argparse.ArgumentParser("LDM Sampler")
    p.add_argument("--ckpt", required=True, help="LDM checkpoint (best_gen.pt or best_val.pt)")
    p.add_argument("--vae_dir", default=None, help="Optional override for vae_dir")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, default=None,
                   help="Default: infer from checkpoint (1.5 with CFG training, else 0.0)")
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  LDM Sampler")
    net, feat_proj, vae, ck, z_mean, z_std = load_checkpoint(args.ckpt, args.vae_dir, device)
    print("=" * 65)

    task      = str(ck["task"])
    pred_type = str(ck["pred_type"]).lower()
    latent_ch = int(ck["latent_ch"])
    H_lat     = int(ck["H_latent"])
    W_lat     = int(ck["W_latent"])
    T         = int(ck["diffusion_steps"])
    drop_ch   = ck.get("drop_feat_ch", [])

    alpha_bar = torch.tensor(
        np.cumprod(1.0 - build_betas(T, ck["beta_schedule"])),
        dtype=torch.float32,
        device=device
    )

    if args.cfg_scale is not None:
        cfg_scale = args.cfg_scale
    else:
        trained_with_cfg = float(ck.get("cfg_drop_prob", 0.0)) > 0
        cfg_scale = 1.5 if trained_with_cfg else 0.0

    print(f"  cfg_scale={cfg_scale} (checkpoint trained with cfg_drop_prob={ck.get('cfg_drop_prob', 0.0)})")

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg = LatentConfig.load(os.path.join(ckpt_dir, "latent_config.json"))
    csv_path = cfg.csv_test if args.split == "test" else cfg.csv_val

    ds = LatentDataset(
        csv_path=csv_path,
        feature_dir=cfg.feature_dir,
        label_dir=cfg.label_dir,
        drop_channels=drop_ch,
        split=args.split,
        verify=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )

    if task == "Congestion":
        ds_tr = LatentDataset(
            csv_path=cfg.csv_train,
            feature_dir=cfg.feature_dir,
            label_dir=cfg.label_dir,
            drop_channels=drop_ch,
            split="train",
            verify=False,
        )
        ldr_tr = DataLoader(
            ds_tr,
            batch_size=64,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate,
            drop_last=False,
        )
        s, n = 0.0, 0
        for _, lb, _ in ldr_tr:
            s += float(lb.sum())
            n += lb.numel()
        train_mean = s / max(n, 1)
        print(f"  Trivial baseline: mean={train_mean:.4f} (full training set, n={len(ds_tr)})")
    else:
        train_mean = 0.0

    print(f"\n  split={args.split} n={len(ds)} | steps={args.steps} eta={args.eta} N={args.N}")
    print(f"  seeds={args.seeds}\n")

    summary = dict(
        ckpt=args.ckpt,
        epoch=ck["epoch"],
        task=task,
        split=args.split,
        N=args.N,
        steps=args.steps,
        cfg_scale=cfg_scale,
        eta=args.eta,
        seed_results={},
    )

    t_total = time.time()
    for seed in args.seeds:
        t1_model, t1_triv = [], []
        t2_model, t2_triv = [], []
        unc_l = []
        per_rows = []

        for bi, (feat, lbl, names) in enumerate(loader, 1):
            if args.max_batches > 0 and bi > args.max_batches:
                break

            feat = feat.float().to(device)
            lbl  = lbl.float().to(device)
            B = feat.shape[0]

            gens = []
            for k in range(args.N):
                x_init = _make_noise(B, latent_ch, H_lat, W_lat, device, seed, list(names), k)
                gen = ddim_sample(
                    net, feat_proj, vae, feat, alpha_bar,
                    pred_type=pred_type,
                    steps=args.steps,
                    eta=args.eta,
                    cfg_scale=cfg_scale,
                    latent_ch=latent_ch,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    x_init=x_init,
                    z_mean=z_mean,
                    z_std=z_std,
                    seed=seed ^ (k * 0x1F2E3D4C),
                )
                gens.append(gen)

            G = torch.stack(gens, 0).clamp(0, 1)
            mean_pred = G.mean(0)
            var_map = torch.var(G, dim=0, unbiased=False)

            gt_np   = lbl.clamp(0, 1).cpu().numpy()[:, 0]
            mean_np = mean_pred.cpu().numpy()[:, 0]
            var_np  = var_map.cpu().numpy()[:, 0]

            for i, name in enumerate(names):
                gt_i = gt_np[i]
                pr_i = mean_np[i]
                vr_i = var_np[i]

                m1 = _table1(gt_i, pr_i)
                m2 = _table2_drc(gt_i, pr_i) if task == "DRC" else _table2_congestion(gt_i, pr_i)
                tv1, tv2 = _trivial(gt_i, task, train_mean)
                unc = _safe_pearson(vr_i.ravel(), np.abs(pr_i - gt_i).ravel())

                t1_model.append(m1)
                t1_triv.append(tv1)
                t2_model.append(m2)
                t2_triv.append(tv2)
                unc_l.append(unc)

                row = dict(seed=seed, name=name)
                row.update({f"t1_{k}": v for k, v in m1.items()})
                row.update({f"t2_{k}": v for k, v in m2.items()})
                row.update({f"tv1_{k}": v for k, v in tv1.items()})
                row["unc_err_corr"] = unc
                per_rows.append(row)

            if bi % 10 == 0:
                print(f"  seed={seed} batch {bi}/{len(loader)}")

        _print_tables(seed, task, len(t1_model), t1_model, t1_triv, t2_model, t2_triv, unc_l, args.N)

        if per_rows:
            keys = list(per_rows[0].keys())
            csv_out = os.path.join(args.out_dir, f"per_sample_seed{seed}.csv")
            with open(csv_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in per_rows:
                    w.writerow({k: r.get(k, "") for k in keys})
            print(f"\n  Saved: {csv_out}")

        summary["seed_results"][str(seed)] = dict(
            n=len(t1_model),
            table1={k: dict(mean=_mn(t1_model, k), std=_sd(t1_model, k), trivial=_mn(t1_triv, k))
                    for k in t1_model[0]},
            table2={k: dict(mean=_mn(t2_model, k), std=_sd(t2_model, k), trivial=_mn(t2_triv, k))
                    for k in t2_model[0]},
            unc_err_corr=float(np.nanmean(unc_l)),
        )

    primary_key = "topk_1pct" if task == "DRC" else "hs_mae"
    primary_tbl = "table2"
    macro_vals = [
        summary["seed_results"][str(s)][primary_tbl][primary_key]["mean"]
        for s in args.seeds
    ]

    summary["macro"] = dict(
        primary_metric=primary_key,
        mean=float(np.nanmean(macro_vals)),
        std=float(np.nanstd(macro_vals)),
        total_time_sec=time.time() - t_total,
    )

    print(f"\n{'='*65}")
    print(f"  MACRO OVER {len(args.seeds)} SEEDS | {task} | {args.split}")
    print(f"{'='*65}")
    print(f"  {primary_key}: {summary['macro']['mean']:.4f} ± {summary['macro']['std']:.4f}")
    print(f"  Total: {summary['macro']['total_time_sec']:.0f}s")

    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_json}")


if __name__ == "__main__":
    main()