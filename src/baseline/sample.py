#!/usr/bin/env python3
"""
sample.py — Pixel-space conditional diffusion. Inference + evaluation.
Metrics aligned exactly with latent_sampler.py and test.py (SOTA).

Table-1 (all methods):
    MAE, NRMS, SSIM, Pearson

Table-2 (routability sign-off, all methods):
    TopK@1%, TopK@0.5%,
    Hotspot-MAE  (top 1% for DRC / top 5% for congestion — index-based argsort)
    NZ-Pearson   (pixels > _NZ_THRESH=0.01)
    F1@0.1, Precision@0.1, Recall@0.1  (DRC only — _PREC_THRESH=0.10)
    Spatial Bias  (congestion only)
    Unc-Error Corr (diffusion only — var vs |error| Pearson)

Dropped vs old sample.py:
    PSNR          — misleading for DRC; not standard for congestion
    RMSE          — covered by NRMS
    PI coverage   — N=8 draws too few for reliable quantile estimation
    exp_mae/rmse  — not in SOTA or LDM tables, orphaned

Usage:
    python sample.py \\
        --ckpt runs/pixel_DRC_N28/best.pt \\
        --split test --out_dir results/pixel_DRC_N28_test

    python sample.py \\
        --ckpt runs/pixel_cong_N28/best.pt \\
        --split test --out_dir results/pixel_cong_N28_test
"""
import os
import argparse
import json
import hashlib
import time
import csv
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from config  import TrainConfig
from data    import CircuitNetDataset, _collate
from diffusion import build_betas
from models  import ConditionalUNet, sinusoidal_embedding
from trainer import Trainer
from utils_ema import EMA

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match latent_sampler.py and test.py exactly
# ─────────────────────────────────────────────────────────────────────────────
_NZ_THRESH    = 0.01
_HOTSPOT_FRAC = 0.01   # top 1% for DRC
_HOTSPOT_FRAC_CONG = 0.05  # top 5% for congestion
_PREC_THRESH  = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# DDIM schedule
# ─────────────────────────────────────────────────────────────────────────────

def make_ddim_timesteps(T: int, steps: int,
                        device: torch.device) -> torch.Tensor:
    ts = np.rint(
        np.linspace(T - 1, 0, steps, dtype=np.float64)
    ).astype(np.int64)
    for i in range(1, len(ts)):
        if ts[i] >= ts[i - 1]:
            ts[i] = ts[i - 1] - 1
    if ts[-1] < 0:
        ts = np.floor(
            np.linspace(T - 1, 0, steps, dtype=np.float64)
        ).astype(np.int64)
        ts = np.clip(ts, 0, T - 1)
        for i in range(1, len(ts)):
            if ts[i] >= ts[i - 1]:
                ts[i] = max(0, ts[i - 1] - 1)
    return torch.tensor(np.clip(ts, 0, T - 1),
                        device=device, dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
# Per-sample seeding — batch-size invariant
# ─────────────────────────────────────────────────────────────────────────────

def stable_int_from_name(name: str) -> int:
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def make_initial_noise(B, C, H, W, device, base_seed,
                       names: List[str], draw_k: int) -> torch.Tensor:
    x = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
    for i, nm in enumerate(names):
        s = (int(base_seed)
             ^ stable_int_from_name(str(nm))
             ^ (int(draw_k) * 0x9E3779B1))
        g = torch.Generator(device=device)
        g.manual_seed(s & 0x7FFFFFFF)
        x[i:i + 1] = torch.randn((1, C, H, W),
                                  device=device, generator=g)
    return x

# ─────────────────────────────────────────────────────────────────────────────
# Model input
# ─────────────────────────────────────────────────────────────────────────────

def build_model_input(x_t, feats, self_cond, use_self_cond):
    parts = [x_t]
    if use_self_cond:
        parts.append(self_cond if self_cond is not None
                     else torch.zeros_like(x_t))
    parts.append(feats)
    return torch.cat(parts, dim=1)

# ─────────────────────────────────────────────────────────────────────────────
# DDIM sampler
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample_one(model, feats, alpha_bar, pred_type,
                    steps, eta, cfg_scale, use_self_cond,
                    x_init) -> torch.Tensor:
    B      = feats.shape[0]
    device = feats.device
    T      = alpha_bar.shape[0]
    x      = x_init
    ts     = make_ddim_timesteps(T, steps, device)
    self_cond   = None
    feats_null  = torch.zeros_like(feats)

    for i in range(steps):
        t     = ts[i].expand(B)
        t_emb = sinusoidal_embedding(t, 128)
        ab_t  = alpha_bar[t].view(B, 1, 1, 1)

        if cfg_scale > 0.0:
            pred_u = model(
                build_model_input(x, feats_null, None, use_self_cond), t_emb)
            pred_c = model(
                build_model_input(x, feats, self_cond, use_self_cond), t_emb)
            pred = pred_u + cfg_scale * (pred_c - pred_u)
        else:
            pred = model(
                build_model_input(x, feats, self_cond, use_self_cond), t_emb)

        if pred_type == "eps":
            eps = pred
            x0  = ((x - torch.sqrt(1.0 - ab_t) * eps)
                   / torch.sqrt(ab_t + 1e-12))
        else:
            v   = pred
            x0  = (torch.sqrt(ab_t) * x
                   - torch.sqrt(1.0 - ab_t) * v)
            eps = (torch.sqrt(1.0 - ab_t) * x
                   + torch.sqrt(ab_t) * v)

        x0 = x0.clamp(-1, 1)
        if use_self_cond:
            self_cond = x0.detach()
        if i == steps - 1:
            x = x0
            break

        t_next  = ts[i + 1].expand(B)
        ab_next = alpha_bar[t_next].view(B, 1, 1, 1)
        sigma   = (
            eta
            * torch.sqrt((1.0 - ab_next) / (1.0 - ab_t + 1e-12))
            * torch.sqrt(
                torch.clamp(1.0 - ab_t / (ab_next + 1e-12), min=0.0))
        )
        noise = torch.randn_like(x) if eta > 0.0 else 0.0
        x = (torch.sqrt(ab_next) * x0
             + torch.sqrt(
                 torch.clamp(1.0 - ab_next - sigma ** 2, min=0.0)) * eps
             + sigma * noise)

    return ((x + 1) / 2).clamp(0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics — identical definitions to latent_sampler.py and test.py
# ─────────────────────────────────────────────────────────────────────────────

def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _nrms(gt: np.ndarray, pred: np.ndarray) -> float:
    rng = float(gt.max()) - float(gt.min())
    if rng < 1e-8:
        return float("nan")
    rmse = float(np.sqrt(np.mean(
        (gt.astype(np.float64) - pred.astype(np.float64)) ** 2)))
    return rmse / rng


def compute_table1(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """MAE, NRMS, SSIM, Pearson — identical to test.py and latent_sampler."""
    gt   = np.clip(gt.astype(np.float32),   0.0, 1.0)
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    mae  = float(np.mean(np.abs(gt - pred)))
    nrms = _nrms(gt, pred)
    ssim = float(structural_similarity(gt, pred, data_range=1.0))
    pear = _safe_pearson(gt, pred)
    return dict(mae=mae, nrms=nrms, ssim=ssim, pearson=pear)


def compute_table2_drc(gt: np.ndarray,
                       pred: np.ndarray) -> Dict[str, float]:
    """DRC sign-off metrics — identical to latent_sampler._table2_drc()."""
    gt   = np.clip(gt.astype(np.float32),   0.0, 1.0)
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    gt_f = gt.ravel()
    pr_f = pred.ravel()
    m    = {}

    # TopK — argsort + intersect1d
    for frac, key in [(0.01, "topk_1pct"), (0.005, "topk_05pct")]:
        k      = max(1, int(frac * gt_f.size))
        gt_idx = np.argsort(gt_f)[-k:]
        pr_idx = np.argsort(pr_f)[-k:]
        m[key] = float(len(np.intersect1d(gt_idx, pr_idx)) / k)

    # NZ-Pearson
    nz           = gt_f > _NZ_THRESH
    m["nz_mae"]  = (float(np.mean(np.abs(gt_f[nz] - pr_f[nz])))
                    if nz.any() else float("nan"))
    m["nz_pear"] = (_safe_pearson(gt_f[nz], pr_f[nz])
                    if nz.sum() >= 2 else float("nan"))

    # Hotspot-MAE — top 1% by GT value, index-based
    k_hs        = max(1, int(_HOTSPOT_FRAC * gt_f.size))
    top_idx     = np.argsort(gt_f)[-k_hs:]
    m["hs_mae"] = float(np.mean(np.abs(gt_f[top_idx] - pr_f[top_idx])))

    # Precision / Recall / F1
    gt_pos = gt_f >= _PREC_THRESH
    pr_pos = pr_f >= _PREC_THRESH
    tp     = float(np.sum(gt_pos & pr_pos))
    fp     = float(np.sum(~gt_pos & pr_pos))
    fn     = float(np.sum(gt_pos & ~pr_pos))
    prec   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1     = (2 * prec * rec / (prec + rec)
              if (not np.isnan(prec) and not np.isnan(rec)
                  and prec + rec > 0)
              else float("nan"))
    m.update(precision=prec, recall=rec, f1=f1)
    return m


def compute_table2_congestion(gt: np.ndarray,
                               pred: np.ndarray) -> Dict[str, float]:
    """Congestion sign-off — identical to latent_sampler congestion metrics."""
    gt   = np.clip(gt.astype(np.float32),   0.0, 1.0)
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    gt_f = gt.ravel()
    pr_f = pred.ravel()
    m    = {}

    for frac, key in [(0.01, "topk_1pct"), (0.005, "topk_05pct")]:
        k      = max(1, int(frac * gt_f.size))
        gt_idx = np.argsort(gt_f)[-k:]
        pr_idx = np.argsort(pr_f)[-k:]
        m[key] = float(len(np.intersect1d(gt_idx, pr_idx)) / k)

    # Hotspot-MAE top 5% for congestion
    k_hs        = max(1, int(_HOTSPOT_FRAC_CONG * gt_f.size))
    top_idx     = np.argsort(gt_f)[-k_hs:]
    m["hs_mae"] = float(np.mean(np.abs(gt_f[top_idx] - pr_f[top_idx])))

    nz           = gt_f > _NZ_THRESH
    m["nz_pear"] = (_safe_pearson(gt_f[nz], pr_f[nz])
                    if nz.sum() >= 2 else float("nan"))

    m["spatial_bias"] = float(np.mean(pr_f) - np.mean(gt_f))
    return m


def compute_trivial(gt: np.ndarray, task: str,
                    train_mean: float) -> tuple:
    pr = (np.zeros_like(gt) if task.upper() == "DRC"
          else np.full_like(gt, train_mean))
    t1 = compute_table1(gt, pr)
    t2 = (compute_table2_drc(gt, pr) if task.upper() == "DRC"
          else compute_table2_congestion(gt, pr))
    return t1, t2


def compute_unc_err_corr(var_map: np.ndarray,
                          abs_err: np.ndarray) -> float:
    return _safe_pearson(var_map.ravel(), abs_err.ravel())

# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mn(lst, key):
    v = [x[key] for x in lst
         if not np.isnan(x.get(key, float("nan")))]
    return float(np.mean(v)) if v else float("nan")


def _sd(lst, key):
    v = [x[key] for x in lst
         if not np.isnan(x.get(key, float("nan")))]
    return float(np.std(v)) if v else float("nan")


def _agg(lst):
    arr = np.array(lst, dtype=np.float64)
    return {"mean": float(np.nanmean(arr)),
            "std":  float(np.nanstd(arr))}


def write_csv(path: str, rows: List[Dict[str, Any]], header: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _row(label, m_lst, tv_lst, key, fmt, higher_better):
    arrow = "↑" if higher_better else "↓"
    mn    = _mn(m_lst, key)
    sd    = _sd(m_lst, key)
    tv    = _mn(tv_lst, key) if tv_lst else float("nan")
    tv_s  = f"  trivial={tv:{fmt}}" if not np.isnan(tv) else ""
    print(f"  {label+' '+arrow:<30} {mn:{fmt}} ± {sd:{fmt}}{tv_s}")


def print_seed_results(seed, task, n, t1_m, t1_tv,
                        t2_m, t2_tv, unc_l):
    is_drc = task.upper() == "DRC"
    print(f"\n{'─'*65}")
    print(f"  SEED {seed} | {task} | n={n}")
    print(f"{'─'*65}")

    print(f"\n  TABLE 1 — Standard Regression")
    _row("MAE",     t1_m, t1_tv, "mae",     ".5f", False)
    _row("NRMS",    t1_m, t1_tv, "nrms",    ".4f", False)
    _row("SSIM",    t1_m, t1_tv, "ssim",    ".4f", True)
    _row("Pearson", t1_m, [],    "pearson", ".4f", True)

    print(f"\n  TABLE 2 — Routability Sign-off")
    _row("TopK@1%",  t2_m, t2_tv, "topk_1pct",  ".4f", True)
    _row("TopK@0.5%",t2_m, t2_tv, "topk_05pct", ".4f", True)
    _row("Hotspot-MAE", t2_m, t2_tv, "hs_mae",  ".5f", False)
    _row("NZ-Pearson",  t2_m, [],    "nz_pear",  ".4f", True)

    if is_drc:
        _row(f"Precision@{_PREC_THRESH}", t2_m, t2_tv,
             "precision", ".4f", True)
        _row(f"Recall@{_PREC_THRESH}",    t2_m, t2_tv,
             "recall",    ".4f", True)
        _row(f"F1@{_PREC_THRESH}",        t2_m, t2_tv,
             "f1",        ".4f", True)
    else:
        _row("Spatial Bias", t2_m, [], "spatial_bias", ".5f", True)

    unc = float(np.nanmean(unc_l))
    print(f"\n  Uncertainty (var vs |error| Pearson): {unc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser(
        "Pixel diffusion sampler — unified metrics")
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--split",       default="test",
                   choices=["val", "test"])
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--steps",       type=int,   default=100)
    p.add_argument("--eta",         type=float, default=0.0)
    p.add_argument("--cfg_scale",   type=float, default=1.5)
    p.add_argument("--N",           type=int,   default=8)
    p.add_argument("--seeds",       nargs="+",  type=int,
                   default=[1234, 2345, 3456])
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--max_batches", type=int,   default=0)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = build_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    info  = Trainer.load_for_inference(args.ckpt, device=device)
    model = info["model"]
    cfg   = info["cfg"]
    ckpt  = info["ckpt"]

    pred_type     = str(ckpt["pred_type"]).lower()
    use_self_cond = bool(ckpt["use_self_cond"])
    C_feat        = int(ckpt["C_feat"])
    C_label       = int(ckpt["C_label"])
    task          = str(cfg.task)
    is_drc        = task.upper() == "DRC"

    T         = int(ckpt["diffusion_steps"])
    betas_np  = build_betas(T, str(ckpt["beta_schedule"]))
    alpha_bar = torch.tensor(
        np.cumprod(1.0 - betas_np),
        dtype=torch.float32, device=device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    csv_path = (cfg.csv_val if args.split == "val"
                else cfg.csv_test)
    ds = CircuitNetDataset(
        csv_path, cfg.feature_dir, cfg.label_dir,
        verify_range=True)
    assert ds.C_feat  == C_feat
    assert ds.C_label == C_label

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False, collate_fn=_collate)

    # ── Training mean for congestion trivial baseline ─────────────────────────
    train_mean = 0.0
    if not is_drc:
        ds_train = CircuitNetDataset(
            cfg.csv_train, cfg.feature_dir, cfg.label_dir,
            verify_range=False)
        ldr_train = DataLoader(
            ds_train, batch_size=64, shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate, drop_last=False)
        s, n = 0.0, 0
        for _, lb, _ in ldr_train:
            s += float(lb.sum()); n += lb.numel()
        train_mean = s / max(n, 1)
        print(f"[Baseline] Congestion train mean={train_mean:.4f} "
              f"(n={len(ds_train)})")

    print(f"\n[INIT] task={task} | split={args.split} | n={len(ds)}")
    print(f"[INIT] C_feat={C_feat} C_label={C_label} T={T}")
    print(f"[INIT] steps={args.steps} eta={args.eta} "
          f"cfg={args.cfg_scale} N={args.N}")
    print(f"[INIT] seeds={args.seeds}")
    print(f"[INIT] NZ_THRESH={_NZ_THRESH}  "
          f"HOTSPOT_FRAC={_HOTSPOT_FRAC}  "
          f"PREC_THRESH={_PREC_THRESH}")
    print(f"[INIT] Metric alignment: latent_sampler.py / test.py")

    summary = {
        "ckpt": args.ckpt, "task": task,
        "split": args.split, "N": args.N,
        "C_feat": C_feat, "seeds": args.seeds,
        "seed_results": {},
    }
    t0 = time.time()

    for seed in args.seeds:
        t1_model, t1_triv = [], []
        t2_model, t2_triv = [], []
        unc_l    = []
        per_rows = []

        for bi, (feats, lbls, names) in enumerate(loader, 1):
            if args.max_batches > 0 and bi > args.max_batches:
                break

            feats = feats.float().to(device)
            lbls  = lbls.float().to(device)
            B, _, H, W = feats.shape

            # N draws
            gens = []
            for k in range(args.N):
                x_init = make_initial_noise(
                    B, C_label, H, W, device,
                    seed, list(names), k)
                gen = ddim_sample_one(
                    model, feats, alpha_bar,
                    pred_type=pred_type,
                    steps=args.steps,
                    eta=args.eta,
                    cfg_scale=args.cfg_scale,
                    use_self_cond=use_self_cond,
                    x_init=x_init)
                gens.append(gen)

            G         = torch.stack(gens, 0)   # (N,B,C,H,W)
            mean_pred = G.mean(0)               # (B,C,H,W)
            var_map   = torch.var(G, dim=0, unbiased=False)

            gt_np   = lbls.clamp(0,1).cpu().numpy()[:, 0]
            mean_np = mean_pred.clamp(0,1).cpu().numpy()[:, 0]
            var_np  = var_map.cpu().numpy()[:, 0]

            for i, name in enumerate(names):
                gt_i   = gt_np[i]
                pr_i   = mean_np[i]
                var_i  = var_np[i]

                m1       = compute_table1(gt_i, pr_i)
                m2       = (compute_table2_drc(gt_i, pr_i)
                            if is_drc
                            else compute_table2_congestion(gt_i, pr_i))
                tv1, tv2 = compute_trivial(gt_i, task, train_mean)
                unc      = compute_unc_err_corr(
                    var_i, np.abs(pr_i - gt_i))

                t1_model.append(m1);  t1_triv.append(tv1)
                t2_model.append(m2);  t2_triv.append(tv2)
                unc_l.append(unc)

                row = dict(seed=seed, name=name)
                row.update({f"t1_{k}": v for k, v in m1.items()})
                row.update({f"t2_{k}": v for k, v in m2.items()})
                row.update({f"tv1_{k}": v for k, v in tv1.items()})
                row["unc_err_corr"] = unc
                per_rows.append(row)

            if bi % 10 == 0:
                print(f"  seed={seed} batch {bi}/{len(loader)}")

        print_seed_results(seed, task, len(t1_model),
                           t1_model, t1_triv,
                           t2_model, t2_triv, unc_l)

        # CSV
        if per_rows:
            keys    = list(per_rows[0].keys())
            csv_out = os.path.join(
                args.out_dir, f"per_sample_seed{seed}.csv")
            with open(csv_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in per_rows:
                    w.writerow({k: r.get(k, "") for k in keys})
            print(f"  [OUT] {csv_out}")

        summary["seed_results"][str(seed)] = dict(
            n=len(t1_model),
            table1={k: dict(mean=_mn(t1_model, k),
                            std=_sd(t1_model, k),
                            trivial=_mn(t1_triv, k))
                    for k in t1_model[0]},
            table2={k: dict(mean=_mn(t2_model, k),
                            std=_sd(t2_model, k),
                            trivial=_mn(t2_triv, k))
                    for k in (t2_model[0] if t2_model else {})},
            unc_err_corr=float(np.nanmean(unc_l)),
        )

    # ── Macro over seeds ──────────────────────────────────────────────────────
    ss          = summary["seed_results"]
    primary_key = "topk_1pct" if is_drc else "mae"
    primary_tbl = "table2"    if is_drc else "table1"

    macro_vals = [
        ss[str(s)][primary_tbl][primary_key]["mean"]
        for s in args.seeds]
    summary["macro"] = dict(
        primary_metric=primary_key,
        mean=float(np.nanmean(macro_vals)),
        std=float(np.nanstd(macro_vals)),
        total_time_sec=time.time() - t0,
    )

    print(f"\n{'='*65}")
    print(f"  MACRO OVER {len(args.seeds)} SEEDS | {task} | {args.split}")
    print(f"{'='*65}")
    print(f"  {primary_key}: "
          f"{summary['macro']['mean']:.4f} "
          f"± {summary['macro']['std']:.4f}")
    print(f"  Total: {summary['macro']['total_time_sec']:.0f}s")

    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [OUT] {out_json}")


if __name__ == "__main__":
    main()