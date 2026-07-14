#!/usr/bin/env python3
"""
vae_eval_congestion.py — Evaluate CongestionVAE on val or test split.

Primary:
  MAE, SSIM

Secondary:
  NRMSE, Pearson, spatial bias

Latent:
  per-channel std, clamp fractions

Baselines:
  constant training-mean predictor

N14 update:
  also reports per-design metrics, because the split is design-wise.
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from vae_config_congestion import CongestionVAEConfig
from vae_model_congestion import CongestionVAE
from vae_data_congestion import CongestionLabelDataset, _collate
from vae_train_congestion import EMA


ALL_DESIGNS = {
    "RISCY",
    "RISCY-FPU",
    "zero-riscy",
    "Vortex-small",
    "Vortex-large",
    "nvdla-small",
    "nvdla-large",
    "openc910-1",
}


def derive_design_key(stem: str) -> str:
    # Optional prefixed form: <design>__<sample_stem>
    if "__" in stem:
        prefix = stem.split("__", 1)[0]
        if prefix in ALL_DESIGNS:
            return prefix

    # Standard N14 processed filename stem begins with "<design>_"
    for design in sorted(ALL_DESIGNS, key=len, reverse=True):
        if stem == design or stem.startswith(design + "_"):
            return design

    return "UNKNOWN"


def _safe_pearson(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_metric_lists(d):
    out = {}
    for key, vals in d.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            out[key] = float("nan")
        elif key in ("nrms", "pearson"):
            out[key] = float(np.nanmean(arr))
        else:
            out[key] = float(np.mean(arr))
    return out


def main():
    p = argparse.ArgumentParser("CongestionVAE evaluation")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))

    ck = torch.load(args.ckpt, map_location=device)
    assert ck.get("model_type") == "CongestionVAE_v2", \
        f"Expected CongestionVAE_v2, got {ck.get('model_type')}"

    model = CongestionVAE(
        C_label=int(ck["C_label"]),
        latent_ch=int(ck["latent_ch"]),
        base_ch=int(ck.get("base_ch", 64)),
        logvar_min=float(ck.get("logvar_min", -4.0)),
        logvar_max=float(ck.get("logvar_max", 4.0)),
    ).to(device)
    model.load_state_dict(ck["net"], strict=True)

    if "ema" in ck:
        ema = EMA(model)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model)
        print("[Eval] EMA weights applied")
    model.eval()

    print(f"[Eval] epoch={ck['epoch']} | latent={ck['latent_ch']}×64×64 | "
          f"beta={ck.get('beta_target', '?')} | tech={ck.get('tech', '?')}")

    cfg = CongestionVAEConfig.load(os.path.join(ckpt_dir, "config.json"))
    csv_path = cfg.csv_test if args.split == "test" else cfg.csv_val

    # Training mean for trivial baseline
    ds_train = CongestionLabelDataset(
        cfg.csv_train, cfg.label_dir, split="train", verify=False
    )
    ldr_tmp = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=False
    )
    s = 0.0
    n = 0
    with torch.no_grad():
        for lbl, _ in ldr_tmp:
            s += float(lbl.sum())
            n += lbl.numel()
    train_mean = s / max(n, 1)
    print(f"[Eval] Training label mean={train_mean:.4f}")

    ds = CongestionLabelDataset(
        csv_path, cfg.label_dir, split=args.split, verify=False
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False
    )
    print(f"[Eval] split={args.split} | n={len(ds)}")

    ch_mu_stds = []
    logvar_all = []

    overall = {
        "mae": [],
        "ssim": [],
        "nrms": [],
        "pearson": [],
        "bias": [],
        "triv_mae": [],
        "triv_ssim": [],
    }

    per_design = defaultdict(lambda: {
        "mae": [],
        "ssim": [],
        "nrms": [],
        "pearson": [],
        "bias": [],
        "triv_mae": [],
        "triv_ssim": [],
    })

    with torch.no_grad():
        for lbl, names in loader:
            lbl = lbl.float().to(device)
            recon, mu, logvar, z = model(lbl, sample=False)

            ch_mu_stds.append(mu.std(dim=(0, 2, 3)).cpu().numpy())
            logvar_all.append(logvar.detach().cpu().float())

            gt_np = lbl.cpu().numpy()[:, 0]
            pr_np = recon.cpu().numpy()[:, 0]

            for i in range(gt_np.shape[0]):
                gt_i = gt_np[i]
                pr_i = pr_np[i]
                name_i = names[i]
                design_i = derive_design_key(name_i)

                mae = float(np.mean(np.abs(gt_i - pr_i)))
                rmse = float(np.sqrt(np.mean((gt_i - pr_i) ** 2)))
                rng = float(gt_i.max() - gt_i.min())
                nrms = (rmse / rng) if rng > 1e-8 else float("nan")
                ssim = float(structural_similarity(gt_i, pr_i, data_range=1.0))
                pear = _safe_pearson(gt_i, pr_i)
                bias = float(np.mean(pr_i) - np.mean(gt_i))

                triv = np.full_like(gt_i, train_mean)
                triv_mae = float(np.mean(np.abs(gt_i - triv)))
                triv_ssim = float(structural_similarity(gt_i, triv, data_range=1.0))

                for bucket in (overall, per_design[design_i]):
                    bucket["mae"].append(mae)
                    bucket["ssim"].append(ssim)
                    bucket["nrms"].append(nrms)
                    bucket["pearson"].append(pear)
                    bucket["bias"].append(bias)
                    bucket["triv_mae"].append(triv_mae)
                    bucket["triv_ssim"].append(triv_ssim)

    ch_stds_avg = np.mean(ch_mu_stds, axis=0)
    lv_cat = torch.cat([lv.reshape(-1) for lv in logvar_all])
    frac_low = float((lv_cat <= float(ck.get("logvar_min", -4.0)) + 1e-4).float().mean())
    frac_high = float((lv_cat >= float(ck.get("logvar_max", 4.0)) - 1e-4).float().mean())

    overall_summary = summarize_metric_lists(overall)
    per_design_summary = {
        d: summarize_metric_lists(vals)
        for d, vals in sorted(per_design.items())
    }
    per_design_counts = {
        d: len(vals["mae"])
        for d, vals in sorted(per_design.items())
    }

    print("\n" + "=" * 72)
    print(f"  CongestionVAE EVAL — {args.split}")
    print("=" * 72)
    print("\n  Overall reconstruction (model vs trivial baseline):")
    print(f"  {'MAE ↓':<22} {overall_summary['mae']:>10.5f}  "
          f"{overall_summary['triv_mae']:>10.5f}  (trivial)")
    print(f"  {'SSIM ↑':<22} {overall_summary['ssim']:>10.4f}  "
          f"{overall_summary['triv_ssim']:>10.4f}  (trivial)")
    print(f"  {'NRMSE ↓':<22} {overall_summary['nrms']:>10.4f}")
    print(f"  {'Pearson ↑':<22} {overall_summary['pearson']:>10.4f}")
    print(f"  {'Spatial bias':<22} {overall_summary['bias']:>10.5f}  (target: ~0)")

    print("\n  Per-design reconstruction:")
    print(f"  {'Design':<14} {'N':>6} {'MAE':>10} {'SSIM':>10} "
          f"{'NRMSE':>10} {'Pearson':>10} {'Bias':>10}")
    print(f"  {'-'*14} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for d in sorted(per_design_summary):
        s = per_design_summary[d]
        n_d = per_design_counts[d]
        print(f"  {d:<14} {n_d:>6d} {s['mae']:>10.5f} {s['ssim']:>10.4f} "
              f"{s['nrms']:>10.4f} {s['pearson']:>10.4f} {s['bias']:>10.5f}")

    print(f"\n  Latent (per-channel std over {args.split} set):")
    print(f"  {'Ch':<6} {'Std':>8}  Status")
    for c in range(len(ch_stds_avg)):
        status = "✓" if ch_stds_avg[c] >= 0.3 else "⚠ weak"
        print(f"  {c:<6} {ch_stds_avg[c]:>8.4f}  {status}")
    print(f"  Global: mean={np.mean(ch_stds_avg):.4f}  "
          f"min={np.min(ch_stds_avg):.4f}")
    print(f"  Clamp: low={frac_low:.3f}  high={frac_high:.3f}")
    print("=" * 72)

    out_dir = os.path.join(ckpt_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "ckpt": args.ckpt,
        "epoch": ck["epoch"],
        "split": args.split,
        "n": len(overall["mae"]),
        "train_mean": train_mean,
        "metrics": {
            "mae": overall_summary["mae"],
            "ssim": overall_summary["ssim"],
            "nrms": overall_summary["nrms"],
            "pearson": overall_summary["pearson"],
            "spatial_bias": overall_summary["bias"],
        },
        "trivial": {
            "mae": overall_summary["triv_mae"],
            "ssim": overall_summary["triv_ssim"],
        },
        "per_design": {
            d: {
                "n": per_design_counts[d],
                "mae": per_design_summary[d]["mae"],
                "ssim": per_design_summary[d]["ssim"],
                "nrms": per_design_summary[d]["nrms"],
                "pearson": per_design_summary[d]["pearson"],
                "spatial_bias": per_design_summary[d]["bias"],
                "trivial_mae": per_design_summary[d]["triv_mae"],
                "trivial_ssim": per_design_summary[d]["triv_ssim"],
            }
            for d in sorted(per_design_summary)
        },
        "latent": {
            "per_channel_std": ch_stds_avg.tolist(),
            "mean_std": float(np.mean(ch_stds_avg)),
            "min_std": float(np.min(ch_stds_avg)),
            "clamp_frac_low": frac_low,
            "clamp_frac_high": frac_high,
        },
    }

    out_path = os.path.join(out_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()