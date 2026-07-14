#!/usr/bin/env python3
"""
vae_eval_congestion.py — Evaluate CongestionVAE on val or test split.

Primary:   MAE, SSIM (dense regression metrics)
Secondary: NRMSE, Pearson, spatial bias
Latent:    per-channel std, clamp fractions
Trivial:   constant training-mean predictor

Usage:
  python vae_eval_congestion.py --ckpt ./runs/vae_Cong_expanded/best_ldm.pt
  python vae_eval_congestion.py --ckpt ./runs/vae_Cong_expanded/best_ldm.pt --split val
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from vae_config_congestion import CongestionVAEConfig
from vae_model_congestion  import CongestionVAE, kl_loss
from vae_data_congestion   import CongestionLabelDataset, _collate
from vae_train_congestion  import EMA


def _safe_pearson(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser("CongestionVAE evaluation")
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))

    ck = torch.load(args.ckpt, map_location=device)
    assert ck.get("model_type") == "CongestionVAE_v2", \
        f"Expected CongestionVAE_v2, got {ck.get('model_type')}"

    model = CongestionVAE(
        C_label    = int(ck["C_label"]),
        latent_ch  = int(ck["latent_ch"]),
        base_ch    = int(ck.get("base_ch", 64)),
        logvar_min = float(ck.get("logvar_min", -4.0)),
        logvar_max = float(ck.get("logvar_max",  4.0)),
    ).to(device)
    model.load_state_dict(ck["net"], strict=True)

    if "ema" in ck:
        ema = EMA(model)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model)
        print("[Eval] EMA weights applied")
    model.eval()

    print(f"[Eval] epoch={ck['epoch']} | latent={ck['latent_ch']}×64×64 | "
          f"beta={ck.get('beta_target','?')}")

    cfg      = CongestionVAEConfig.load(os.path.join(ckpt_dir, "config.json"))
    csv_path = cfg.csv_test if args.split == "test" else cfg.csv_val

    # Training mean for trivial baseline
    ds_train = CongestionLabelDataset(
        cfg.csv_train, cfg.label_dir, split="train", verify=False)
    ldr_tmp  = DataLoader(ds_train, batch_size=64, shuffle=False,
                          num_workers=args.num_workers,
                          collate_fn=_collate, drop_last=False)
    s, n = 0.0, 0
    with torch.no_grad():
        for lbl, _ in ldr_tmp:
            s += float(lbl.sum()); n += lbl.numel()
    train_mean = s / max(n, 1)
    print(f"[Eval] Training label mean={train_mean:.4f}")

    ds = CongestionLabelDataset(csv_path, cfg.label_dir,
                                split=args.split, verify=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=_collate, drop_last=False)
    print(f"[Eval] split={args.split} | n={len(ds)}")

    C = ck["latent_ch"]
    ch_mu_stds  = []
    logvar_all  = []
    mae_l, ssim_l, nrms_l, pear_l, bias_l = [], [], [], [], []
    triv_mae_l, triv_ssim_l = [], []

    with torch.no_grad():
        for lbl, _ in loader:
            lbl = lbl.float().to(device)
            recon, mu, logvar, z = model(lbl, sample=False)

            ch_mu_stds.append(mu.std(dim=(0, 2, 3)).cpu().numpy())
            logvar_all.append(logvar.detach().cpu().float())

            gt_np = lbl.cpu().numpy()[:, 0]
            pr_np = recon.cpu().numpy()[:, 0]
            for i in range(gt_np.shape[0]):
                gt_i, pr_i = gt_np[i], pr_np[i]
                mae  = float(np.mean(np.abs(gt_i - pr_i)))
                rmse = float(np.sqrt(np.mean((gt_i - pr_i)**2)))
                rng  = float(gt_i.max() - gt_i.min())
                nrms = (rmse / rng) if rng > 1e-8 else float("nan")
                ssim = float(structural_similarity(gt_i, pr_i, data_range=1.0))
                pear = _safe_pearson(gt_i, pr_i)
                bias = float(np.mean(pr_i) - np.mean(gt_i))
                mae_l.append(mae); ssim_l.append(ssim)
                nrms_l.append(nrms); pear_l.append(pear); bias_l.append(bias)
                triv = np.full_like(gt_i, train_mean)
                triv_mae_l.append(float(np.mean(np.abs(gt_i - triv))))
                triv_ssim_l.append(float(structural_similarity(
                    gt_i, triv, data_range=1.0)))

    ch_stds_avg = np.mean(ch_mu_stds, axis=0)
    lv_cat      = torch.cat([lv.reshape(-1) for lv in logvar_all])
    frac_low    = float((lv_cat <= float(ck.get("logvar_min", -4.0)) + 1e-4).float().mean())
    frac_high   = float((lv_cat >= float(ck.get("logvar_max",  4.0)) - 1e-4).float().mean())

    print("\n" + "=" * 65)
    print(f"  CongestionVAE EVAL — {args.split}")
    print("=" * 65)
    print(f"\n  Reconstruction (model vs trivial mean={train_mean:.4f}):")
    print(f"  {'MAE ↓':<22} {np.mean(mae_l):>10.5f}  "
          f"{np.mean(triv_mae_l):>10.5f}  (trivial)")
    print(f"  {'SSIM ↑':<22} {np.mean(ssim_l):>10.4f}  "
          f"{np.mean(triv_ssim_l):>10.4f}  (trivial)")
    print(f"  {'NRMSE ↓':<22} {np.nanmean(nrms_l):>10.4f}")
    print(f"  {'Pearson ↑':<22} {np.nanmean(pear_l):>10.4f}")
    print(f"  {'Spatial bias':<22} {np.mean(bias_l):>10.5f}  (target: ~0)")
    print(f"\n  Latent (per-channel std over {args.split} set):")
    print(f"  {'Ch':<6} {'Std':>8}  Status")
    for c in range(len(ch_stds_avg)):
        status = "✓" if ch_stds_avg[c] >= 0.3 else "⚠ weak"
        print(f"  {c:<6} {ch_stds_avg[c]:>8.4f}  {status}")
    print(f"  Global: mean={np.mean(ch_stds_avg):.4f}  "
          f"min={np.min(ch_stds_avg):.4f}")
    print(f"  Clamp: low={frac_low:.3f}  high={frac_high:.3f}")
    print("=" * 65)

    out_dir = os.path.join(ckpt_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "ckpt": args.ckpt, "epoch": ck["epoch"],
        "split": args.split, "n": len(mae_l),
        "train_mean": train_mean,
        "metrics": {
            "mae":         float(np.mean(mae_l)),
            "ssim":        float(np.mean(ssim_l)),
            "nrms":        float(np.nanmean(nrms_l)),
            "pearson":     float(np.nanmean(pear_l)),
            "spatial_bias":float(np.mean(bias_l)),
        },
        "trivial": {
            "mae":  float(np.mean(triv_mae_l)),
            "ssim": float(np.mean(triv_ssim_l)),
        },
        "latent": {
            "per_channel_std": ch_stds_avg.tolist(),
            "mean_std":        float(np.mean(ch_stds_avg)),
            "min_std":         float(np.min(ch_stds_avg)),
            "clamp_frac_low":  frac_low,
            "clamp_frac_high": frac_high,
        },
    }
    out_path = os.path.join(out_dir, f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Saved] {out_path}")

if __name__ == "__main__":
    main()