#!/usr/bin/env bash
# Evaluate LDM+ControlNet on N28 Congestion test set — reproduces Table 9 & 10 results
set -e

VAE_DIR="${1:-./runs/vae_Cong_N28}"
CKPT="${2:-./runs/ldm_Cong_N28_control/best_gen.pt}"
OUT_DIR="${3:-./results/eval_cong_n28_control}"

python src/ldm_control/latent_sampler.py \
    --ckpt      "$CKPT" \
    --vae_dir   "$VAE_DIR" \
    --split     test \
    --steps     100 \
    --eta       0.0 \
    --cfg_scale 0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   "$OUT_DIR"

# Expected results (mean over 3 seeds):
#   MAE=0.02859  NRMS=0.03430  SSIM=0.90310  Pearson=0.36870
#   NZ-Pearson=0.36923  Spatial Bias=-0.00441  Uncertainty=0.35920
