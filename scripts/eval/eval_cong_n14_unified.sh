#!/usr/bin/env bash
# Evaluate LDM (unified) on N14 Congestion test set — reproduces Table 12 results
set -e

VAE_DIR="${1:-./runs/vae_Cong_N14}"
CKPT="${2:-./runs/ldm_Cong_N14_unified/best_gen.pt}"
OUT_DIR="${3:-./results/eval_cong_n14_unified}"

python src/n14/ldm_unified/latent_sampler.py \
    --ckpt      "$CKPT" \
    --vae_dir   "$VAE_DIR" \
    --split     test \
    --steps     100 \
    --eta       0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   "$OUT_DIR"

# Expected results (mean over 3 seeds):
#   MAE=0.03416  SSIM=0.7678  Pearson=0.0369
#   Spatial Bias=-0.00472  Uncertainty=0.0094
