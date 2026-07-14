#!/usr/bin/env bash
# Evaluate LDM (unified) on N28 Congestion test set — reproduces Table 9 & 10 results
set -e

VAE_DIR="${1:-./runs/vae_Cong_N28}"
CKPT="${2:-./runs/ldm_Cong_N28_unified/best_gen.pt}"
OUT_DIR="${3:-./results/eval_cong_n28_unified}"

python src/ldm_unified/latent_sampler.py \
    --ckpt      "$CKPT" \
    --vae_dir   "$VAE_DIR" \
    --split     test \
    --steps     100 \
    --eta       0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   "$OUT_DIR"

# Expected results (mean over 3 seeds):
#   MAE=0.02914  NRMS=0.03380  SSIM=0.91223  Pearson=0.33127
#   NZ-Pearson=0.33167  Spatial Bias=-0.01259  Uncertainty=0.21233
