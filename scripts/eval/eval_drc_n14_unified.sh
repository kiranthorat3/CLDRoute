#!/usr/bin/env bash
# Evaluate LDM (unified) on N14 DRC test set — reproduces Table 11 results
set -e

VAE_DIR="${1:-./runs/vae_DRC_N14}"
CKPT="${2:-./runs/ldm_DRC_N14_unified/best_gen.pt}"
OUT_DIR="${3:-./results/eval_drc_n14_unified}"

python src/n14/ldm_unified/latent_sampler.py \
    --ckpt      "$CKPT" \
    --vae_dir   "$VAE_DIR" \
    --split     test \
    --steps     200 \
    --eta       0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   "$OUT_DIR"

# Expected results (mean over 3 seeds):
#   MAE=0.00627  SSIM=0.7136  TopK@1%=0.0125  Hotspot-MAE=0.00571
#   NZ-Pearson=0.0874  Uncertainty=0.4972
