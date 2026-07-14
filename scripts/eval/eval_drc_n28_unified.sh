#!/usr/bin/env bash
# Evaluate LDM (unified) on N28 DRC test set — reproduces Table 7 & 8 results
set -e

VAE_DIR="${1:-./runs/vae_DRC_N28}"
CKPT="${2:-./runs/ldm_DRC_N28_unified/best_gen.pt}"
OUT_DIR="${3:-./results/eval_drc_n28_unified}"

python src/ldm_unified/latent_sampler.py \
    --ckpt      "$CKPT" \
    --vae_dir   "$VAE_DIR" \
    --split     test \
    --steps     200 \
    --eta       0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   "$OUT_DIR"

# Expected results (mean over 3 seeds):
#   MAE=0.00292  NRMS=0.03373  SSIM=0.96470  Pearson=0.50477
#   TopK@1%=0.33837  TopK@0.5%=0.33250  Hotspot-MAE=0.06013
#   NZ-Pearson=0.44733  F1@0.1=0.40247  Uncertainty=0.5954
