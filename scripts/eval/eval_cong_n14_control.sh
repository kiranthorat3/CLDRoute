#!/usr/bin/env bash
# Evaluate LDM+ControlNet on N14 Congestion test set — reproduces Table 12 results
set -e

VAE_DIR="${1:-./runs/vae_Cong_N14}"
CKPT="${2:-./runs/ldm_Cong_N14_control/best_gen.pt}"
OUT_DIR="${3:-./results/eval_cong_n14_control}"

python src/n14/ldm_control/latent_sampler.py \
    --ckpt      "$CKPT" \
    --vae_dir   "$VAE_DIR" \
    --split     test \
    --steps     100 \
    --eta       0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   "$OUT_DIR"

# Expected results (mean over 3 seeds):
#   MAE=0.03588  SSIM=0.7654  Pearson=0.0370
#   Spatial Bias=0.00297  Uncertainty=0.0197
