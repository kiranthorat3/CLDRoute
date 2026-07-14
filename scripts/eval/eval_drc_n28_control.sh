#!/usr/bin/env bash
# Evaluate LDM+ControlNet on N28 DRC test set — reproduces Table 7 & 8 results
# Downloads: kiranthorat/CLDRoute  checkpoints/n28/ldm_DRC_control_best_gen.pt
#                                  checkpoints/n28/vae_DRC_best_ldm.pt
set -e

VAE_DIR="${1:-./runs/vae_DRC_N28}"
CKPT="${2:-./runs/ldm_DRC_N28_control/best_gen.pt}"
OUT_DIR="${3:-./results/eval_drc_n28_control}"

python src/ldm_control/latent_sampler.py \
    --ckpt      "$CKPT" \
    --vae_dir   "$VAE_DIR" \
    --split     test \
    --steps     200 \
    --eta       0.0 \
    --cfg_scale 1.5 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   "$OUT_DIR"

# Expected results (mean over 3 seeds):
#   MAE=0.00280  NRMS=0.02893  SSIM=0.96780  Pearson=0.52483
#   TopK@1%=0.34940  TopK@0.5%=0.34830  Hotspot-MAE=0.05580
#   NZ-Pearson=0.46143  F1@0.1=0.44203  Uncertainty=0.5735
