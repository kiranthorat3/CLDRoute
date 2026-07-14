#!/usr/bin/env bash
# Train LDM (unified, no ControlNet) for DRC — N28
# Requires: trained DRC VAE checkpoint (best_ldm.pt)
# Expected runtime: ~10 h on one A6000 (200 epochs)
set -e

DATA_ROOT="${1:-/path/to/CircuitNet-N28/training_set_expanded}"
VAE_DIR="${2:-./runs/vae_DRC_N28}"
OUT_DIR="${3:-./runs/ldm_DRC_N28_unified}"

python src/ldm_unified/latent_trainer.py \
    --ae_ckpt  "$VAE_DIR/best_ldm.pt" \
    --vae_dir  "$VAE_DIR" \
    --out_dir  "$OUT_DIR"
