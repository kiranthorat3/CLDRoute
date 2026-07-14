#!/usr/bin/env bash
# Train LDM + multi-scale ControlNet for Congestion — N28
# Requires: trained Congestion VAE checkpoint (best_ldm.pt)
# Expected runtime: ~10 h on one A6000 (200 epochs)
set -e

DATA_ROOT="${1:-/path/to/CircuitNet-N28/training_set_expanded}"
VAE_DIR="${2:-./runs/vae_Cong_N28}"
OUT_DIR="${3:-./runs/ldm_Cong_N28_control}"

python src/ldm_control/latent_trainer.py \
    --ae_ckpt  "$VAE_DIR/best_ldm.pt" \
    --vae_dir  "$VAE_DIR" \
    --out_dir  "$OUT_DIR"
