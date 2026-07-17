#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train LDM (no ControlNet, unified conditioning) for DRC — N28
# (Table 7: MAE=0.00292  SSIM=0.9647  Pearson=0.5048)
#
# Prerequisite : trained DRC VAE → $VAE_DIR/best_ldm.pt
# Runtime      : ~10 h on one A6000 (200 epochs, batch 16)
# ─────────────────────────────────────────────────────────────────────────────
set -e

VAE_DIR="${1:-./runs/vae_DRC_N28}"
OUT_DIR="${2:-./runs/ldm_DRC_N28_unified}"

python src/ldm_unified/latent_trainer.py \
    --ae_ckpt       "$VAE_DIR/best_ldm.pt" \
    --vae_dir       "$VAE_DIR" \
    --out_dir       "$OUT_DIR" \
    --epochs        200 \
    --batch_size    16 \
    --lr            1e-4 \
    --cfg_drop_prob 0.1 \
    --seed          42

# Best checkpoint: $OUT_DIR/best_gen.pt
