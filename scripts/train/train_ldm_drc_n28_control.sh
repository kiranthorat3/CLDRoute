#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train LDM + multi-scale ControlNet for DRC — N28
# (Tables 7 & 8: MAE=0.00280  SSIM=0.9678  TopK@1%=0.3494)
#
# Conditioning : multi-scale ControlNet at 64×64, 32×32, 16×16 via ZeroConv
# Diffusion    : T=1000 cosine schedule, v-prediction, min-SNR γ=5
# CFG          : drop_prob=0.1 during training  |  scale=1.5 at inference
# Runtime      : ~10 h on one A6000 (200 epochs, batch 16)
#
# Prerequisite : trained DRC VAE → $VAE_DIR/best_ldm.pt
# ─────────────────────────────────────────────────────────────────────────────
set -e

VAE_DIR="${1:-./runs/vae_DRC_N28}"
OUT_DIR="${2:-./runs/ldm_DRC_N28_control}"

# NOTE: data paths are set automatically from the task type stored in the VAE
# checkpoint. If your data is not at the default location, edit _EXPANDED_ROOT
# in src/ldm_control/latent_config.py before running.

python src/ldm_control/latent_trainer.py \
    --ae_ckpt      "$VAE_DIR/best_ldm.pt" \
    --vae_dir      "$VAE_DIR" \
    --out_dir      "$OUT_DIR" \
    --epochs       200 \
    --batch_size   16 \
    --lr           1e-4 \
    --cfg_drop_prob 0.1 \
    --seed         42

# Best checkpoint: $OUT_DIR/best_gen.pt
