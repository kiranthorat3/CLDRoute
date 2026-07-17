#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train Congestion VAE — N28  (Table 6: MAE=0.00132  SSIM=0.9932  Pearson=0.9608)
#
# Architecture : 8×64×64 latent  |  no input transform
# Loss         : L1 reconstruction  |  β=0.005 KL  |  free-bits λ=0.5 nats
# Runtime      : ~6 h on one A6000 (150 epochs, batch 32)
# ─────────────────────────────────────────────────────────────────────────────
set -e

DATA_ROOT="${1:-/path/to/CircuitNet-N28/training_set_expanded}"
OUT_DIR="${2:-./runs/vae_Cong_N28}"

python src/cong_vae/vae_train_congestion.py \
    --out_dir     "$OUT_DIR" \
    --label_dir   "$DATA_ROOT/congestion/label" \
    --csv_train   "$DATA_ROOT/congestion/files_design/train_N28.csv" \
    --csv_val     "$DATA_ROOT/congestion/files_design/val_N28.csv" \
    --latent_ch   8 \
    --beta_target 0.005 \
    --free_bits   0.5 \
    --epochs      150 \
    --batch_size  32 \
    --lr          1e-3 \
    --seed        42

# Best checkpoint: $OUT_DIR/best_ldm.pt  — use this for LDM training
