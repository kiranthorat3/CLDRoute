#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train DRC VAE — N28  (Table 6: MAE=0.00095  SSIM=0.9934  Pearson=0.9870)
#
# Architecture : 12×64×64 latent  |  log(1+10x) input transform
# Loss         : Focal (γ=20) + Hotspot top-1% MSE  |  β=0.05 KL
# Runtime      : ~6 h on one A6000 (150 epochs, batch 32)
# ─────────────────────────────────────────────────────────────────────────────
set -e

DATA_ROOT="${1:-/path/to/CircuitNet-N28/training_set_expanded}"
OUT_DIR="${2:-./runs/vae_DRC_N28}"

python src/drc_vae/vae_train.py \
    --out_dir        "$OUT_DIR" \
    --label_dir      "$DATA_ROOT/DRC/label" \
    --csv_train      "$DATA_ROOT/DRC/files_design/train_N28.csv" \
    --csv_val        "$DATA_ROOT/DRC/files_design/val_N28.csv" \
    --csv_test       "$DATA_ROOT/DRC/files_design/test_N28.csv" \
    --latent_ch      12 \
    --beta_target    0.05 \
    --free_bits      0.5 \
    --hotspot_weight 0.5 \
    --epochs         150 \
    --batch_size     32 \
    --lr             1e-3 \
    --seed           42

# Best checkpoint: $OUT_DIR/best_ldm.pt  — use this for LDM training
