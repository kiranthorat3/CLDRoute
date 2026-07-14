#!/usr/bin/env bash
# Train DRC VAE (N28) — 12×64×64 latent, focal + hotspot loss
# Expected runtime: ~6 h on one A6000 (150 epochs)
set -e

DATA_ROOT="${1:-/path/to/CircuitNet-N28/training_set_expanded}"
OUT_DIR="${2:-./runs/vae_DRC_N28}"

python src/drc_vae/vae_train.py \
    --out_dir  "$OUT_DIR" \
    --label_dir  "$DATA_ROOT/DRC/label" \
    --csv_train  "$DATA_ROOT/DRC/files_design/train_N28.csv" \
    --csv_val    "$DATA_ROOT/DRC/files_design/val_N28.csv" \
    --csv_test   "$DATA_ROOT/DRC/files_design/test_N28.csv"
