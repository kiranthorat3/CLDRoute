#!/usr/bin/env bash
# Train DRC VAE (N14)
set -e

DATA_ROOT="${1:-/path/to/CircuitNet-N14/training_set_expanded}"
OUT_DIR="${2:-./runs/vae_DRC_N14}"

python src/n14/drc_vae/vae_train.py \
    --out_dir  "$OUT_DIR" \
    --label_dir  "$DATA_ROOT/DRC/label" \
    --csv_train  "$DATA_ROOT/DRC/files_design/train_N14.csv" \
    --csv_val    "$DATA_ROOT/DRC/files_design/val_N14.csv" \
    --csv_test   "$DATA_ROOT/DRC/files_design/test_N14.csv"
