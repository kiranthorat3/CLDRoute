#!/usr/bin/env bash
# Train Congestion VAE (N14)
set -e

DATA_ROOT="${1:-/path/to/CircuitNet-N14/training_set_expanded}"
OUT_DIR="${2:-./runs/vae_Cong_N14}"

python src/n14/cong_vae/vae_train_congestion.py \
    --out_dir  "$OUT_DIR" \
    --label_dir  "$DATA_ROOT/congestion/label" \
    --csv_train  "$DATA_ROOT/congestion/files_design/train_N14.csv" \
    --csv_val    "$DATA_ROOT/congestion/files_design/val_N14.csv" \
    --csv_test   "$DATA_ROOT/congestion/files_design/test_N14.csv"
