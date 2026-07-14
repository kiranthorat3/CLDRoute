#!/usr/bin/env bash
# Train Congestion VAE (N28) — 8×64×64 latent, L1 loss + free-bits KL
# Expected runtime: ~6 h on one A6000 (100 epochs)
set -e

DATA_ROOT="${1:-/path/to/CircuitNet-N28/training_set_expanded}"
OUT_DIR="${2:-./runs/vae_Cong_N28}"

python src/cong_vae/vae_train_congestion.py \
    --out_dir  "$OUT_DIR" \
    --label_dir  "$DATA_ROOT/congestion/label" \
    --csv_train  "$DATA_ROOT/congestion/files_design/train_N28.csv" \
    --csv_val    "$DATA_ROOT/congestion/files_design/val_N28.csv" \
    --csv_test   "$DATA_ROOT/congestion/files_design/test_N28.csv"
