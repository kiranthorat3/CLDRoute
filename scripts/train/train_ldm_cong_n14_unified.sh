#!/usr/bin/env bash
set -e
VAE_DIR="${1:-./runs/vae_Cong_N14}"
OUT_DIR="${2:-./runs/ldm_Cong_N14_unified}"
python src/n14/ldm_unified/latent_trainer.py \
    --ae_ckpt "$VAE_DIR/best_ldm.pt" --vae_dir "$VAE_DIR" --out_dir "$OUT_DIR"
