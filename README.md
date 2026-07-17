<div align="center">

# CLDRoute

### Conditional Latent Diffusion for Routability Map Generation in Physical Design

[![ICCAD 2026](https://img.shields.io/badge/ICCAD-2026-blue?style=for-the-badge)](https://iccad.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![HuggingFace Models](https://img.shields.io/badge/🤗_Models-CLDRoute-orange?style=for-the-badge)](https://huggingface.co/kiranthorat/CLDRoute)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-CLDRoute--dataset-orange?style=for-the-badge)](https://huggingface.co/datasets/kiranthorat/CLDRoute-dataset)

**Accepted at ICCAD 2026**

</div>

---

CLDRoute is a two-stage conditional latent diffusion framework for **routability map generation at placement stage**. It models DRC violation maps and routing congestion maps as stochastic spatial fields, producing both a **mean prediction** and a **per-pixel uncertainty map** from a single placed design — without running the router.

**Key properties:**
- Task-specific VAEs for sparse DRC fields (95% zeros) and dense congestion fields — no shared encoder
- Multi-scale ControlNet conditioning at 64×64, 32×32, 16×16 via ZeroConv projections
- Uncertainty quantification via N=8 DDIM samples per design
- Evaluated on CircuitNet 2.0 at two nodes: **N28** and **N14**

---

## Table of Contents

1. [Pipeline](#1-pipeline)
2. [Results](#2-results)
3. [Setup](#3-setup)
4. [Pretrained Checkpoints](#4-pretrained-checkpoints)
5. [Dataset](#5-dataset)
6. [Reproduce Paper Results](#6-reproduce-paper-results)
   - [Step 1 — Train DRC VAE](#step-1--train-drc-vae)
   - [Step 2 — Train Congestion VAE](#step-2--train-congestion-vae)
   - [Step 3 — Train LDM + ControlNet](#step-3--train-ldm--controlnet)
   - [Step 4 — Evaluate](#step-4--evaluate)
   - [Quick eval with pretrained checkpoints](#quick-eval-with-pretrained-checkpoints)
7. [Repository Layout](#7-repository-layout)
8. [VAE Design Details](#8-vae-design-details)
9. [Routing Control Features](#9-routing-control-features)
10. [Citation](#10-citation)

---

## 1. Pipeline

<div align="center">

| ![Pipeline placeholder](assets/Overview.pdf) |
|:---:|
| *Fig. 1 — Two-stage pipeline: physics-aware routing controls → task-specific VAE latent space → conditional DDIM denoising → mean map + uncertainty map* |

</div>

**Stage 1 — Task-specific VAE.**
DRC labels (sparse, skewed) and congestion labels (dense, narrow range) are encoded into separate latent spaces: 12×64×64 for DRC (focal + hotspot loss, β=0.05) and 8×64×64 for congestion (L1 + free-bits KL, β=0.005).

**Stage 2 — Conditional LDM.**
A U-Net with multi-scale ControlNet conditioning runs T=1000-step cosine DDIM in latent space. Physics-aware routing control features are projected to three spatial scales and injected via ZeroConv. At inference, N=8 samples yield a mean map and a spatial uncertainty map.

---

## 2. Results

All numbers are averaged over **3 seeds** (1234, 2345, 3456), **N=8 DDIM samples** per design, DDIM η=0.

### Table 6 — VAE Reconstruction Quality

| Node | Task | MAE ↓ | SSIM ↑ | Correlation ↑ |
|------|------|-------:|-------:|-------------:|
| N28 | DRC | 0.00095 | 0.9934 | 0.9870 |
| N28 | Congestion | 0.00132 | 0.9932 | 0.9608 |
| N14 | DRC | 0.00648 | 0.7009 | 0.2782 (NZ) |
| N14 | Congestion | 0.00390 | 0.9676 | 0.9305 |

### Tables 7 & 8 — N28 DRC Violation Map Generation

| Method | MAE ↓ | NRMS ↓ | SSIM ↑ | Pearson ↑ | TopK@1% ↑ | F1@0.1 ↑ |
|--------|-------:|-------:|-------:|----------:|----------:|---------:|
| Pixel Diffusion (9 ch) | 0.01961 | 0.20180 | 0.59270 | 0.28990 | 0.22073 | 0.15370 |
| LDM (15 ch) | 0.00292 | 0.03373 | 0.96470 | 0.50477 | 0.33837 | 0.40247 |
| **LDM + ControlNet (15 ch)** | **0.00280** | **0.02893** | **0.96780** | **0.52483** | **0.34940** | **0.44203** |

### Tables 9 & 10 — N28 Congestion Map Generation

| Method | MAE ↓ | NRMS ↓ | SSIM ↑ | Pearson ↑ | NZ-Pearson ↑ | Uncertainty ↑ |
|--------|-------:|-------:|-------:|----------:|-------------:|--------------:|
| Pixel Diffusion (3 ch) | 0.02730 | 0.03167 | 0.91990 | 0.30770 | 0.30883 | 0.24107 |
| LDM (10 ch) | 0.02915 | 0.03380 | 0.91223 | 0.33127 | 0.33167 | 0.21233 |
| **LDM + ControlNet (10 ch)** | **0.02859** | **0.03430** | **0.90310** | **0.36870** | **0.36923** | **0.35920** |

### Table 11 — N14 DRC Violation Map Generation

| Method | MAE ↓ | SSIM ↑ | TopK@1% ↑ | Hotspot-MAE ↓ | NZ-Pearson ↑ | Uncertainty ↑ |
|--------|-------:|-------:|----------:|--------------:|-------------:|--------------:|
| **LDM** | **0.00627** | **0.7136** | 0.0125 | **0.00571** | **0.0874** | **0.4972** |
| LDM + ControlNet | 0.00633 | 0.7089 | **0.0148** | 0.00578 | 0.0358 | 0.4329 |

### Table 12 — N14 Congestion Map Generation

| Method | MAE ↓ | SSIM ↑ | Pearson ↑ | Spatial Bias → 0 | Uncertainty ↑ |
|--------|-------:|-------:|----------:|-----------------:|--------------:|
| **LDM** | **0.03416** | **0.7678** | 0.0369 | −0.00472 | 0.0094 |
| LDM + ControlNet | 0.03588 | 0.7654 | **0.0370** | **0.00297** | **0.0197** |

<div align="center">

| ![Qualitative results placeholder](assets/fig5_qualitative_results.png) |
|:---:|
| *Fig. 5 — Mean map and spatial uncertainty map for DRC (top) and congestion (bottom) on N28 test designs* |

</div>

---

## 3. Setup

```bash
git clone https://github.com/kiranthorat3/CLDRoute.git
cd CLDRoute
pip install -r requirements.txt
```

**Tested on:** Python 3.10 · PyTorch 2.1 · CUDA 12.1 · 4× NVIDIA RTX A6000 48 GB

---

## 4. Pretrained Checkpoints

All checkpoints are on Hugging Face: [**kiranthorat/CLDRoute**](https://huggingface.co/kiranthorat/CLDRoute)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('kiranthorat/CLDRoute', local_dir='./checkpoints_hf')
"
```

| File | Description | Size |
|------|-------------|-----:|
| `checkpoints/n28/vae_DRC_best_ldm.pt` | N28 DRC VAE | 35 MB |
| `checkpoints/n28/vae_Cong_best_ldm.pt` | N28 Congestion VAE | 35 MB |
| `checkpoints/n28/ldm_DRC_control_best_gen.pt` | N28 LDM+ControlNet DRC | 347 MB |
| `checkpoints/n28/ldm_Cong_control_best_gen.pt` | N28 LDM+ControlNet Congestion | 347 MB |
| `checkpoints/n28/ldm_DRC_unified_best_gen.pt` | N28 LDM DRC (no ControlNet) | 327 MB |
| `checkpoints/n28/ldm_Cong_unified_best_gen.pt` | N28 LDM Congestion (no ControlNet) | 327 MB |
| `checkpoints/n14/vae_DRC_best_ldm.pt` | N14 DRC VAE | 35 MB |
| `checkpoints/n14/vae_Cong_best_ldm.pt` | N14 Congestion VAE | 35 MB |
| `checkpoints/n14/ldm_DRC_control_best_gen.pt` | N14 LDM+ControlNet DRC | 347 MB |
| `checkpoints/n14/ldm_Cong_control_best_gen.pt` | N14 LDM+ControlNet Congestion | 347 MB |
| `checkpoints/n14/ldm_DRC_unified_best_gen.pt` | N14 LDM DRC (no ControlNet) | 327 MB |
| `checkpoints/n14/ldm_Cong_unified_best_gen.pt` | N14 LDM Congestion (no ControlNet) | 327 MB |

---

## 5. Dataset

Pre-extracted physics-aware routing control features:
[**kiranthorat/CLDRoute-dataset**](https://huggingface.co/datasets/kiranthorat/CLDRoute-dataset)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='kiranthorat/CLDRoute-dataset',
    repo_type='dataset',
    local_dir='./data/features'
)
"
```

| Node | Task | Channels | Shape | Size |
|------|------|:--------:|-------|-----:|
| N28 | DRC | 16 ch (15 effective) | 256×256×16 float32 | ~41 GB |
| N28 | Congestion | 11 ch (10 effective) | 256×256×11 float32 | ~28 GB |
| N14 | DRC | 16 ch (15 effective) | 256×256×16 float32 | ~43 GB |
| N14 | Congestion | 11 ch (10 effective) | 256×256×11 float32 | ~30 GB |

Labels (256×256×1 float32) and design-wise CSV splits are included.

| Node | Train | Val | Test |
|------|------:|----:|-----:|
| N28 | 7,872 | 1,248 | 1,122 |
| N14 | 10,368 | 169 | 250 |

<div align="center">

| ![Routing controls placeholder](assets/fig2_routing_controls.png) |
|:---:|
| *Fig. 2 — Physics-aware routing control channels: demand (RUDY, RUDY_pin), supply (GR_util, eGR_overflow), geometry (cell_density, macro_region)* |

</div>

---

## 6. Reproduce Paper Results

> **Data path note:** LDM training reads the dataset root from `_EXPANDED_ROOT` in
> `src/ldm_control/latent_config.py` (and `src/ldm_unified/latent_config.py`).
> Set this to your local path before training. VAE scripts accept `--label_dir`
> and `--csv_*` directly as shown below.

---

### Step 1 — Train DRC VAE

Trains the DRC VAE (12×64×64 latent, focal + hotspot loss). Reproduces **Table 6, N28 DRC** row.

```bash
bash scripts/train/train_drc_vae_n28.sh \
    /path/to/CircuitNet-N28/training_set_expanded \
    ./runs/vae_DRC_N28
```

<details>
<summary>Full command with explicit hyperparameters</summary>

```bash
python src/drc_vae/vae_train.py \
    --out_dir        ./runs/vae_DRC_N28 \
    --label_dir      /path/to/CircuitNet-N28/training_set_expanded/DRC/label \
    --csv_train      /path/to/CircuitNet-N28/training_set_expanded/DRC/files_design/train_N28.csv \
    --csv_val        /path/to/CircuitNet-N28/training_set_expanded/DRC/files_design/val_N28.csv \
    --csv_test       /path/to/CircuitNet-N28/training_set_expanded/DRC/files_design/test_N28.csv \
    --latent_ch      12 \
    --beta_target    0.05 \
    --free_bits      0.5 \
    --hotspot_weight 0.5 \
    --epochs         150 \
    --batch_size     32 \
    --lr             1e-3 \
    --seed           42
```
</details>

Expected: `best_ldm.pt` → MAE≈0.00095, SSIM≈0.9934 on N28 test set (~6 h, one A6000)

---

### Step 2 — Train Congestion VAE

Trains the Congestion VAE (8×64×64 latent, L1 + free-bits KL). Reproduces **Table 6, N28 Congestion** row.

```bash
bash scripts/train/train_cong_vae_n28.sh \
    /path/to/CircuitNet-N28/training_set_expanded \
    ./runs/vae_Cong_N28
```

<details>
<summary>Full command with explicit hyperparameters</summary>

```bash
python src/cong_vae/vae_train_congestion.py \
    --out_dir     ./runs/vae_Cong_N28 \
    --label_dir   /path/to/CircuitNet-N28/training_set_expanded/congestion/label \
    --csv_train   /path/to/CircuitNet-N28/training_set_expanded/congestion/files_design/train_N28.csv \
    --csv_val     /path/to/CircuitNet-N28/training_set_expanded/congestion/files_design/val_N28.csv \
    --latent_ch   8 \
    --beta_target 0.005 \
    --free_bits   0.5 \
    --epochs      150 \
    --batch_size  32 \
    --lr          1e-3 \
    --seed        42
```
</details>

Expected: `best_ldm.pt` → MAE≈0.00132, SSIM≈0.9932 on N28 test set (~6 h, one A6000)

---

### Step 3 — Train LDM + ControlNet

**DRC** — Reproduces **Tables 7 & 8** (best method in paper).

```bash
# Set data root first:
# Edit _EXPANDED_ROOT in src/ldm_control/latent_config.py

bash scripts/train/train_ldm_drc_n28_control.sh \
    ./runs/vae_DRC_N28 \
    ./runs/ldm_DRC_N28_control
```

<details>
<summary>Full command with explicit hyperparameters</summary>

```bash
python src/ldm_control/latent_trainer.py \
    --ae_ckpt       ./runs/vae_DRC_N28/best_ldm.pt \
    --vae_dir       ./runs/vae_DRC_N28 \
    --out_dir       ./runs/ldm_DRC_N28_control \
    --epochs        200 \
    --batch_size    16 \
    --lr            1e-4 \
    --cfg_drop_prob 0.1 \
    --seed          42
```
</details>

Expected: `best_gen.pt` → MAE=0.00280, SSIM=0.9678, TopK@1%=0.3494 (~10 h, one A6000)

**Congestion** — Reproduces **Tables 9 & 10**.

```bash
bash scripts/train/train_ldm_cong_n28_control.sh \
    ./runs/vae_Cong_N28 \
    ./runs/ldm_Cong_N28_control
```

<details>
<summary>Full command with explicit hyperparameters</summary>

```bash
python src/ldm_control/latent_trainer.py \
    --ae_ckpt       ./runs/vae_Cong_N28/best_ldm.pt \
    --vae_dir       ./runs/vae_Cong_N28 \
    --out_dir       ./runs/ldm_Cong_N28_control \
    --epochs        200 \
    --batch_size    16 \
    --lr            1e-4 \
    --cfg_drop_prob 0.1 \
    --seed          42
```
</details>

Expected: `best_gen.pt` → MAE=0.02859, NZ-Pearson=0.3692 (~10 h, one A6000)

---

### Step 4 — Evaluate

#### N28 DRC — LDM + ControlNet (Tables 7 & 8)

```bash
python src/ldm_control/latent_sampler.py \
    --ckpt      ./runs/ldm_DRC_N28_control/best_gen.pt \
    --vae_dir   ./runs/vae_DRC_N28 \
    --split     test \
    --steps     200 \
    --eta       0.0 \
    --cfg_scale 1.5 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   ./results/drc_n28_control
```

Expected (mean over 3 seeds):
```
MAE=0.00280   NRMS=0.02893   SSIM=0.96780   Pearson=0.52483
TopK@1%=0.34940   TopK@0.5%=0.34830   Hotspot-MAE=0.05580
NZ-Pearson=0.46143   F1@0.1=0.44203   Uncertainty=0.5735
```

#### N28 Congestion — LDM + ControlNet (Tables 9 & 10)

```bash
python src/ldm_control/latent_sampler.py \
    --ckpt      ./runs/ldm_Cong_N28_control/best_gen.pt \
    --vae_dir   ./runs/vae_Cong_N28 \
    --split     test \
    --steps     100 \
    --eta       0.0 \
    --cfg_scale 0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   ./results/cong_n28_control
```

Expected:
```
MAE=0.02859   NRMS=0.03430   SSIM=0.90310   Pearson=0.36870
NZ-Pearson=0.36923   Spatial-Bias=-0.00441   Uncertainty=0.35920
```

#### N14 DRC — LDM unified (Table 11)

```bash
python src/n14/ldm_unified/latent_sampler.py \
    --ckpt      ./runs/ldm_DRC_N14_unified/best_gen.pt \
    --vae_dir   ./runs/vae_DRC_N14 \
    --split     test \
    --steps     200 \
    --eta       0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   ./results/drc_n14_unified
```

Expected:
```
MAE=0.00627   SSIM=0.7136   TopK@1%=0.0125   NZ-Pearson=0.0874
```

#### N14 Congestion — LDM unified (Table 12)

```bash
python src/n14/ldm_unified/latent_sampler.py \
    --ckpt      ./runs/ldm_Cong_N14_unified/best_gen.pt \
    --vae_dir   ./runs/vae_Cong_N14 \
    --split     test \
    --steps     100 \
    --eta       0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   ./results/cong_n14_unified
```

Expected:
```
MAE=0.03416   SSIM=0.7678   Pearson=0.0369   Spatial-Bias=-0.00472
```

---

### Quick eval with pretrained checkpoints

```bash
# 1. Download all checkpoints (~2.5 GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('kiranthorat/CLDRoute', local_dir='./ckpts')
"

# 2. Evaluate N28 DRC — reproduces Tables 7 & 8
python src/ldm_control/latent_sampler.py \
    --ckpt      ./ckpts/checkpoints/n28/ldm_DRC_control_best_gen.pt \
    --vae_dir   ./ckpts/checkpoints/n28/vae_DRC_best_ldm.pt \
    --split     test \
    --steps     200 \
    --eta       0.0 \
    --cfg_scale 1.5 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   ./results/drc_n28_pretrained

# 3. Evaluate N28 Congestion — reproduces Tables 9 & 10
python src/ldm_control/latent_sampler.py \
    --ckpt      ./ckpts/checkpoints/n28/ldm_Cong_control_best_gen.pt \
    --vae_dir   ./ckpts/checkpoints/n28/vae_Cong_best_ldm.pt \
    --split     test \
    --steps     100 \
    --eta       0.0 \
    --cfg_scale 0.0 \
    --N         8 \
    --seeds     1234 2345 3456 \
    --out_dir   ./results/cong_n28_pretrained
```

---

## 7. Repository Layout

```
CLDRoute/
├── src/
│   ├── drc_vae/                   # N28 DRC VAE
│   │   ├── vae_train.py           #   ← training entry point
│   │   ├── vae_eval.py            #   ← standalone evaluation
│   │   ├── vae_model.py           #   encoder / decoder / loss
│   │   ├── vae_data.py            #   dataset loader
│   │   ├── vae_config.py          #   hyperparameter dataclass
│   │   └── vae_latent_stats.py    #   compute μ/σ for LDM normalisation
│   │
│   ├── cong_vae/                  # N28 Congestion VAE
│   │   ├── vae_train_congestion.py
│   │   ├── vae_eval_congestion.py
│   │   ├── vae_model_congestion.py
│   │   ├── vae_data_congestion.py
│   │   └── vae_config_congestion.py
│   │
│   ├── ldm_control/               # N28 LDM + multi-scale ControlNet
│   │   ├── latent_trainer.py      #   ← training entry point
│   │   ├── latent_sampler.py      #   ← inference / evaluation entry point
│   │   ├── models.py              #   U-Net + ControlNet architecture
│   │   ├── diffusion.py           #   DDIM scheduler, v-prediction
│   │   ├── latent_config.py       #   hyperparameter dataclass
│   │   ├── latent_data.py         #   dataset loader
│   │   ├── utils_ema.py
│   │   └── utils_log.py
│   │
│   ├── ldm_unified/               # N28 LDM (single-scale conditioning)
│   │   └── ...                    #   same structure as ldm_control
│   │
│   ├── baseline/                  # Pixel-space diffusion baseline
│   │
│   └── n14/                       # N14 counterparts (identical structure)
│       ├── drc_vae/
│       ├── cong_vae/
│       ├── ldm_control/
│       └── ldm_unified/
│
├── scripts/
│   ├── train/                     # 12 training shell scripts
│   │   ├── train_drc_vae_n28.sh
│   │   ├── train_cong_vae_n28.sh
│   │   ├── train_ldm_drc_n28_control.sh
│   │   ├── train_ldm_cong_n28_control.sh
│   │   ├── train_ldm_drc_n28_unified.sh
│   │   ├── train_ldm_cong_n28_unified.sh
│   │   └── ... (N14 equivalents)
│   └── eval/                      # 8 evaluation shell scripts
│       ├── eval_drc_n28_control.sh
│       ├── eval_cong_n28_control.sh
│       └── ... (unified + N14 variants)
│
├── data/splits/                   # Design-wise CSV splits (N28 and N14)
│   ├── n28_train.csv
│   ├── n28_val.csv
│   ├── n28_test.csv
│   └── ...
│
└── assets/                        # Paper figures
```

---

## 8. VAE Design Details

| | DRC VAE | Congestion VAE |
|--|---------|----------------|
| Latent shape | 12 × 64 × 64 | 8 × 64 × 64 |
| Label sparsity | 95.4% zero pixels | 99.8% active pixels |
| Input transform | log(1 + 10x) | none |
| Reconstruction loss | Focal (γ=20) + Hotspot top-1% MSE | L1 |
| KL weight (β) | 0.05 | 0.005 |
| Free-bits (λ) | 0.5 nats/channel | 0.5 nats/channel |
| KL warmup | 15 epochs | 40 epochs |
| Batch size / LR | 32 / 1e-3 | 32 / 1e-3 |
| Epochs | 150 | 150 |
| Active channels | 12/12 · σ ∈ [0.630, 0.642] | 8/8 · σ ∈ [0.784, 0.801] |

<div align="center">

| ![DRC distribution placeholder](assets/fig3_drc_distribution.png) | ![Congestion distribution placeholder](assets/fig4_cong_distribution.png) |
|:---:|:---:|
| *Fig. 3 — DRC label distribution (95% zeros → focal loss)* | *Fig. 4 — Congestion label distribution (99.8% active → L1)* |

</div>

---

## 9. Routing Control Features

| Group | Congestion (10 ch effective) | DRC (15 ch effective) |
|-------|-----------------------------|-----------------------|
| **Demand** | RUDY, RUDY_pin, RUDY_long, RUDY_short | RUDY, RUDY_pin, RUDY_long, RUDY_short, RUDY_pin_long |
| **Supply** | GR_util_H/V, eGR_overflow_H/V | GR_util_H/V, GR_overflow_H/V, eGR_util_H/V, eGR_overflow_H/V |
| **Geometry** | macro_region, cell_density | macro_region, cell_density |

One dead channel is dropped per task (DRC ch 13, Congestion ch 6) — these carry zero variance across the training set and are excluded before feature projection.

---

## 10. Citation

If you use CLDRoute in your research, please cite:

```bibtex
@inproceedings{cldroute2026,
  title     = {{CLDRoute}: Conditional Latent Diffusion for Routability Map
               Generation in Physical Design},
  booktitle = {Proceedings of the IEEE/ACM International Conference on
               Computer-Aided Design (ICCAD)},
  year      = {2026}
}
```

---

## Acknowledgements

We use the [CircuitNet 2.0](https://circuitnet.github.io/) dataset (N28 and N14).
The diffusion backbone builds on ideas from
[LDM](https://github.com/CompVis/latent-diffusion) and
[ControlNet](https://github.com/lllyasviel/ControlNet).
