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

## Abstract

Accurate routability estimation during physical design is important for reducing costly post-routing iterations. Prior learning-based methods treat this task as deterministic prediction, mapping placement-stage features to a single congestion or DRC outcome. We instead formulate routability estimation as a *conditional generation* problem, where both routing congestion and DRC violations are modeled as spatially structured *routability fields*.

Our framework, **C**onditional **L**atent **D**iffusion for **Route**ability estimation (**CLDRoute**), uses physics-aware conditioning and task-specific latent modeling to handle the different characteristics of congestion and DRC maps. This allows our method to support sample-based inference, producing both a mean prediction and a spatial uncertainty estimate for the same input design.

On CircuitNet 2.0 (N28), CLDRoute achieves for DRC violation generation an SSIM of **0.9678**, MAE of **0.0028**, and TopK@1% of **0.3494**; for congestion generation an SSIM of **0.9031**, MAE of **0.0286**, and NZ-Pearson of **0.3692**. Overall, our framework provides a more practical view of routability at placement by generating both the expected outcome and its uncertainty.

---

## Framework Overview

<div align="center">

**[📄 View Framework Overview Figure (PDF)](assets/Overview.pdf)**

</div>

CLDRoute is a two-stage pipeline:

1. **Task-specific VAE** — DRC maps (95% zeros, sparse) and congestion maps (99.8% active, dense) are encoded into separate latent spaces: **12×64×64** for DRC (focal loss + hotspot MSE, β=0.05) and **8×64×64** for congestion (L1 + free-bits KL, β=0.005).

2. **Conditional LDM** — A U-Net with multi-scale ControlNet conditioning denoises in latent space over T=1000 DDIM steps. Physics-aware routing control features (demand · supply · geometry) are projected at three scales (64×64, 32×32, 16×16) via ZeroConv. N=8 samples at inference yield a **mean map** and a **per-pixel uncertainty map**.

| Figure | Link |
|--------|------|
| Framework overview | [assets/Overview.pdf](assets/Overview.pdf) |
| Generative architecture | [assets/Geneartive_arch.pdf](assets/Geneartive_arch.pdf) |
| DRC channel visualisation | [assets/drc_channels.pdf](assets/drc_channels.pdf) |
| DRC comparison | [assets/DRC_comparison.pdf](assets/DRC_comparison.pdf) |
| Label distributions | [assets/fig_label_distributions.pdf](assets/fig_label_distributions.pdf) |

---

## Results

All metrics averaged over **3 seeds** {1234, 2345, 3456} with **N=8 DDIM samples**, η=0.

### N28 — DRC Violation Map Generation

| Method | MAE ↓ | SSIM ↑ | Pearson ↑ | TopK@1% ↑ | F1@0.1 ↑ |
|--------|-------:|-------:|----------:|----------:|---------:|
| Pixel Diffusion (9 ch) | 0.01961 | 0.5927 | 0.2899 | 0.2207 | 0.1537 |
| LDM (15 ch) | 0.00292 | 0.9647 | 0.5048 | 0.3384 | 0.4025 |
| **LDM + ControlNet (15 ch)** | **0.00280** | **0.9678** | **0.5248** | **0.3494** | **0.4420** |

### N28 — Congestion Map Generation

| Method | MAE ↓ | SSIM ↑ | Pearson ↑ | NZ-Pearson ↑ | Uncertainty ↑ |
|--------|-------:|-------:|----------:|-------------:|--------------:|
| Pixel Diffusion (3 ch) | 0.02730 | 0.9199 | 0.3077 | 0.3088 | 0.2411 |
| LDM (10 ch) | 0.02915 | 0.9122 | 0.3313 | 0.3317 | 0.2123 |
| **LDM + ControlNet (10 ch)** | **0.02859** | **0.9031** | **0.3687** | **0.3692** | **0.3592** |

### N14

| Node | Task | Best method | MAE ↓ | SSIM ↑ |
|------|------|-------------|-------:|-------:|
| N14 | DRC | LDM | 0.00627 | 0.7136 |
| N14 | Congestion | LDM | 0.03416 | 0.7678 |

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/kiranthorat3/CLDRoute.git
cd CLDRoute
pip install -r requirements.txt
```

**Tested on:** Python 3.10 · PyTorch 2.1 · CUDA 12.1 · NVIDIA A6000 48 GB

### 2. Download pretrained checkpoints (~2.5 GB)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('kiranthorat/CLDRoute', local_dir='./ckpts')
"
```

### 3. Evaluate — N28 DRC (reproduces Tables 7 & 8)

```bash
python src/ldm_control/latent_sampler.py \
    --ckpt      ./ckpts/checkpoints/n28/ldm_DRC_control_best_gen.pt \
    --vae_dir   ./ckpts/checkpoints/n28/vae_DRC_best_ldm.pt \
    --split     test  --steps 200  --eta 0.0  --cfg_scale 1.5 \
    --N 8  --seeds 1234 2345 3456  --out_dir ./results/drc_n28
# Expected: MAE=0.00280  SSIM=0.9678  TopK@1%=0.3494
```

### 4. Evaluate — N28 Congestion (reproduces Tables 9 & 10)

```bash
python src/ldm_control/latent_sampler.py \
    --ckpt      ./ckpts/checkpoints/n28/ldm_Cong_control_best_gen.pt \
    --vae_dir   ./ckpts/checkpoints/n28/vae_Cong_best_ldm.pt \
    --split     test  --steps 100  --eta 0.0  --cfg_scale 0.0 \
    --N 8  --seeds 1234 2345 3456  --out_dir ./results/cong_n28
# Expected: MAE=0.02859  SSIM=0.9031  NZ-Pearson=0.3692
```

---

## Training from Scratch

> **Data:** Download physics-aware routing features from [kiranthorat/CLDRoute-dataset](https://huggingface.co/datasets/kiranthorat/CLDRoute-dataset) (~152 GB).
> Set `_EXPANDED_ROOT` in `src/ldm_control/latent_config.py` to your local data path before LDM training.

```bash
# Step 1 — DRC VAE  (~6 h, one A6000)
bash scripts/train/train_drc_vae_n28.sh  /path/to/data  ./runs/vae_DRC_N28

# Step 2 — Congestion VAE  (~6 h, one A6000)
bash scripts/train/train_cong_vae_n28.sh /path/to/data  ./runs/vae_Cong_N28

# Step 3 — LDM + ControlNet DRC  (~10 h, one A6000)
bash scripts/train/train_ldm_drc_n28_control.sh  ./runs/vae_DRC_N28   ./runs/ldm_DRC_N28_control

# Step 4 — LDM + ControlNet Congestion  (~10 h, one A6000)
bash scripts/train/train_ldm_cong_n28_control.sh ./runs/vae_Cong_N28  ./runs/ldm_Cong_N28_control
```

N14 equivalents follow the same pattern — see `scripts/train/`.

All scripts pass explicit hyperparameters. Key settings:

| Component | latent ch | Loss | β | epochs | batch | lr |
|-----------|:---------:|------|:-:|-------:|------:|---:|
| DRC VAE | 12 | Focal (γ=20) + Hotspot | 0.05 | 150 | 32 | 1e-3 |
| Cong VAE | 8 | L1 + free-bits (λ=0.5) | 0.005 | 150 | 32 | 1e-3 |
| LDM+ControlNet | — | v-prediction, min-SNR γ=5 | — | 200 | 16 | 1e-4 |

---

## Dataset

Physics-aware routing control features extracted from [CircuitNet 2.0](https://circuitnet.github.io/):

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

| Node | Task | Shape | Size |
|------|------|-------|-----:|
| N28 | DRC (15 ch) | 256×256×16 float32 | ~41 GB |
| N28 | Congestion (10 ch) | 256×256×11 float32 | ~28 GB |
| N14 | DRC (15 ch) | 256×256×16 float32 | ~43 GB |
| N14 | Congestion (10 ch) | 256×256×11 float32 | ~30 GB |

---

## Citation

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

We use the [CircuitNet 2.0](https://circuitnet.github.io/) dataset.
Diffusion backbone inspired by [LDM](https://github.com/CompVis/latent-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet).
