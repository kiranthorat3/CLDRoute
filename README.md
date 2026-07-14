# CLDRoute: Conditional Latent Diffusion for Routability Map Generation in Physical Design

[![ICCAD 2026](https://img.shields.io/badge/ICCAD-2026-blue)](https://iccad.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20Models-kiranthorat/CLDRoute-orange)](https://huggingface.co/kiranthorat/CLDRoute)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-kiranthorat/CLDRoute--dataset-orange)](https://huggingface.co/datasets/kiranthorat/CLDRoute-dataset)

> **Accepted at ICCAD 2026**

CLDRoute is a conditional latent diffusion framework for routability map generation at the placement stage of physical design. Instead of treating routability as a deterministic regression problem, CLDRoute models both **routing congestion** and **DRC violation maps** as stochastic routability fields conditioned on physics-aware placement-stage features.

**Key capabilities:**
- Generates both a **mean prediction** and a **spatial uncertainty map** from the same placed design
- Handles the fundamentally different statistics of dense congestion fields and sparse DRC violation maps through task-specific latent encoding
- Uses multi-scale ControlNet-style conditioning with routing-relevant physical signals (demand · supply · geometry)
- Evaluated on CircuitNet 2.0 at two technology nodes: **N28** and **N14**

---

## Framework Overview

<!-- Figure 1: Replace with assets/fig1_pipeline.png once available -->
> **Figure 1 — placeholder**
> *Pipeline diagram: feature extraction → multi-scale conditioner → task-specific VAE → conditional U-Net denoiser → mean map + uncertainty map.*

![CLDRoute Pipeline](assets/fig1_pipeline.png)

The framework proceeds in three stages:
1. **Physics-aware routing controls** extracted from DEF/LEF files are projected to three spatial scales (64×64, 32×32, 16×16)
2. **Task-specific VAEs** compress DRC and congestion labels into separate latent spaces matched to their statistical properties
3. **Conditional latent diffusion** denoises in latent space guided by multi-scale ControlNet injections; N=8 samples produce both a mean routability map and a spatial uncertainty estimate

---

## Routing Control Examples

<!-- Figure 2: Replace with assets/fig2_routing_controls.png once available -->
> **Figure 2 — placeholder**
> *Four representative routing-control channels for one test design: (a) cell density, (b) RUDY\_pin, (c) GR\_overflow\_V, (d) eGR\_util\_V.*

![Routing Controls](assets/fig2_routing_controls.png)

---

## Label Statistics

<!-- Figures 3 & 4: Replace once available -->
> **Figure 3 (left) — placeholder:** DRC label distribution — 95.4% zero pixels, sparse violation structure.
> **Figure 4 (right) — placeholder:** Congestion label distribution — 99.8% active pixels, narrow value range.

| ![DRC Distribution](assets/fig3_drc_distribution.png) | ![Congestion Distribution](assets/fig4_cong_distribution.png) |
|:---:|:---:|
| *Fig. 3 — DRC label distribution* | *Fig. 4 — Congestion label distribution* |

---

## Qualitative Results

<!-- Figure 5: Replace with assets/fig5_qualitative_results.png once available -->
> **Figure 5 — placeholder**
> *Generated mean map and spatial uncertainty map for DRC (top) and congestion (bottom) on N28 test designs.*

![Qualitative Results](assets/fig5_qualitative_results.png)

---

## Results

All numbers are averaged over 3 random seeds (1234, 2345, 3456) with N=8 DDIM samples per design.

### Stage 1 — VAE Reconstruction Quality (Table 6)

| Dataset | Task | MAE ↓ | SSIM ↑ | Correlation ↑ |
|---------|------|-------:|-------:|-------------:|
| N28 | DRC | 0.00095 | 0.9934 | 0.9870 |
| N28 | Congestion | 0.00132 | 0.9932 | 0.9608 |
| N14 | DRC | 0.00648 | 0.7009 | 0.2782 (NZ) |
| N14 | Congestion | 0.00390 | 0.9676 | 0.9305 |

### N28 — DRC Violation Map Generation

**Standard metrics (Table 7)**

| Method | MAE ↓ | NRMS ↓ | SSIM ↑ | Pearson ↑ |
|--------|-------:|-------:|-------:|----------:|
| Pixel Diffusion (9 ch) | 0.01961 | 0.20180 | 0.59270 | 0.28990 |
| LDM (15 ch) | 0.00292 | 0.03373 | 0.96470 | 0.50477 |
| **LDM + ControlNet (15 ch)** | **0.00280** | **0.02893** | **0.96780** | **0.52483** |

**Task-specific metrics (Table 8)**

| Method | TopK@1% ↑ | TopK@0.5% ↑ | Hotspot-MAE ↓ | NZ-Pearson ↑ | F1@0.1 ↑ | Uncertainty ↑ |
|--------|----------:|------------:|--------------:|-------------:|---------:|--------------:|
| Pixel Diffusion (9 ch) | 0.22073 | 0.24693 | 0.07683 | 0.46287 | 0.15370 | 0.7835 |
| LDM (15 ch) | 0.33837 | 0.33250 | 0.06013 | 0.44733 | 0.40247 | 0.5954 |
| **LDM + ControlNet (15 ch)** | **0.34940** | **0.34830** | **0.05580** | **0.46143** | **0.44203** | 0.5735 |

### N28 — Congestion Map Generation

**Standard metrics (Table 9)**

| Method | MAE ↓ | NRMS ↓ | SSIM ↑ | Pearson ↑ |
|--------|-------:|-------:|-------:|----------:|
| Pixel Diffusion (3 ch) | 0.02730 | 0.03167 | 0.91990 | 0.30770 |
| LDM (10 ch) | 0.02915 | 0.03380 | 0.91223 | 0.33127 |
| **LDM + ControlNet (10 ch)** | 0.02859 | 0.03430 | 0.90310 | **0.36870** |

**Task-specific metrics (Table 10)**

| Method | NZ-Pearson ↑ | Spatial Bias → 0 | Uncertainty ↑ |
|--------|-------------:|-----------------:|--------------:|
| Pixel Diffusion (3 ch) | 0.30883 | 0.00708 | 0.24107 |
| LDM (10 ch) | 0.33167 | −0.01259 | 0.21233 |
| **LDM + ControlNet (10 ch)** | **0.36923** | **−0.00441** | **0.35920** |

### N14 — DRC Violation Map Generation (Table 11)

| Method | MAE ↓ | SSIM ↑ | TopK@1% ↑ | Hotspot-MAE ↓ | NZ-Pearson ↑ | Uncertainty ↑ |
|--------|-------:|-------:|----------:|--------------:|-------------:|--------------:|
| **LDM** | **0.00627** | **0.7136** | 0.0125 | **0.00571** | **0.0874** | **0.4972** |
| LDM + ControlNet | 0.00633 | 0.7089 | **0.0148** | 0.00578 | 0.0358 | 0.4329 |

### N14 — Congestion Map Generation (Table 12)

| Method | MAE ↓ | SSIM ↑ | Pearson ↑ | Spatial Bias → 0 | Uncertainty ↑ |
|--------|-------:|-------:|----------:|-----------------:|--------------:|
| **LDM** | **0.03416** | **0.7678** | 0.0369 | −0.00472 | 0.0094 |
| LDM + ControlNet | 0.03588 | 0.7654 | **0.0370** | **0.00297** | **0.0197** |

---

## Repository Layout

```
CLDRoute/
├── src/
│   ├── drc_vae/          # N28 DRC VAE (12×64×64 latent, focal+hotspot loss)
│   ├── cong_vae/         # N28 Congestion VAE (8×64×64 latent, L1+free-bits KL)
│   ├── ldm_control/      # N28 LDM + multi-scale ControlNet
│   ├── ldm_unified/      # N28 LDM (single-scale conditioning)
│   ├── baseline/         # Pixel-space diffusion baseline
│   └── n14/              # N14 counterparts (same structure)
├── scripts/
│   ├── train/            # One shell script per training stage
│   └── eval/             # One shell script per evaluation setting
├── data/splits/          # Design-wise CSV splits (N28 and N14)
└── assets/               # Figures (pipeline, routing controls, results)
```

---

## Setup

```bash
git clone https://github.com/kiranthorat/CLDRoute.git
cd CLDRoute
pip install -r requirements.txt
```

**Tested on:** Python 3.10, PyTorch 2.1, CUDA 12.1, 4× NVIDIA RTX A6000 (48 GB)

---

## Dataset

We provide our pre-extracted physics-aware routing control features on Hugging Face: [kiranthorat/CLDRoute-dataset](https://huggingface.co/datasets/kiranthorat/CLDRoute-dataset)

These are computed from [CircuitNet 2.0](https://circuitnet.github.io/) at N28 and N14.

| Node | Task | Features | Shape | Size |
|------|------|----------|-------|-----:|
| N28 | DRC | 16 ch | 256×256×16 float32 | 41 GB |
| N28 | Congestion | 11 ch | 256×256×11 float32 | 28 GB |
| N14 | DRC | 16 ch | 256×256×16 float32 | 43 GB |
| N14 | Congestion | 11 ch | 256×256×11 float32 | 30 GB |

Labels (256×256×1 float32) and design-wise CSV splits are included.

| Node | Train | Val | Test |
|------|------:|----:|-----:|
| N28  | 7,872 | 1,248 | 1,122 |
| N14  | 10,368 | 169 | 250 |

Download with:
```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="kiranthorat/CLDRoute-dataset",
    repo_type="dataset",
    local_dir="./data/features"
)
EOF
```

---

## Physics-Aware Routing Controls

| Group | Congestion (10 ch effective) | DRC (15 ch effective) |
|-------|-----------------------------|-----------------------|
| Demand | RUDY, RUDY\_pin, RUDY\_long, RUDY\_short | RUDY, RUDY\_pin, RUDY\_long, RUDY\_short, RUDY\_pin\_long |
| Supply | GR\_util\_H/V, eGR\_overflow\_H/V | GR\_util\_H/V, GR\_overflow\_H/V, eGR\_util\_H/V, eGR\_overflow\_H/V |
| Geometry | macro\_region, cell\_density | macro\_region, cell\_density |

---

## Pretrained Models

All checkpoints: [kiranthorat/CLDRoute](https://huggingface.co/kiranthorat/CLDRoute)

| File | Description | Size |
|------|-------------|-----:|
| `checkpoints/n28/vae_DRC_best_ldm.pt` | N28 DRC VAE | 35 MB |
| `checkpoints/n28/vae_Cong_best_ldm.pt` | N28 Congestion VAE | 35 MB |
| `checkpoints/n28/ldm_DRC_unified_best_gen.pt` | N28 LDM DRC | 327 MB |
| `checkpoints/n28/ldm_Cong_unified_best_gen.pt` | N28 LDM Congestion | 327 MB |
| `checkpoints/n28/ldm_DRC_control_best_gen.pt` | N28 LDM+ControlNet DRC | 347 MB |
| `checkpoints/n28/ldm_Cong_control_best_gen.pt` | N28 LDM+ControlNet Congestion | 347 MB |
| `checkpoints/n14/vae_DRC_best_ldm.pt` | N14 DRC VAE | 35 MB |
| `checkpoints/n14/vae_Cong_best_ldm.pt` | N14 Congestion VAE | 35 MB |
| `checkpoints/n14/ldm_DRC_unified_best_gen.pt` | N14 LDM DRC | 327 MB |
| `checkpoints/n14/ldm_Cong_unified_best_gen.pt` | N14 LDM Congestion | 327 MB |
| `checkpoints/n14/ldm_DRC_control_best_gen.pt` | N14 LDM+ControlNet DRC | 347 MB |
| `checkpoints/n14/ldm_Cong_control_best_gen.pt` | N14 LDM+ControlNet Congestion | 347 MB |

Download with:
```bash
from huggingface_hub import snapshot_download
snapshot_download(repo_id="kiranthorat/CLDRoute", local_dir="./checkpoints_hf")
```

---

## Reproducing Paper Results

### Step 1 — Train VAEs

```bash
# N28 DRC VAE (~6 h on one A6000)
bash scripts/train/train_drc_vae_n28.sh  /path/to/data/N28  ./runs/vae_DRC_N28

# N28 Congestion VAE (~6 h on one A6000)
bash scripts/train/train_cong_vae_n28.sh /path/to/data/N28  ./runs/vae_Cong_N28
```

### Step 2 — Train LDMs

```bash
# LDM + ControlNet — DRC  (~10 h)
bash scripts/train/train_ldm_drc_n28_control.sh  /path/to/data ./runs/vae_DRC_N28  ./runs/ldm_DRC_N28_control

# LDM + ControlNet — Congestion  (~10 h)
bash scripts/train/train_ldm_cong_n28_control.sh /path/to/data ./runs/vae_Cong_N28 ./runs/ldm_Cong_N28_control

# LDM (no ControlNet) — DRC
bash scripts/train/train_ldm_drc_n28_unified.sh  /path/to/data ./runs/vae_DRC_N28  ./runs/ldm_DRC_N28_unified

# LDM (no ControlNet) — Congestion
bash scripts/train/train_ldm_cong_n28_unified.sh /path/to/data ./runs/vae_Cong_N28 ./runs/ldm_Cong_N28_unified
```

### Step 3 — Evaluate

```bash
# Table 7 & 8 — N28 DRC, LDM+ControlNet
bash scripts/eval/eval_drc_n28_control.sh  ./runs/vae_DRC_N28  ./runs/ldm_DRC_N28_control/best_gen.pt  ./results/drc_n28_ctrl

# Table 9 & 10 — N28 Congestion, LDM+ControlNet
bash scripts/eval/eval_cong_n28_control.sh ./runs/vae_Cong_N28 ./runs/ldm_Cong_N28_control/best_gen.pt ./results/cong_n28_ctrl

# Table 11 — N14 DRC (unified LDM)
bash scripts/eval/eval_drc_n14_unified.sh  ./runs/vae_DRC_N14  ./runs/ldm_DRC_N14_unified/best_gen.pt  ./results/drc_n14_unified

# Table 12 — N14 Congestion (unified LDM)
bash scripts/eval/eval_cong_n14_unified.sh ./runs/vae_Cong_N14 ./runs/ldm_Cong_N14_unified/best_gen.pt ./results/cong_n14_unified
```

### Quick eval with pretrained checkpoints

```bash
# Download checkpoints
python -c "from huggingface_hub import snapshot_download; snapshot_download('kiranthorat/CLDRoute', local_dir='./ckpts')"

# Evaluate DRC N28 (reproduces Table 7 & 8)
bash scripts/eval/eval_drc_n28_control.sh \
    ./ckpts/checkpoints/n28/vae_DRC_best_ldm.pt \
    ./ckpts/checkpoints/n28/ldm_DRC_control_best_gen.pt \
    ./results/drc_n28_pretrained
```

---

## Task-Specific VAE Design

| | DRC VAE | Congestion VAE |
|--|---------|----------------|
| Latent shape | 12 × 64 × 64 | 8 × 64 × 64 |
| Label sparsity | 95.4% zero pixels | 99.8% active pixels |
| Input transform | log(1 + 10x) | none |
| Reconstruction loss | Focal (γ=20) + Hotspot top-1% MSE | L1 |
| KL weight (β) | 0.05 | 0.005 |
| Free-bits (λ) | 0.5 nats | 0.5 nats |
| KL warmup | 15 epochs | 40 epochs |
| Active channels | 12 / 12 — σ ∈ [0.630, 0.642] | 8 / 8 — σ ∈ [0.784, 0.801] |

---

## Citation

If you use CLDRoute in your research, please cite:

```bibtex
@inproceedings{cldroute2026,
  title     = {{CLDRoute}: Conditional Latent Diffusion for Routability Map Generation in Physical Design},
  booktitle = {Proceedings of the IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  year      = {2026}
}
```

---

## Acknowledgements

We use the [CircuitNet 2.0](https://circuitnet.github.io/) dataset.
The diffusion framework builds on ideas from [LDM](https://github.com/CompVis/latent-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet).
