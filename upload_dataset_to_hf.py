#!/usr/bin/env python3
"""
Upload CLDRoute physics-aware routing control features to Hugging Face Datasets.

What is uploaded (features + labels + splits):
  N28  DRC features:        256×256×16 float32  ~41 GB  (10,242 files)
  N28  DRC labels:          256×256×1  float32  ~2.6 GB
  N28  Congestion features: 256×256×11 float32  ~28 GB
  N28  Congestion labels:   256×256×1  float32  ~2.6 GB
  N14  DRC features:        256×256×16 float32  ~43 GB
  N14  DRC labels:          256×256×1  float32  ~2.7 GB
  N14  Congestion features: 256×256×11 float32  ~30 GB
  N14  Congestion labels:   256×256×1  float32  ~2.7 GB
  CSV splits (train/val/test) for both nodes

Total: ~152 GB  |  Expected upload time: several hours

Run with:
    conda activate eda_vision
    python upload_dataset_to_hf.py
"""
import os, glob
from huggingface_hub import HfApi, create_repo

REPO_ID   = "kiranthorat/CLDRoute-dataset"
N28_ROOT  = "/data2/kgt22001/CircuitNet-N28/training_set_expanded"
N14_ROOT  = "/data2/kgt22001/CircuitNet-N14/training_set_expanded"
SPLIT_DIR = "/data2/kgt22001/CLDRoute/data/splits"

api = HfApi()

print(f"Creating dataset repo: {REPO_ID}")
create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)

def upload_dir(local_dir, repo_subdir, desc):
    files = sorted(glob.glob(os.path.join(local_dir, "*.npy")))
    n = len(files)
    total_gb = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"\n[{desc}]  {n} files  ({total_gb:.1f} GB)  →  {repo_subdir}/")
    for i, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        repo_path = f"{repo_subdir}/{fname}"
        if (i - 1) % 500 == 0:
            mb = os.path.getsize(fpath) / 1e6
            print(f"  {i}/{n}  {fname}  ({mb:.1f} MB)")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
    print(f"  Done: {repo_subdir}/")

# ── N28 ──────────────────────────────────────────────────────────────────────
upload_dir(f"{N28_ROOT}/DRC/feature",          "n28/DRC/feature",          "N28 DRC features (16ch)")
upload_dir(f"{N28_ROOT}/DRC/label",            "n28/DRC/label",            "N28 DRC labels")
upload_dir(f"{N28_ROOT}/congestion/feature",   "n28/congestion/feature",   "N28 Cong features (11ch)")
upload_dir(f"{N28_ROOT}/congestion/label",     "n28/congestion/label",     "N28 Cong labels")

# ── N14 ──────────────────────────────────────────────────────────────────────
upload_dir(f"{N14_ROOT}/DRC/feature",          "n14/DRC/feature",          "N14 DRC features (16ch)")
upload_dir(f"{N14_ROOT}/DRC/label",            "n14/DRC/label",            "N14 DRC labels")
upload_dir(f"{N14_ROOT}/congestion/feature",   "n14/congestion/feature",   "N14 Cong features (11ch)")
upload_dir(f"{N14_ROOT}/congestion/label",     "n14/congestion/label",     "N14 Cong labels")

# ── CSV splits ───────────────────────────────────────────────────────────────
print("\n[Splits]")
for csv in sorted(glob.glob(os.path.join(SPLIT_DIR, "*.csv"))):
    fname = os.path.basename(csv)
    api.upload_file(
        path_or_fileobj=csv,
        path_in_repo=f"splits/{fname}",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"  splits/{fname}")

print(f"\nAll done → https://huggingface.co/datasets/{REPO_ID}")
