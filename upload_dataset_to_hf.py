#!/usr/bin/env python3
"""
Upload CLDRoute physics-aware routing control features to Hugging Face Datasets.

Uses upload_large_folder() — batches thousands of files into few commits,
avoids the 128-commits/hour rate limit.

What is uploaded:
  N28  DRC features:        256×256×16 float32  ~41 GB
  N28  DRC labels:          256×256×1  float32  ~2.6 GB
  N28  Congestion features: 256×256×11 float32  ~28 GB
  N28  Congestion labels:   256×256×1  float32  ~2.6 GB
  N14  DRC features:        256×256×16 float32  ~43 GB
  N14  DRC labels:          256×256×1  float32  ~2.7 GB
  N14  Congestion features: 256×256×11 float32  ~30 GB
  N14  Congestion labels:   256×256×1  float32  ~2.7 GB
  CSV splits for both nodes

Total: ~152 GB  |  Expected time: several hours

Run with:
    conda activate eda_vision
    python upload_dataset_to_hf.py
"""
import os, shutil, tempfile
from huggingface_hub import HfApi, create_repo

REPO_ID  = "kiranthorat/CLDRoute-dataset"
N28_ROOT = "/data2/kgt22001/CircuitNet-N28/training_set_expanded"
N14_ROOT = "/data2/kgt22001/CircuitNet-N14/training_set_expanded"
SPLITS   = "/data2/kgt22001/CLDRoute/data/splits"

api = HfApi()

print(f"Creating dataset repo: {REPO_ID}")
create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)

# Build a staging directory that mirrors the repo layout, then upload in one call.
# upload_large_folder() handles chunking, retries, and minimises commit count.
staging = tempfile.mkdtemp(prefix="cldroute_dataset_")
print(f"Staging dir: {staging}")

def stage(src, rel_dst):
    dst = os.path.join(staging, rel_dst)
    os.makedirs(dst, exist_ok=True)
    print(f"  Symlinking {src}  →  {rel_dst}/")
    # Use symlinks so we don't duplicate 152 GB on disk
    for fname in os.listdir(src):
        src_f = os.path.join(src, fname)
        dst_f = os.path.join(dst, fname)
        if os.path.isfile(src_f) and not os.path.exists(dst_f):
            os.symlink(src_f, dst_f)

print("\nStaging data directories (symlinks, no copy)...")
stage(f"{N28_ROOT}/DRC/feature",        "n28/DRC/feature")
stage(f"{N28_ROOT}/DRC/label",          "n28/DRC/label")
stage(f"{N28_ROOT}/congestion/feature", "n28/congestion/feature")
stage(f"{N28_ROOT}/congestion/label",   "n28/congestion/label")
stage(f"{N14_ROOT}/DRC/feature",        "n14/DRC/feature")
stage(f"{N14_ROOT}/DRC/label",          "n14/DRC/label")
stage(f"{N14_ROOT}/congestion/feature", "n14/congestion/feature")
stage(f"{N14_ROOT}/congestion/label",   "n14/congestion/label")
stage(SPLITS,                           "splits")

print(f"\nUploading ~152 GB to {REPO_ID} ...")
print("(This will take several hours — progress logged below)\n")

api.upload_large_folder(
    folder_path=staging,
    repo_id=REPO_ID,
    repo_type="dataset",
    private=False,
)

print(f"\nDone → https://huggingface.co/datasets/{REPO_ID}")

# Cleanup symlink staging dir (does not delete actual data)
shutil.rmtree(staging, ignore_errors=True)
