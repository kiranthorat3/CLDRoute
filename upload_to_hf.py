#!/usr/bin/env python3
"""Upload CLDRoute checkpoints to HuggingFace Hub."""
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "kiranthorat/CLDRoute"
BASE = "/data2/kgt22001/cong_gen"

checkpoints = {
    # N28
    "checkpoints/n28/vae_DRC_best_ldm.pt":
        f"{BASE}/gen_additional/drc_vae/runs/vae_DRC_expanded/best_ldm.pt",
    "checkpoints/n28/vae_Cong_best_ldm.pt":
        f"{BASE}/gen_additional/conegestion_vae/runs/vae_Cong_expanded_v2/best_ldm.pt",
    "checkpoints/n28/ldm_DRC_unified_best_gen.pt":
        f"{BASE}/gen_additional/ldm_unified/runs/ldm_DRC_expanded/best_gen.pt",
    "checkpoints/n28/ldm_Cong_unified_best_gen.pt":
        f"{BASE}/gen_additional/ldm_unified/runs/ldm_Cong_unified_v2/best_gen.pt",
    "checkpoints/n28/ldm_DRC_control_best_gen.pt":
        f"{BASE}/gen_additional/ldm_control/runs/ldm_DRC_control/best_gen.pt",
    "checkpoints/n28/ldm_Cong_control_best_gen.pt":
        f"{BASE}/gen_additional/ldm_control/runs/ldm_Cong_control_v2/best_gen.pt",
    # N14
    "checkpoints/n14/vae_DRC_best_ldm.pt":
        f"{BASE}/n14_gen/drc_vae/runs/vae_DRC_N14_splitB_v2_beta001_w40/best_ldm.pt",
    "checkpoints/n14/vae_Cong_best_ldm.pt":
        f"{BASE}/n14_gen/cong_vae/runs/vae_Cong_N14_splitB_v1/best_ldm.pt",
    "checkpoints/n14/ldm_DRC_unified_best_gen.pt":
        f"{BASE}/n14_gen/ldm_unified/runs/ldm_DRC_N14_splitB_v2_unified/best_gen.pt",
    "checkpoints/n14/ldm_Cong_unified_best_gen.pt":
        f"{BASE}/n14_gen/ldm_unified/runs/ldm_Cong_N14_unified_v1/best_gen.pt",
    "checkpoints/n14/ldm_DRC_control_best_gen.pt":
        f"{BASE}/n14_gen/ldm_controlnet/runs/ldm_DRC_N14_splitB_v2_controlnet/best_gen.pt",
    "checkpoints/n14/ldm_Cong_control_best_gen.pt":
        f"{BASE}/n14_gen/ldm_controlnet/runs/ldm_Cong_N14_control_v1/best_gen.pt",
}

api = HfApi()

print(f"Creating repo {REPO_ID} ...")
create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)

for hf_path, local_path in checkpoints.items():
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {local_path}")
        continue
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  Uploading {hf_path}  ({size_mb:.0f} MB) ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hf_path,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  Done: {hf_path}")

print("\nAll checkpoints uploaded to https://huggingface.co/kiranthorat/CLDRoute")
