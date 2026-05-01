"""
Upload V2 model checkpoints and code to HuggingFace Hub.

Usage:
    export HUGGINGFACE_TOKEN=hf_...
    export HF_REPO=your-username/peptide-diffusion-v2
    python upload_to_hf.py
"""
import os
import glob
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    raise ImportError("pip install huggingface_hub")

TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
REPO  = os.environ.get("HF_REPO")

if not TOKEN:
    raise ValueError("Set HUGGINGFACE_TOKEN env var")
if not REPO:
    raise ValueError("Set HF_REPO env var (e.g. yourname/peptide-diffusion-v2)")

api = HfApi(token=TOKEN)

# Create repo if it doesn't exist
create_repo(REPO, repo_type="model", exist_ok=True, private=False, token=TOKEN)
print(f"Repo: https://huggingface.co/{REPO}")

ROOT = Path(__file__).parent

# Upload V2 final checkpoints (seed_0/1/2 diffusion_final.pt)
v2_dir = ROOT / "checkpoints" / "v2"
if not v2_dir.exists():
    # Fall back to seed_* dirs if v2/ not yet created
    v2_dir = ROOT / "checkpoints"

for seed in [0, 1, 2]:
    ckpt = v2_dir / f"seed_{seed}" / "diffusion_final.pt"
    if ckpt.exists():
        dest = f"checkpoints/seed_{seed}/diffusion_final.pt"
        print(f"Uploading {ckpt} → {dest}")
        api.upload_file(path_or_fileobj=str(ckpt), path_in_repo=dest,
                        repo_id=REPO, repo_type="model")
    else:
        print(f"WARNING: {ckpt} not found, skipping")

# Upload metrics CSV
metrics = ROOT / "checkpoints" / "diffusion_metrics.csv"
if metrics.exists():
    api.upload_file(path_or_fileobj=str(metrics),
                    path_in_repo="checkpoints/diffusion_metrics.csv",
                    repo_id=REPO, repo_type="model")
    print("Uploaded diffusion_metrics.csv")

# Upload source code
for f in ["src/diffusion.py", "retrain_all.py", "eval_novels.py",
          "requirements.txt"]:
    p = ROOT / f
    if p.exists():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f,
                        repo_id=REPO, repo_type="model")
        print(f"Uploaded {f}")

# Upload results
for csv in (ROOT / "results").glob("*.csv"):
    api.upload_file(path_or_fileobj=str(csv),
                    path_in_repo=f"results/{csv.name}",
                    repo_id=REPO, repo_type="model")
    print(f"Uploaded results/{csv.name}")

print("\nDone. Model card: https://huggingface.co/" + REPO)
