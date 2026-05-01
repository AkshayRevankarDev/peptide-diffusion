"""
Quick eval: re-score existing seed checkpoints with the new iterative
reverse diffusion inference. No retraining needed.
"""
import sys, os, glob, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from diffusion import (
    build_diffusion_dataset, load_checkpoint,
    evaluate_aa_recall, decode_tokens, aa_recall,
)

BASE  = os.path.join(os.path.dirname(__file__), 'data', 'raw')
mzml_paths = sorted(glob.glob(os.path.join(BASE, 'Ecoli_EV_*.mzML')))
xlsx_paths = sorted(glob.glob(os.path.join(BASE, 'Database search output_Ecoli_EV_*.xlsx')))

if not mzml_paths:
    raise FileNotFoundError(f"No mzML files found under {BASE}")

print(f"Data: {[os.path.basename(p) for p in mzml_paths]}")

# Build full dataset (same split logic as training)
from diffusion import build_diffusion_dataset
import numpy as np

Xs, ys, ms = [], [], []
for mzml, xlsx in zip(mzml_paths, xlsx_paths):
    X, y, m = build_diffusion_dataset(mzml, xlsx, max_spectra=5000)
    Xs.append(X); ys.append(y); ms.append(m)
X = np.concatenate(Xs); y = np.concatenate(ys); masses = np.concatenate(ms)

rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
n_tr = int(0.70 * len(X)); n_va = int(0.15 * len(X))
te_idx = idx[n_tr + n_va:]
X_te, y_te, m_te = X[te_idx], y[te_idx], masses[te_idx]
print(f"Test spectra: {len(X_te)}")

device = torch.device('cpu')
seed_dirs = sorted(glob.glob('checkpoints/seed_*/diffusion_best.pt'))
if not seed_dirs:
    seed_dirs = ['checkpoints/diffusion_best.pt']

results = []
for ckpt_path in seed_dirs:
    print(f"\n=== {ckpt_path} ===")
    encoder, denoiser = load_checkpoint(ckpt_path, device=device)
    aa_rec, pep_acc = evaluate_aa_recall(
        encoder, denoiser, X_te, y_te, m_te,
        batch_size=64, results_dir='results', device=device,
        use_gate=False
    )
    seed = os.path.basename(os.path.dirname(ckpt_path)).replace('seed_', '')
    results.append({'seed': seed, 'AA Recall %': aa_rec, 'Pep Acc %': pep_acc})
    print(f"  AA Recall: {aa_rec:.2f}%  |  Pep Acc: {pep_acc:.2f}%")

df = pd.DataFrame(results)
print("\n=== Summary ===")
print(df.to_string(index=False))
print(f"\nMean AA Recall : {df['AA Recall %'].mean():.2f} ± {df['AA Recall %'].std():.2f}")
print(f"Mean Pep Acc   : {df['Pep Acc %'].mean():.2f} ± {df['Pep Acc %'].std():.2f}")

df.to_csv('results/iterative_eval_results.csv', index=False)
print("\nSaved → results/iterative_eval_results.csv")
