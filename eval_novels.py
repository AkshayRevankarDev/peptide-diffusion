"""
Ablation evaluation: baseline vs NOVEL #8 (beam) vs NOVEL #9 (SGIR) vs NOVEL #10 (rerank).
Loads the best checkpoint from each seed and averages results.
Writes results/novels_ablation.csv
"""
import sys, os, glob
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from diffusion import (
    build_diffusion_dataset, load_checkpoint, evaluate_aa_recall,
)

BASE       = os.path.join(os.path.dirname(__file__), 'data', 'raw')
mzml_paths = sorted(glob.glob(os.path.join(BASE, 'Ecoli_EV_*.mzML')))
xlsx_paths = sorted(glob.glob(os.path.join(BASE, 'Database search output_Ecoli_EV_*.xlsx')))

if not mzml_paths:
    raise FileNotFoundError(f"No mzML files under {BASE}")

print(f"Loading data from {[os.path.basename(p) for p in mzml_paths]}")

Xs, ys, ms, rps = [], [], [], []
for mzml, xlsx in zip(mzml_paths, xlsx_paths):
    X, y, m, rp = build_diffusion_dataset(mzml, xlsx, max_spectra=5000, return_raw=True)
    Xs.append(X); ys.append(y); ms.append(m); rps.extend(rp)

X = np.concatenate(Xs)
y = np.concatenate(ys)
masses = np.concatenate(ms)

rng    = np.random.default_rng(42)
idx    = rng.permutation(len(X))
n_tr   = int(0.70 * len(X)); n_va = int(0.15 * len(X))
te_idx = idx[n_tr + n_va:]
X_te   = X[te_idx]; y_te = y[te_idx]; m_te = masses[te_idx]
rp_te  = [rps[i] for i in te_idx]
print(f"Test spectra: {len(X_te)}")

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Prefer diffusion_final.pt (last epoch, best AA recall) over diffusion_best.pt
# (best val CE loss, which correlates poorly with positional AA recall).
ckpt_paths = sorted(glob.glob('checkpoints/seed_*/diffusion_final.pt'))
if not ckpt_paths:
    ckpt_paths = sorted(glob.glob('checkpoints/seed_*/diffusion_best.pt'))
if not ckpt_paths:
    ckpt_paths = ['checkpoints/diffusion_best.pt']

# use_esm=False throughout: ESM-2 pseudo-PPL is O(L) forward passes per sequence,
# i.e. hours on CPU. Spectral reranking alone is fast and already meaningful.
configs = [
    dict(label='Baseline (argmax)',          use_beam=False, use_cfid=False, use_rerank=False, use_sgir=False, use_mass_correct=False),
    dict(label='Option B mass-correct',      use_beam=False, use_cfid=False, use_rerank=False, use_sgir=False, use_mass_correct=True),
    dict(label='NOVEL#11 CFID',              use_beam=False, use_cfid=True,  use_rerank=False, use_sgir=False, use_mass_correct=False),
    dict(label='NOVEL#11 CFID + mass-corr',  use_beam=False, use_cfid=True,  use_rerank=False, use_sgir=False, use_mass_correct=True),
    dict(label='NOVEL#11 CFID + SGIR',       use_beam=False, use_cfid=True,  use_rerank=False, use_sgir=True,  use_mass_correct=False),
    dict(label='NOVEL#9 SGIR',               use_beam=False, use_cfid=False, use_rerank=False, use_sgir=True,  use_mass_correct=False),
    dict(label='NOVEL#10 rerank-spectral',   use_beam=False, use_cfid=False, use_rerank=True,  use_sgir=False,
         n_rerank=10, T_sample=0.8, use_esm=False, use_mass_correct=False),
    # NOVEL #8: mass-constrained beam search — constrains mass DURING decoding (correct fix)
    dict(label='NOVEL#8 beam',               use_beam=True,  use_cfid=False, use_rerank=False, use_sgir=False, use_mass_correct=False),
    dict(label='NOVEL#8 beam + SGIR',        use_beam=True,  use_cfid=False, use_rerank=False, use_sgir=True,  use_mass_correct=False),
]

rows = []
for ckpt_path in ckpt_paths:
    seed = os.path.basename(os.path.dirname(ckpt_path)).replace('seed_', '')
    print(f"\n=== {ckpt_path} ===")
    encoder, denoiser = load_checkpoint(ckpt_path, device=device)

    for cfg in configs:
        label  = cfg['label']
        kwargs = {k: v for k, v in cfg.items() if k != 'label'}
        print(f"  [{label}]")
        aa_rec, pep_acc = evaluate_aa_recall(
            encoder, denoiser, X_te, y_te, m_te,
            batch_size=256, results_dir='results', device=device,
            raw_peaks=rp_te if kwargs.get('use_sgir') else None,
            **kwargs
        )
        rows.append({'seed': seed, 'config': label,
                     'AA Recall %': aa_rec, 'Pep Acc %': pep_acc})
        print(f"    AA Recall: {aa_rec:.2f}%  |  Pep Acc: {pep_acc:.2f}%")

df = pd.DataFrame(rows)
print("\n=== Summary ===")
print(df.groupby('config')[['AA Recall %', 'Pep Acc %']].mean().to_string())

os.makedirs('results', exist_ok=True)
df.to_csv('results/novels_ablation.csv', index=False)
print("\nSaved → results/novels_ablation.csv")
