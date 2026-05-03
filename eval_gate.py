"""
Evaluate entropy-adaptive gate configs (NOVEL #1) against v2 checkpoints.
Appends results to results/novels_ablation_v2.csv.
"""
import sys, os, glob
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from diffusion import build_diffusion_dataset, load_checkpoint, evaluate_aa_recall

BASE       = os.path.join(os.path.dirname(__file__), 'data', 'raw')
mzml_paths = sorted(glob.glob(os.path.join(BASE, 'Ecoli_EV_*.mzML')))
xlsx_paths = sorted(glob.glob(os.path.join(BASE, 'Database search output_Ecoli_EV_*.xlsx')))

if not mzml_paths:
    raise FileNotFoundError(f"No mzML files under {BASE}")

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_paths = sorted(glob.glob('checkpoints/v2/seed_*/diffusion_final.pt'))
if not ckpt_paths:
    ckpt_paths = sorted(glob.glob('checkpoints/v2/seed_*/diffusion_best.pt'))
print(f"Checkpoints: {ckpt_paths}")

configs = [
    dict(label='NOVEL#1 gate',           use_gate=True,  use_beam=False, use_cfid=False, use_rerank=False, use_sgir=False, use_mass_correct=False),
    dict(label='NOVEL#1 gate + CFID',    use_gate=True,  use_beam=False, use_cfid=True,  use_rerank=False, use_sgir=False, use_mass_correct=False),
    dict(label='NOVEL#1 gate+CFID+SGIR', use_gate=True,  use_beam=False, use_cfid=True,  use_rerank=False, use_sgir=True,  use_mass_correct=False),
]

rows = []
for ckpt_path in ckpt_paths:
    seed = os.path.basename(os.path.dirname(ckpt_path)).replace('seed_', '')
    print(f"\n=== seed {seed}: {ckpt_path} ===")
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
        rows.append({'seed': int(seed), 'config': label,
                     'AA Recall %': aa_rec, 'Pep Acc %': pep_acc})
        print(f"    AA Recall: {aa_rec:.2f}%  |  Pep Acc: {pep_acc:.2f}%")

new_df = pd.DataFrame(rows)

out_path = 'results/novels_ablation_v2.csv'
existing = pd.read_csv(out_path)
existing = existing[~existing['config'].str.startswith('NOVEL#1')]
combined = pd.concat([existing, new_df], ignore_index=True)
combined.to_csv(out_path, index=False)

print("\n=== Gate results (3-seed mean) ===")
summary = new_df.groupby('config')[['AA Recall %', 'Pep Acc %']].mean().round(2)
print(summary)

print("\n=== Full ablation summary (3-seed mean) ===")
full_summary = combined.groupby('config')[['AA Recall %', 'Pep Acc %']].mean().round(2)
print(full_summary.sort_values('Pep Acc %', ascending=False))

print(f"\nSaved → {out_path}")
