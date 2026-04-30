import sys, os, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from diffusion import train_diffusion, evaluate_aa_recall

BASE = os.path.join(os.path.dirname(__file__), 'data', 'raw')
mzml = sorted(glob.glob(os.path.join(BASE, 'Ecoli_EV_*.mzML')))
xlsx = sorted(glob.glob(os.path.join(BASE, 'Database search output_Ecoli_EV_*.xlsx')))

results = []
for seed in [0, 1, 2]:
    print(f"\n{'='*40}\nSeed {seed}\n{'='*40}", flush=True)
    enc, den, (X_te, y_te, m_te) = train_diffusion(
        mzml, xlsx,
        checkpoint_dir=os.path.join('checkpoints', f'seed_{seed}'),
        epochs=50, seed=seed, lr=1e-3
    )
    aa_rec, pep_acc = evaluate_aa_recall(
        enc, den, X_te, y_te, m_te,
        batch_size=64, results_dir='results'
    )
    results.append((seed, aa_rec, pep_acc))
    print(f"Seed {seed} FINAL | AA: {aa_rec:.2f}% | Pep: {pep_acc:.2f}%", flush=True)

print("\n=== ALL SEEDS ===")
aas  = [r[1] for r in results]
peps = [r[2] for r in results]
for s, a, p in results:
    print(f"  seed {s}: AA {a:.2f}%  Pep {p:.2f}%")
print(f"  Mean: AA {np.mean(aas):.2f} +/- {np.std(aas):.2f}%  |  Pep {np.mean(peps):.2f} +/- {np.std(peps):.2f}%")
