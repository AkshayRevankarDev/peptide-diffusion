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
        epochs=200, seed=seed, lr=3e-4, batch_size=256
    )
    # Argmax eval
    aa_rec, pep_acc = evaluate_aa_recall(
        enc, den, X_te, y_te, m_te,
        batch_size=256, results_dir='results'
    )
    # CFID eval
    aa_cfid, pep_cfid = evaluate_aa_recall(
        enc, den, X_te, y_te, m_te,
        batch_size=256, results_dir='results', use_cfid=True
    )
    results.append((seed, aa_rec, pep_acc, aa_cfid, pep_cfid))
    print(f"Seed {seed} | argmax AA {aa_rec:.2f}% Pep {pep_acc:.2f}%"
          f" | CFID AA {aa_cfid:.2f}% Pep {pep_cfid:.2f}%", flush=True)

print("\n=== ALL SEEDS ===")
for s, aa, pep, aa_c, pep_c in results:
    print(f"  seed {s}: argmax {aa:.2f}/{pep:.2f}  CFID {aa_c:.2f}/{pep_c:.2f}")
aas  = [r[1] for r in results]; peps  = [r[2] for r in results]
aasc = [r[3] for r in results]; pepsc = [r[4] for r in results]
print(f"  Argmax mean: AA {np.mean(aas):.2f}±{np.std(aas):.2f}  Pep {np.mean(peps):.2f}±{np.std(peps):.2f}")
print(f"  CFID   mean: AA {np.mean(aasc):.2f}±{np.std(aasc):.2f}  Pep {np.mean(pepsc):.2f}±{np.std(pepsc):.2f}")
