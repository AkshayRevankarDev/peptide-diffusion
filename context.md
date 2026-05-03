# Peptide-Diffusion: Project Context

**Team:** Vaishak Girish Kumar, Akshay Mohan Revankar, Sanika Vilas Nanjan (UB CSE 676 Deep Learning final project)

## Overview

De novo peptide sequencing using **absorbing multinomial diffusion** trained on E. coli EV proteomics (MS/MS spectra). The model performs no database lookup—given a spectrum, it generates the peptide sequence directly. This enables discovery in novel organisms and mixed communities where reference proteomes don't exist.

## Architecture (V2 Final)

**Components:**
- **PeakEncoder**: Learns m/z spectrum → hidden representation (critical component; removes it drops AA accuracy to 41.88%)
- **B/Y-ion pair bias**: Learnable scalar s (initialized at 0, AlphaFold3-inspired design) predicts likelihood of matched b/y ion pairs
- **Diffusion backbone**: Absorbing multinomial diffusion with 20 reverse timesteps
- **Self-conditioning**: Conditions diffusion on previous predicted sequence (adds +14.6 pp to peptide accuracy—biggest single gain)
- **Cosine noise schedule**: Replaces linear; provides better empirical performance
- **Attention mask fix**: Prevents cross-spectrum contamination during batch processing

**Inference reranking:**
- CFID (Consistency-Folding by ID): Uses ESM-2 folding consistency
- SGIR (Sequence Generation Importance Ranking): Structural validity check
- Best config: CFID+SGIR achieves 76.00% AA / 59.96% peptide accuracy at 5% FDR

## Results (3-seed mean)

| Model | AA Accuracy | Peptide Accuracy | Notes |
|-------|-------------|------------------|-------|
| LSTM baseline | 31.51% | 2.68% | TransformerCoder |
| GRU baseline | 44.30% | 6.70% | TransformerCoder |
| Absorbing diffusion (no PeakEnc) | 41.88% | — | PeakEncoder is critical |
| V1 (PeakEnc+BY+CFID) | 75.19% | 44.99% | Prior version |
| V2 argmax | 76.30% | 57.49% | Direct sampling |
| **V2 CFID+SGIR** | **76.00% ± 0.19** | **59.96% ± 3.82** | **Best final** |
| InstaNovo (published SOTA) | 72.9% | 33.1% | Baseline comparison |

**Beat published SOTA by +3.1 pp AA, +26.9 pp peptide accuracy.**

## Datasets

**E. coli EV (training + eval):**
- ~3600 MS/MS spectra from E. coli extracellular vesicles
- Ground truth from database search (xlsx files)
- Train/val/test split at 70/15/15

**Wastewater (held-out):**
- Sample 2: 6 peptides passing 5% FDR with no reference proteome
- ESM-2 per-residue PPL: 31.5–41.1 (z-scores < 1.0 vs E. coli GT; uncertain validity)
- BLAST validation pending

## Key Files

- **src/train_diffusion.py** — V2 training loop with self-conditioning and cosine schedule
- **src/eval_novels.py** — Ablation across 7 inference strategies (argmax, CFID, SGIR, etc.)
- **src/esm_scoring.py** — ESM-2 per-residue PPL scorer for predicted sequences
- **results/novels_ablation_v2.csv** — Full 3-seed × 7-config ablation results
- **results/wastewater_predictions_5pct_fdr.csv** — Wastewater peptides + ESM-2 scores
- **report/report.pdf** — NeurIPS-style final paper
- **checkpoints/v2/seed_{0,1,2}/** — Trained V2 models (all 3 seeds)
- **figures/figure*.png** — Presentation figures (white background)

## How to Run Inference

```bash
# Load a trained checkpoint
from src.train_diffusion import load_checkpoint
model = load_checkpoint('checkpoints/v2/seed_0/best.pt')

# Predict on new spectra (must match training preprocessing)
predictions = model.predict(spectra_batch, num_samples=4)  # num_samples for CFID reranking

# Apply CFID+SGIR reranking
from src.eval_novels import rerank_cfid_sgir
best = rerank_cfid_sgir(predictions, esm_model)
```

## Known Issues & Decisions

- **Mass constraint at inference — fully explored, all approaches failed:** Three strategies were evaluated. (1) Post-hoc single-swap: −9.6 pp pep. (2) Mass-constrained beam search (NOVEL #8): −44 pp pep — left-to-right decoding is architecturally incompatible with bidirectional diffusion. (3) Entropy-adaptive gate (NOVEL #1) alone: −48 pp pep; gate + CFID/SGIR: statistically identical to CFID/SGIR without gate. CFID's iterative mask-predict refinement subsumes all mass-aware improvements. This is a resolved negative result, not a TODO.
- **InstaNovo comparison is cross-dataset:** The 72.9%/33.1% InstaNovo numbers are from their published paper on their benchmark, not re-run on our E. coli EV test set. The comparison is directional — our model is evaluated on our test set; InstaNovo's published numbers are on theirs. Valid to report but not a direct apples-to-apples benchmark.
- **Attention mask fix:** Early versions allowed cross-spectrum contamination in batches. Fixed by masking off-diagonal attention during training.
- **Self-conditioning critical:** Removes ~14.6 pp if disabled. Was initially optional but proved essential.
- **PeakEncoder architecture:** Initial attempts without it yielded 41.88% AA. Now non-negotiable for results.

## Presentation Status

All figures complete (white background). Report in LaTeX + PDF. Code clean and reproducible with 3-seed validation. Ready for defense.
