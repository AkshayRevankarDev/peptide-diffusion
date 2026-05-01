# De Novo Peptide Sequencing via Multinomial Diffusion

**CSE 676 Deep Learning — Group Project (CP2)**  
**Team:** CSE 676 Deep Learning Group Project  
**GitHub:** https://github.com/AkshayRevankarDev/peptide-diffusion

---

## Results

| Model | AA Recall | Peptide Accuracy |
|---|---|---|
| LSTM Baseline | 31.51% | — |
| GRU Baseline | 44.30% | — |
| Diffusion V1 (CFID) | 75.19% | 44.99% |
| **Diffusion V2 (CFID+SGIR)** | **76.00%** | **59.96%** |
| InstaNovo (published 2025 SOTA) | 72.90% | 33.10% |

Our V2 model beats InstaNovo by **+3.1 pp AA recall** and **+26.9 pp peptide accuracy** on the E. coli EV benchmark (3-seed mean, 5% FDR).

---

## Architecture

```
MS/MS Spectrum (m/z, intensity pairs)
        │
        ▼
  PeakEncoder  ← learnable b/y-ion pair bias (physics injection)
        │
        ▼
  Transformer Backbone (absorbing diffusion, T=500 steps)
        │
        ▼
  Self-Conditioning  ← feeds previous denoised estimate back at each step
        │
        ▼
  Candidate Sequences
        │
        ▼
  CFID + SGIR reranking  ← Cosine-scheduled CFID + Spectral-Guided Ion Rescoring
        │
        ▼
  Final Peptide Sequences (CSV, 5% FDR)
```

**Key design choices:**

- **PeakEncoder with B/Y-ion bias** — an AlphaFold3-inspired learnable scalar `s` (initialized to 0) that up-weights fragment ion pairs. The model can learn to ignore this if unhelpful.
- **Self-conditioning** — at each diffusion step, the previous denoised estimate is fed back as an additional input. This is the single largest gain: +14.6 pp peptide accuracy over V1.
- **Cosine noise schedule** — replaces linear schedule; better preserves signal at low noise levels.
- **Attention mask fix** — correctly masks padding tokens, preventing cross-contamination across spectra.
- **CFID + SGIR** — Combined Fragment Ion Distribution reranking with Spectral-Guided Ion Rescoring selects the best candidate from beam search.

---

## Data

- **E. coli EV proteomics** (`Data/E coli EV proteomics/`) — mzML files + database search xlsx ground truth. Used for training and benchmarking.
- **Wastewater proteomics** (`Data/Wastewater/`) — mzML files from a mixed-community sample. No reference proteome. 6 peptides passed 5% FDR filtering (Sample 2).

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
python src/train_diffusion.py \
    --data results/diffusion_predictions.csv \
    --seed 0
```

### Evaluate (novel inference strategies)

```bash
python src/eval_novels.py \
    --checkpoint checkpoints/v2/seed_0.pt \
    --out results/novels_ablation_v2.csv
```

### Score wastewater sequences with ESM-2

```bash
python src/esm_scoring.py \
    --diffusion_csv results/diffusion_predictions.csv \
    --out_csv results/esm2_scores.csv
```

---

## Key Files

| Path | Description |
|---|---|
| `src/train_diffusion.py` | V2 training loop (absorbing diffusion + self-conditioning) |
| `src/eval_novels.py` | Evaluates CFID, SGIR, mass-correction ablations |
| `src/esm_scoring.py` | ESM-2 per-residue pseudo-perplexity scorer |
| `src/preprocessing.py` | mzML → tensor pipeline |
| `results/novels_ablation_v2.csv` | Full ablation table (3 seeds × 7 configs) |
| `results/wastewater_predictions_5pct_fdr.csv` | Wastewater FDR-filtered peptides with ESM-2 PPL |
| `figures/figure5_progression.png` | Model progression chart (LSTM → V2 vs. InstaNovo) |
| `checkpoints/v2/` | V2 model checkpoints (3 seeds) |

---

## Wastewater Findings

6 peptides from Sample 2 survived 5% FDR filtering without a reference proteome:

| Sequence | ESM-2 PPL | Anomalous Frac |
|---|---|---|
| FNDVIPMGEQAINTNEGAYR | 41.13 | 0.85 |
| NNGNAIGVDLAAIPFVAGDR | 31.51 | 0.85 |
| GSNYNEVVTLADVTIVQGIR | 36.63 | 0.90 |
| DLDVEFTALDGASVQVIAYR | 38.35 | 0.95 |
| ALDNAIDGGQYSFLEVAINR | 37.87 | 0.90 |
| QLDNNCVYLGATAGVPIAK  | 37.42 | 0.84 |

High anomalous fraction indicates these are structurally novel vs. the E. coli EV reference distribution — consistent with mixed-community microbial origin.

---

## Ablation Summary (3-seed mean)

| Config | AA Recall | Pep Accuracy |
|---|---|---|
| V2 argmax (baseline) | 76.21% | 57.84% |
| V2 CFID | 75.93% | 59.60% |
| V2 CFID + mass-correct | 75.21% | 50.44% |
| **V2 CFID + SGIR** | **75.99%** | **59.96%** |
| V2 SGIR only | 76.17% | 57.61% |
| V2 rerank-spectral | 73.73% | 53.67% |

Mass correction is a negative result: filtering is too aggressive at this model's confidence level, hurting peptide accuracy by ~9 pp.
