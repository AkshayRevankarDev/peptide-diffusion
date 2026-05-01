# Development Notes

## Key Decisions Made

### 1. Absorbing Multinomial Diffusion (vs. Continuous Gaussian)
**Decision:** Use discrete diffusion model with absorbing state for peptide tokens.
**Rationale:** Sequences are inherently discrete. Continuous diffusion would require argmax quantization at inference, introducing bias. Absorbing diffusion directly models token corruption and recovery.
**Tradeoff:** Slower to train (20 steps) but semantically sound; avoids quantization error.

### 2. Self-Conditioning Architecture
**Decision:** Condition reverse diffusion on the model's previous predicted sequence.
**Rationale:** Discovered +14.6 pp gain in peptide accuracy—largest single improvement across all ablations.
**Implementation:** At inference, sample → use sample as conditioning signal → re-sample → rerank.
**Critical:** This is now non-negotiable for competitive results.

### 3. PeakEncoder + B/Y-Ion Pair Bias
**Decision:** Learn spectrum → representation mapping (PeakEncoder) + pair likelihood (scalar s).
**Rationale:** Removes PeakEncoder drops to 41.88% AA. The encoder learns to suppress noise and highlight matched ions.
**B/Y bias design:** Learnable scalar s (init=0, similar to AlphaFold3 pairwise attention biases).
**Result:** Enables 76%+ accuracy on E. coli EV data.

### 4. Cosine Noise Schedule
**Decision:** Replace linear schedule with cosine (β_t = 0.5 * (1 - cos(πt/T))).
**Rationale:** Empirically outperforms linear in early ablations. Cosine keeps signal longer in the beginning, better matching natural noise decay.
**No strong theory:** Schedule choice is dataset/task-dependent; no ablation paper justified it analytically.

### 5. Attention Mask Fix (Critical Bug)
**Decision:** Add causal/off-diagonal masking during batch training to prevent cross-spectrum contamination.
**Rationale:** Transformer attention was computing relationships between spectra in a batch. During val/test on new spectra, this signal vanishes → train/test mismatch.
**Impact:** High (affects generalization to held-out data).
**Resolution:** Mask off-diagonal attention in batch dimension; self-attention within each spectrum only.

### 6. No Mass-Conditioned Loss (Negative Result)
**Decision:** Attempted post-hoc 50 ppm filtering but abandoned it.
**Why it failed:** Hurts peptide accuracy by ~9 pp (59.96% → ~51%).
**Root cause:** Model was not trained with mass constraint. At inference, enforcing mass filter contradicts learned prior.
**Correct fix (TODO):** Incorporate mass penalty into training loss or use diffusion guidance.
**Lesson:** Inference constraints must align with training objectives.

### 7. CFID + SGIR Reranking
**Decision:** Use dual reranking: ESM-2 folding consistency + structural validity.
**Rationale:** CFID alone plateaued at ~73% pep. SGIR adds orthogonal signal (sequence viability).
**Best config:** CFID+SGIR achieves 76.00% AA / 59.96% pep (vs. argmax 76.30% AA / 57.49% pep).
**Trade:** Introduces inference latency (ESM-2 forward passes) but improves reproducibility of long sequences.

## Gotchas & Bugs Encountered

### Gotcha 1: Batch Attention Contamination
**Symptom:** Val loss good, but held-out test accuracy much lower.
**Root cause:** Multi-spectrum batches allowed cross-attention. Model learned spurious correlations during training.
**Fix:** Add mask tensor to transformer layers; set batch-off-diagonal to -inf.
**Detection time:** ~2 weeks into V2 training. Wasted iterations.

### Gotcha 2: Self-Conditioning Inference Loop
**Symptom:** Marginal gain from self-conditioning during dev; huge gain in final eval.
**Root cause:** Early experiments ran self-conditioning on only a few samples. Final ablation used all 3 seeds × multiple samples.
**Lesson:** Always validate improvements with full statistical power before committing to architecture.

### Gotcha 3: ESM-2 PPL Scaling
**Symptom:** Wastewater peptide PPL (31–41) seemed reasonable until comparison with E. coli GT distribution.
**Discovery:** E. coli GT PPL is ~10–20. Wastewater peptides are outliers (z < -1.0 in many cases).
**Interpretation:** Wastewater predictions may be false positives; ESM-2 alignment to E. coli training data is problematic.
**Action:** Flag results as "uncertain validity" pending BLAST validation.

### Gotcha 4: Cosine Schedule Instability at High Steps
**Symptom:** At T=50 steps, cosine schedule had near-zero gradients at early timesteps.
**Why it happened:** cos(πt/T) approaches 1 for small t, making β_t ≈ 0.
**Resolution:** Use T=20 (sweet spot) or add small epsilon to prevent collapse.
**Current setting:** T=20 timesteps, cosine schedule, stable training.

## Things That Didn't Work

### 1. Linear Noise Schedule
Tried first, but cosine empirically better. No ablation paper backs this up—likely dataset/task-dependent.

### 2. Separate Encoder for Mass Features
Added mass, charge, precursor m/z as separate inputs. Model ignored them. Removed in favor of implicit signal from spectrum.

### 3. Post-hoc Mass Filtering (50 ppm)
Hurts peptide accuracy by ~9 pp. Needs mass-conditioned training loss instead.

### 4. Dropout in PeakEncoder
Added to prevent overfitting but reduced val accuracy. Removed; batch norm sufficient.

### 5. Variable-Length Diffusion
Tried absorbing non-sequence tokens dynamically. Complexity not worth marginal gains. Fixed max length (30) works.

## Future TODOs

### High Priority
1. **Mass-conditioned training loss:** Incorporate m/z constraint into diffusion objective to enable safe post-hoc filtering.
2. **Wastewater BLAST validation:** Confirm 6 predicted peptides against NCBI nr. Currently flagged as uncertain.
3. **Inference latency:** Profile CFID+SGIR bottleneck; consider batched ESM-2 forward pass optimization.

### Medium Priority
4. **Checkpoint model compression:** Current checkpoints ~500 MB. Quantization or knowledge distillation to smaller model for deployment.
5. **Ablation on schedule hyperparameters:** Validate T=20 choice. Try T=10 vs T=50 systematically.
6. **Cross-dataset generalization:** Train on E. coli, evaluate on held-out wastewater. Reverse test. Report transfer metrics.

### Low Priority
7. **Fine-tune on noisy spectra:** Current data is high-quality. Synthetic noise injection for robustness.
8. **Ensemble predictions:** Combine 3-seed predictions via voting. Likely marginal gains.
9. **Visualization of attention patterns:** Debug what PeakEncoder learns about ion pairs.

## Statistics & Reproducibility

- **3-seed ablation:** Seeds 0, 1, 2 all run to completion. Results in `results/novels_ablation_v2.csv`.
- **Standard deviations:** Peptide accuracy ±3.82 pp; AA accuracy ±0.19 pp. Relatively stable across seeds.
- **Baseline comparisons:** LSTM, GRU, InstaNovo all from published sources or direct inference.
- **FDR filtering:** All results at 5% FDR threshold (industry standard). Pre-FDR results also logged.

## Code Quality

- All source code follows PEP 8 and type hints.
- Training/eval scripts are deterministic (seed fixed in all runs).
- Checkpoints saved with model state + optimizer state for resumption.
- Inference code decoupled from training; no circular dependencies.

## Presentation Readiness

- All figures have white background for slides.
- Report in NeurIPS format (11pt, 2-column, 8 pages).
- Key results table ready for defense (AA/pep accuracy, 3-seed mean ± std).
- Ablation table complete (7 inference strategies × 3 seeds = 21 configs).
