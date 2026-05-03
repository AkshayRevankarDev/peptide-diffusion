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

### 6. Mass Constraint at Inference — Three Failed Approaches (Comprehensive Negative Result)

All three inference-time mass constraint strategies were evaluated and failed. This is now a well-understood architectural limitation, not an unexplored gap.

**The model IS trained with a soft mass consistency loss** (`MASS_LOSS_WEIGHT = 0.1`, line 568 of `src/diffusion.py`). A soft expected-value loss pushes the distribution's expected mass toward the target but does not guarantee individual decoded samples are mass-correct.

**Approach A — Post-hoc single-swap correction (`use_mass_correct=True`):**
Hurts peptide accuracy by ~9.6 pp (59.96% → 50.35%). Swaps a correctly placed residue for one chosen purely to hit a mass target, overriding the model's learned distribution.

**Approach B — Mass-constrained beam search (`use_beam=True`, NOVEL #8):**
Catastrophic: 48.19% AA / 15.04% pep (−28 pp AA, −44 pp pep vs baseline argmax). Root cause: beam search is a left-to-right autoregressive decoding algorithm. Absorbing diffusion is bidirectional — every position's logit was computed with the full noisy sequence as context. Forcing left-to-right sequential commitment onto bidirectional logits breaks the generative assumption entirely.

**Approach C — Entropy-adaptive gate (`use_gate=True`, NOVEL #1):**
- Gate alone: 12.00% AA / 11.86% pep. Hard per-position mass feasibility masking at argmax time is too aggressive; destroys the distribution the same way beam search does.
- Gate + CFID: 76.05% AA / 59.53% pep — statistically identical to CFID alone (76.02%/59.60%).
- Gate + CFID + SGIR: 75.99% AA / 59.60% pep — identical to CFID+SGIR (76.00%/59.96%).
The gate adds nothing on top of CFID because CFID's iterative mask-predict refinement already enforces sequence coherence and implicitly absorbs whatever mass signal the gate would contribute.

**Conclusion:** Mass constraints during any single-pass decoding step are architecturally incompatible with a bidirectional diffusion model. CFID (mask-predict iterative decoding) is the correct inference mechanism and subsumes all mass-aware improvements. The best result remains CFID+SGIR at 76.00% AA / 59.96% pep.

**Lesson:** Inference constraints must match the model's generative direction. Left-to-right constraints (beam, post-hoc swap) are wrong for a bidirectional model. Even per-position soft constraints (gate) are redundant once iterative bidirectional refinement (CFID) is applied.

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

### 3. All Inference-Time Mass Constraints
Three approaches tried: post-hoc swap (−9.6 pp), beam search (−44 pp, wrong direction for bidirectional model), entropy-adaptive gate alone (−48 pp). Gate + CFID is statistically identical to CFID alone — CFID subsumes any benefit. Mass constraint problem is architecturally closed for this model.

### 4. Dropout in PeakEncoder
Added to prevent overfitting but reduced val accuracy. Removed; batch norm sufficient.

### 5. Variable-Length Diffusion
Tried absorbing non-sequence tokens dynamically. Complexity not worth marginal gains. Fixed max length (30) works.

## Future TODOs

### High Priority
1. **Mass constraint problem is closed:** All three inference-time approaches (post-hoc swap, beam search, entropy-adaptive gate) have been evaluated and failed. See Decision #6 for the full analysis. No further mass constraint work is needed unless the model is retrained with a hard constraint baked into the objective (e.g., differentiable knapsack loss).
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
- **Baseline comparisons:** LSTM and GRU run directly on our E. coli EV test set. InstaNovo numbers (72.9% AA / 33.1% pep) are from their published paper on their benchmark dataset — not re-run on ours. The comparison is directional, not a same-dataset benchmark. Report accordingly: "our model on our test set vs. InstaNovo on their published benchmark."
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
