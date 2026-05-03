# Presentation Script — Peptide Diffusion (CSE 676)
# Estimated time: ~12 minutes + Q&A
# Speaker assignments suggested in [brackets] — adjust to your team

---

## Slide 1: Title [Vaishak]

"Good afternoon. Our project is on de novo peptide sequencing using absorbing diffusion.
The core question we set out to answer is: can a diffusion model read a protein sequence
directly from a mass spectrum, with no reference database at all?
I'm Vaishak, with Akshay and Sanika, and we'll walk you through what we built and what we found."

---

## Slide 2: The Problem [Vaishak]

"When a mass spectrometer fragments a peptide, it produces a spectrum — a set of peaks, each
corresponding to a fragment ion at a particular mass-to-charge ratio. Standard proteomics
identifies peptides by matching that spectrum against a database of known proteins.

The problem is: if the protein is not in the database, it cannot be identified. This matters for
novel organisms, environmental samples, and metaproteomics — settings where there is no
reference proteome. De novo sequencing bypasses the database entirely. Given a spectrum, the
model generates a sequence directly."

---

## Slide 3: Why Diffusion [Akshay]

"Most de novo tools — DeepNovo, CasaNovo, InstaNovo — use autoregressive decoding.
They predict one amino acid at a time, left to right, which creates two problems: errors
compound, and the model cannot revise earlier positions in light of later ones.

We use absorbing multinomial diffusion. Every position starts as a MASK token, and the model
learns to unmask all positions simultaneously. This is bidirectional — position 1 sees position
30 during every forward pass. That bidirectionality is what makes self-conditioning effective,
and it is what distinguishes our approach from autoregressive baselines."

---

## Slide 4: Architecture [Akshay]

"Our model has three targeted additions on top of absorbing diffusion.

First, the PeakEncoder. It maps the raw spectrum — up to 800 peaks, each an m/z and intensity
pair — into a hidden representation using sinusoidal embeddings. The encoder also includes a
learnable scalar bias for b/y ion pairs, initialized at zero so the model starts data-driven
and learns the physics from the data. Removing the PeakEncoder drops AA recall from 76% to 42%
— the same as a GRU baseline. The encoder is not a convenience; it is what makes the problem tractable.

Second, self-conditioning. At each reverse diffusion step, we first run a no-gradient forward
pass to get a rough sequence prediction, then pass that prediction as an additional input to
the actual forward pass. The model learns to refine a draft rather than predict from noise.
This single addition accounts for +14.6 percentage points in peptide accuracy — the largest
single gain in our ablation.

Third, the attention mask fix. Early training allowed cross-attention between spectra in the
same batch. The model learned spurious batch-level shortcuts that disappeared at test time.
Adding a mask that restricts each spectrum to attend only to itself resolved a persistent
train-test gap."

---

## Slide 5: Inference — CFID and SGIR [Sanika]

"At inference, we run 20 reverse diffusion steps with a cosine noise schedule. The final
step uses CFID: mask-predict iterative decoding. We commit the highest-confidence positions
and re-corrupt the uncertain ones, letting the model refine them conditioned on what
is already committed. This runs for 8 iterations.

After decoding, SGIR rescores candidates using the observed spectrum directly — comparing
the theoretical fragment ion distribution of each predicted sequence against the measured
peaks, weighted by intensity. CFID and SGIR together contribute orthogonal signal: CFID
improves sequence coherence, SGIR grounds the prediction in the physical measurement.

Their combination, CFID+SGIR, is our best configuration: 76.00% AA recall,
59.96% peptide accuracy at 5% FDR."

---

## Slide 6: Results [Sanika]

"Here is the development progression. The LSTM baseline reaches 2.68% peptide accuracy —
essentially random. The GRU baseline reaches 6.70%. Absorbing diffusion without the PeakEncoder
reaches 6.36% — no better than GRU, which confirms the encoder is responsible for the jump,
not the diffusion architecture itself.

Adding the PeakEncoder and B/Y bias brings argmax accuracy to 36.51%. Self-conditioning
then brings it to 57.49%, and CFID+SGIR reranking pushes it to 59.96%.

For reference, InstaNovo — the 2025 published SOTA — reports 33.1% peptide accuracy on their
benchmark. On our E. coli EV test set, our model reaches 59.96%. That is a difference of
26.9 percentage points. We note that this is a directional comparison across datasets,
not a same-benchmark head-to-head."

---

## Slide 7: Ablation [Vaishak]

"We ran seven inference strategies across three random seeds each, for 21 total configurations.

CFID+SGIR is Pareto-optimal on both metrics. The two mass-correction strategies — post-hoc
single-swap and CFID with post-hoc correction — consistently reduce peptide accuracy by
8 to 12 percentage points. We will explain why in a moment."

---

## Slide 8: Mass Constraint — Negative Results [Akshay]

"The mass constraint experiments are the most instructive part of the project.

We had three hypotheses for enforcing precursor mass consistency at inference.
All three failed, and they failed for the same underlying reason.

Post-hoc single-swap correction: minus 9.6 percentage points. When the model places a residue,
it does so for sequence-level reasons. Replacing it with a different residue to satisfy a
mass arithmetic condition overrides that learned signal.

Mass-constrained beam search: minus 44 percentage points. Beam search decodes left to right,
committing one position at a time. Our model is bidirectional — every token's logit at
t=0 was computed with the full noisy sequence as context. Forcing left-to-right commitment
onto bidirectional logits invalidates the conditioning the model relied on. The distribution
collapses.

Entropy-adaptive gate: minus 48 percentage points when applied alone. Combined with CFID,
it matches CFID alone to within 0.4 percentage points — the gate adds nothing, because
CFID's iterative bidirectional refinement already enforces sequence coherence and subsumes
any mass-feasibility benefit.

The conclusion: single-pass mass constraints are architecturally incompatible with a
bidirectional generative model. CFID is the right decoding strategy."

---

## Slide 9: Wastewater Application [Sanika]

"We applied the trained model to wastewater metaproteomics — Sample 2, a mixed microbial
community with no reference proteome. Six peptides survived 5% FDR filtering.

We validated structural plausibility using ESM-2 pseudo-perplexity. All six sequences have
z-scores below 1.0 relative to the E. coli ground-truth distribution, indicating the predicted
sequences are structurally reasonable. We have not yet confirmed organism identity through BLAST,
so we report these as candidates pending validation. This is the real-world case the model
was designed for: no database, no reference, sequence recovered from the spectrum alone."

---

## Slide 10: Conclusion [Vaishak]

"To summarize: absorbing diffusion with a PeakEncoder and self-conditioning reaches 76.00% AA
recall and 59.96% peptide accuracy on our E. coli EV benchmark. The PeakEncoder is the
enabling component; self-conditioning is the largest single performance driver.

The mass constraint experiments reveal a general principle: inference constraints must match
the model's generative direction. Bidirectional models cannot be decoded with left-to-right
algorithms.

Going forward: BLAST validation of the wastewater sequences, evaluation on larger multi-species
benchmarks, and incorporating a hard mass constraint into the training objective rather than
at inference.

Thank you. We are happy to take questions."

---

## Backup Notes (not spoken — for Q&A reference)

- Self-conditioning training: 50% of steps only; no-gradient first pass, argmax prediction used as self_cond input
- CFID iterations: 8 rounds; linear commitment schedule (commit L/8 more positions each round)
- Beam search beam width: 20; tolerance 0.1 Da
- Gate base tolerance: 0.02 Da, scaled by entropy / log(V)
- Mass loss weight: 0.1, relative L1 on expected sequence mass
- Test set: 472 spectra after 5% FDR (from ~3600 total, 70/15/15 split, seed 42)
- Seeds tested: 0, 1, 2 — all to 200 epochs with cosine LR schedule, AdamW
