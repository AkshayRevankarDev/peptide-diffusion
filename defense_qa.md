# Defense Q&A Prep — Peptide Diffusion (CSE 676)

## Hard Technical Questions

**Q: You trained with a mass loss. Why does post-hoc mass filtering still fail?**
A: The mass loss is a soft, expected-value term. It pushes the distribution's average predicted mass toward the precursor target, but it does not guarantee that any individual argmax sample lands within tolerance. When we then apply a hard post-hoc swap, we are overriding a residue the model placed for sequence-level reasons — learned chemistry and context — with one chosen purely to satisfy a mass arithmetic condition. That swap is correct on mass and wrong on sequence more often than not, which is why accuracy drops by 9.6 pp.

**Q: Why does beam search fail so catastrophically?**
A: Beam search is a left-to-right algorithm: it commits to position 1, then conditions position 2 on that committed choice, and so on. Our model is bidirectional. Every token's logit at t=0 was computed with the full noisy sequence as context — all 32 positions attended to each other simultaneously. The moment you force left-to-right commitment onto those bidirectional logits, you invalidate the conditioning the model relied on. The distribution collapses because each committed prefix is inconsistent with what the remaining logits assumed. That is why AA recall falls from 76% to 48% — the model is being decoded in a fundamentally different mode than it was trained.

**Q: Is the +26.9 pp over InstaNovo a fair comparison?**
A: It is directional, not a same-benchmark comparison. InstaNovo's 33.1% is reported on their own held-out test set; our 59.96% is on our E. coli EV test set. Both are evaluated at 5% FDR with the same positional accuracy metric, so the comparison is methodologically reasonable. We cannot claim we would beat InstaNovo on their benchmark without running on their data. What we can say is that on our dataset, with our architecture, we substantially outperform their published number, and we outperform CasaNovo (39%) by a similar margin.

**Q: CFID stands for "Consistency-Folding by ID" — what does that mean exactly?**
A: CFID is our name for the decoding strategy: we run mask-predict iterative decoding, which is grounded in Ghazvininejad et al. (2019) and LLaDA (2025). At each of 8 iterations, we commit the highest-confidence positions and re-corrupt the uncertain ones, letting the bidirectional model refine them conditioned on what has already been committed. We call it CFID because it uses ESM-2 structural consistency as part of the reranking signal. The name is project-internal — the underlying algorithm is standard mask-predict.

**Q: Why only 472 test spectra? Is that enough?**
A: The E. coli EV dataset has roughly 3,600 spectra with ground truth from database search. At a 70/15/15 split, the test set is approximately 540 spectra before FDR filtering, and 472 pass the 5% FDR threshold. That is a small set by large-scale proteomics standards, but it is the entire available ground-truth benchmark for this organism and sample type. We report 3-seed mean and standard deviation precisely because of the small set size — the ±3.82 pp standard deviation on peptide accuracy captures that uncertainty honestly.

**Q: What does self-conditioning actually do, mechanistically?**
A: During training, 50% of steps run a no-gradient forward pass first to get a rough sequence prediction, then pass that prediction as an additional input token sequence to the actual forward pass. The model learns to refine a draft rather than predict from scratch. At inference, this means each reverse diffusion step conditions on the previous step's best guess, creating a preview-then-refine loop. The gain is +14.6 pp on peptide accuracy — the largest single improvement in the ablation — because predicting from a partial draft is a much easier task than predicting from pure noise.

**Q: Why is peptide accuracy variance so much higher than AA recall variance?**
A: AA recall is a positional average — even a mostly-wrong sequence gets partial credit at every correct position. Peptide accuracy requires the entire sequence to be correct, so a single wrong residue anywhere is a full miss. That makes it sensitive to difficult spectra and sequence length. CFID+SGIR reduces this variance (±3.82 pp vs ±6.49 pp for argmax) because reranking stabilizes which candidate wins, making predictions less sensitive to stochastic differences across seeds.

**Q: What would you do next if you had more time?**
A: Three things. First, BLAST the six wastewater peptides — we have sequences but no organism assignment. Second, evaluate on a larger multi-species benchmark to test generalization beyond E. coli. Third, incorporate a hard mass constraint into the training objective rather than at inference — something like a differentiable knapsack loss that forces the model to generate mass-consistent sequences during training, not just in expectation.

**Q: The attention mask fix — why did this take two weeks to find?**
A: During training, spectra are batched together and processed through the transformer simultaneously. Without masking, cross-attention between spectra in the same batch created spurious correlations — the model effectively learned to "cheat" by looking at other spectra's patterns. Validation loss appeared fine because the same batch structure was used for validation. The problem only manifested on held-out test data where the batch-level shortcuts were unavailable. We identified it by comparing batch-size-1 inference against batched inference and observing the discrepancy.

---

## Softer / Presentation Questions

**Q: What is the key takeaway from this project?**
A: The PeakEncoder is what makes the problem tractable — without it, absorbing diffusion performs no better than a GRU. Self-conditioning is what makes it competitive — it accounts for most of the peptide accuracy gain. And the mass constraint experiments clarify something fundamental: inference-time constraints must match the model's generative direction. Bidirectional models cannot be decoded with left-to-right algorithms.

**Q: How does this help with wastewater or novel organisms?**
A: Database search requires a reference proteome. When you are sequencing a mixed microbial community from wastewater, you do not have one. De novo sequencing generates peptide sequences directly from the spectrum with no reference, so proteins from organisms not in any database can still be identified. Our six wastewater peptides passed 5% FDR and ESM-2 structural plausibility checks — we do not know what organisms they come from yet, but the model found them without needing to.

**Q: Why absorbing diffusion and not Gaussian diffusion?**
A: Amino acid sequences are discrete — each position takes one of 20 values. Gaussian diffusion operates in continuous space and requires quantization back to tokens at inference, which introduces a source of error and breaks the theoretical guarantees of the generative process. Absorbing diffusion corrupts tokens by replacing them with a special MASK token and learns to unmask them, which is a native discrete operation. The generative process and the sequence representation are aligned.
