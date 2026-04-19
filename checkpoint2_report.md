# Checkpoint 2 Progress Report: Multinomial Diffusion Model for Peptide Sequencing

**Team:** Akshay Mohan Revankar, Sanika Nanjan, Vaishak Girish Kumar
**Project:** Peptide and Protein Sequencing by Multinomial Diffusion Model
**Date:** April 19, 2026

---

## 1. Project Progress Since Checkpoint 1

In Checkpoint 1, we successfully implemented out Data Exploration (EDA), established our rigorous data preprocessing pipeline extracting tandem mass spectrometry (MS/MS) features, and built a baseline LSTM encoder-decoder model. Our primary goal set for Checkpoint 2 was to formally pivot from the sequential baseline model to the proposed core of our project: the **Multinomial Diffusion Model**.

Over the last several weeks, we have made the following progress:

### 1.1 Architecture Implementation
We have successfully implemented the structural framework of the discrete diffusion model tailored specifically for categorical amino acid sequences:
- **Spectrum Encoder (`SpectrumEncoder`)**: A feedforward neural network designed to compress the 20,000-dimensional MS/MS spectrum vector into a dense 256-dimensional contextual embedding.
- **Categorical Forward Process (`CategoricalDiffusion`)**: A noise-adding forward schedule that progressively corrupts an initial "true" peptide sequence into a uniform noisy distribution over 100 timesteps ($T=100$).
- **Diffusion Transformer Decoder (`DiffusionTransformer`)**: We replaced the LSTM decoder with a more robust multi-head Attention Transformer structure. The Transformer is conditioned on the time-step embeddings and the MS/MS context to iteratively denoise the latent states back into valid peptide chains.

### 1.2 Training Pipeline & Preliminary Verification
We expanded our data-loader routines to integrate tightly with the new discrete variable formulation. We created a proof-of-concept Jupyter notebook (`04_diffusion.ipynb`) that constructs the dataset, passes it through the new Transformer-based diffusion pipeline, calculates the Cross-Entropy bounds for the discrete variables, and handles backpropagation.

**Preliminary Result Highlights:**
While our current focus was to ensure structural integrity rather than final convergence, our preliminary miniature training run successfully optimized the variational lower bounds over the MS/MS spectrum subsets: 
- Training loops initialize successfully on PyTorch with GPU compatibility.
- The forward permutation masking operates logically, verifying our categorical corruption approach.
- Over trial epochs on small subsets, gradient descent shows stable decline across the negative log-likelihood bounds, verifying the mathematical translation of the discrete diffusion equations into code.

![Preliminary Training Loss Curve](/Users/akshaymohanrevankar/Desktop/peptide%20/peptide-diffusion/loss_curve.png)

---

## 2. Future Plan to Final Report

Our focus now shifts entirely to hyperparameter tuning, full-scale training on the E. coli EV spectra, and rigorous evaluation. 

The roadmap for the weeks remaining before the Final Submission on May 11, 2026 is:

1. **Week 1 (April 20 - April 26): Full Dataset Training and Inference**
   - Execute parallelized training of the Multinomial Diffusion Model on our complete preprocessed E. coli and Wastewater datasets using available High-Performance Compute clusters.
   - Implement the complete Reverse Process ancestral sampling method for model inference, enabling end-to-end peptide prediction during testing.

2. **Week 2 (April 27 - May 3): Quantitative Evaluation & Ablation**
   - Evaluate the newly trained diffusion model against our Checkpoint 1 Baseline LSTM model using exact sequence accuracy and amino-acid level similarity thresholds.
   - Integrate the Mass-Constraint gate to forcibly condition the model, bounding the generated peptide mass closer to the precursor MZ read by the mass-spectrometer.

3. **Week 3 (May 4 - Final Presentation): Final Write-Up & Packaging**
   - Synthesize all accuracy curves and loss distributions cleanly into evaluation plots.
   - Finalize the codebase structure, document standard operating procedures for reviewers on GitHub, and record/prepare our final comprehensive project presentation slides.

---
*End of Report*
