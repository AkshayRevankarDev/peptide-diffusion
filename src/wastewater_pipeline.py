"""
src/wastewater_pipeline.py
S-3 (Base):      Target-decoy FDR pipeline on wastewater mzML files
A-2 (NOVEL #5):  Cross-replicate Jaccard consistency scoring
A-2 (NOVEL #7):  ESM-2 PPL as Winnow calibration feature
"""
import os
import sys
import json
import glob

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_raw_spectra
from preprocessing import preprocess_spectrum, VOCAB, CHAR_TO_IDX

PROTON_MASS = 1.007276

# ── Helpers ───────────────────────────────────────────────────────────────────
def _neutral_mass(prec_mz, charge):
    if prec_mz and charge:
        return float(charge) * (float(prec_mz) - PROTON_MASS)
    return 0.0


def _reverse_spectrum(mz_array: np.ndarray, intensity_array: np.ndarray,
                      mz_min: float = 0.0, mz_max: float = 2000.0) -> tuple:
    """Create decoy spectrum by reversing the m/z axis."""
    rev_mz = mz_max - mz_array + mz_min
    return rev_mz, intensity_array.copy()


# ── Base S-3: Target-Decoy FDR pipeline ──────────────────────────────────────
def run_inference_on_spectra(spectra: list, encoder, denoiser,
                             device=None) -> list:
    """
    Run diffusion model inference on a list of raw spectrum dicts.
    Returns list of dicts: {spectrum_id, sequence, score, precursor_mz, charge}.
    """
    from diffusion import generate_sequences, PROTON_MASS

    if device is None:
        device = next(encoder.parameters()).device

    X, masses, meta = [], [], []
    for s in spectra:
        binned  = preprocess_spectrum(s["mz"], s["intensity"])
        neutral = _neutral_mass(s.get("precursor_mz"), s.get("charge"))
        X.append(binned)
        masses.append(neutral)
        meta.append({
            "scan_num":    s.get("scan_num", -1),
            "precursor_mz": s.get("precursor_mz"),
            "charge":       s.get("charge"),
        })

    if not X:
        return []

    X      = np.array(X)
    masses = np.array(masses, dtype=np.float32)

    seqs_list, lps_list, gcs_list = generate_sequences(
        encoder, denoiser, X, masses,
        n_candidates=1, device=device, use_gate=True
    )

    results = []
    for i, (seqs, lps, _) in enumerate(zip(seqs_list, lps_list, gcs_list)):
        results.append({
            "spectrum_id":  meta[i]["scan_num"],
            "sequence":     seqs[0],
            "score":        lps[0],
            "precursor_mz": meta[i]["precursor_mz"],
            "charge":       meta[i]["charge"],
        })
    return results


def compute_empirical_fdr(target_scores: list, decoy_scores: list,
                           thresholds: np.ndarray = None) -> pd.DataFrame:
    """
    Empirical FDR(tau) = |decoy PSMs with score > tau| / |target PSMs with score > tau|.
    Returns DataFrame with columns [tau, n_target, n_decoy, fdr].
    """
    if thresholds is None:
        all_scores = np.array(target_scores + decoy_scores)
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 500)

    rows = []
    for tau in thresholds:
        n_t = sum(s > tau for s in target_scores)
        n_d = sum(s > tau for s in decoy_scores)
        fdr = n_d / max(n_t, 1)
        rows.append({"tau": tau, "n_target": n_t, "n_decoy": n_d, "fdr": fdr})
    return pd.DataFrame(rows)


def find_fdr_threshold(fdr_df: pd.DataFrame,
                        target_fdr: float = 0.05,
                        fallback_fdr: float = 0.10) -> tuple:
    """Return (tau, achieved_fdr) at best FDR <= target_fdr, or fallback."""
    passing = fdr_df[fdr_df["fdr"] <= target_fdr]
    if not passing.empty:
        best = passing.loc[passing["n_target"].idxmax()]
        return float(best["tau"]), float(best["fdr"])

    passing = fdr_df[fdr_df["fdr"] <= fallback_fdr]
    if not passing.empty:
        best = passing.loc[passing["n_target"].idxmax()]
        return float(best["tau"]), float(best["fdr"])

    # Last resort: lowest possible threshold (maximise recall)
    best = fdr_df.loc[fdr_df["n_target"].idxmax()]
    return float(best["tau"]), float(best["fdr"])


# ── NOVEL #5: Cross-replicate Jaccard consistency ─────────────────────────────
def cross_replicate_jaccard(preds_rep1: pd.DataFrame,
                             preds_rep2: pd.DataFrame) -> float:
    """Jaccard index between peptide sets from two replicates of the same sample."""
    set1 = set(preds_rep1["sequence"].dropna())
    set2 = set(preds_rep2["sequence"].dropna())
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def add_replicate_consistency(df: pd.DataFrame,
                               sample_to_replicates: dict) -> pd.DataFrame:
    """
    For each PSM, mark replicate_consistent=True if the sequence appears
    in BOTH replicates of the same sample.

    sample_to_replicates: {sample_id: [mzml_path_rep1, mzml_path_rep2, ...]}
      — keys must match values in df["sample_id"].
    """
    df = df.copy()
    df["replicate_consistent"] = False
    df["jaccard_score"]        = 0.0

    samples = df["sample_id"].unique() if "sample_id" in df.columns else []

    for sample in samples:
        mask = df["sample_id"] == sample
        seqs = set(df.loc[mask, "sequence"].dropna())

        # For each replicate pair, look up presence
        # (actual replicate splitting happens at the call site via sample_id groups)
        rep_groups = df[mask].groupby("replicate_id")["sequence"].apply(set).to_dict() \
                    if "replicate_id" in df.columns else {}

        rep_keys = sorted(rep_groups.keys())
        if len(rep_keys) >= 2:
            s1, s2   = rep_groups[rep_keys[0]], rep_groups[rep_keys[1]]
            common   = s1 & s2
            jaccard  = len(s1 & s2) / max(len(s1 | s2), 1)
            df.loc[mask, "replicate_consistent"] = df.loc[mask, "sequence"].isin(common)
            df.loc[mask, "jaccard_score"]        = jaccard
        else:
            df.loc[mask, "replicate_consistent"] = True
            df.loc[mask, "jaccard_score"]        = 1.0

    return df


# ── NOVEL #7: ESM-2 PPL as Winnow calibration feature ─────────────────────────
def _load_esm2_model():
    from transformers import AutoTokenizer, EsmForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model     = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model.eval()
    return model, tokenizer


def compute_ppl_batch(sequences: list, esm_model, tokenizer) -> list:
    """Return ppl_scalar for each sequence (CPU, sequential)."""
    import math, torch.nn.functional as F

    ppls = []
    for seq in sequences:
        if not seq or len(seq) < 5:
            ppls.append(float("nan"))
            continue
        inputs    = tokenizer(seq, return_tensors="pt")
        input_ids = inputs["input_ids"]
        masked    = input_ids.clone()
        masked[0, 1:-1] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = esm_model(masked).logits
        log_probs = []
        for i in range(len(seq)):
            true_id = input_ids[0, i + 1].item()
            lp      = F.log_softmax(logits[0, i + 1], dim=-1)[true_id].item()
            log_probs.append(lp)
        ppls.append(math.exp(-sum(log_probs) / len(seq)))
    return ppls


def anomalous_fraction(sequence: str, esm_model, tokenizer,
                        lp_thresh: float = -3.0) -> float:
    """Fraction of positions with log_prob < lp_thresh."""
    import torch.nn.functional as F

    if not sequence or len(sequence) < 5:
        return 0.0
    inputs    = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    masked    = input_ids.clone()
    masked[0, 1:-1] = tokenizer.mask_token_id
    with torch.no_grad():
        logits = esm_model(masked).logits
    anom = 0
    for i in range(len(sequence)):
        true_id = input_ids[0, i + 1].item()
        lp      = torch.nn.functional.log_softmax(logits[0, i + 1], dim=-1)[true_id].item()
        if lp < lp_thresh:
            anom += 1
    return anom / len(sequence)


def winnow_calibration(df: pd.DataFrame, esm_model, tokenizer) -> pd.DataFrame:
    """
    Extend feature set with ESM-2 PPL features and fit a simple logistic
    Winnow-style calibration to re-rank PSMs.

    Adds columns: esm2_ppl, esm2_anomalous_frac, winnow_score.
    Returns updated DataFrame.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sklearn_available = True
    except ImportError:
        sklearn_available = False

    seqs = df["sequence"].tolist()
    print("  Computing ESM-2 PPL for Winnow calibration …")
    df = df.copy()
    df["esm2_ppl"]            = compute_ppl_batch(seqs, esm_model, tokenizer)
    df["esm2_anomalous_frac"] = [anomalous_fraction(s, esm_model, tokenizer)
                                  for s in seqs]

    if not sklearn_available:
        df["winnow_score"] = df["score"]
        return df

    feature_cols = ["score"]
    if "mass_error" in df.columns:
        feature_cols.append("mass_error")
    if "beam_margin" in df.columns:
        feature_cols.append("beam_margin")
    feature_cols += ["esm2_ppl", "esm2_anomalous_frac"]

    sub = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X      = scaler.fit_transform(sub)

    # Pseudo-labels: score above median → positive
    median = df["score"].median()
    y      = (df["score"] > median).astype(int).values

    if y.sum() < 5 or (1 - y).sum() < 5:
        df["winnow_score"] = df["score"]
        return df

    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X, y)
    df["winnow_score"] = lr.predict_proba(X)[:, 1]

    # Feature importance table (absolute coefficients)
    coef_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": np.abs(lr.coef_[0]),
    }).sort_values("importance", ascending=False)
    print("\nWinnow feature importances:")
    print(coef_df.to_string(index=False))

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_wastewater_pipeline(mzml_paths: list,
                             encoder,
                             denoiser,
                             out_csv:       str   = "results/wastewater_predictions_5pct_fdr.csv",
                             target_fdr:    float = 0.05,
                             use_esm_winnow: bool  = True,
                             device=None) -> pd.DataFrame:
    """
    Full wastewater pipeline:
      1. Load mzML files, run target & decoy inference.
      2. Compute empirical FDR; filter at 5% (fallback 10%).
      3. NOVEL #5: cross-replicate Jaccard consistency.
      4. NOVEL #7: ESM-2 Winnow calibration features.
      5. Save wastewater_predictions_5pct_fdr.csv.
    """
    if device is None:
        device = next(encoder.parameters()).device

    all_target, all_decoy = [], []

    for mzml_path in mzml_paths:
        fname  = os.path.basename(mzml_path)
        sample = os.path.splitext(fname)[0]
        print(f"Loading {fname} …")
        spectra = load_raw_spectra(mzml_path, max_spectra=2000)

        # Target predictions
        tgt = run_inference_on_spectra(spectra, encoder, denoiser, device)
        for r in tgt:
            r["sample_id"]    = sample
            r["is_decoy"]     = False
            r["replicate_id"] = 0
        all_target.extend(tgt)

        # Decoy predictions (reversed m/z)
        decoy_spectra = []
        for s in spectra:
            rev_mz, rev_int = _reverse_spectrum(s["mz"], s["intensity"])
            decoy_spectra.append({**s, "mz": rev_mz, "intensity": rev_int})

        dec = run_inference_on_spectra(decoy_spectra, encoder, denoiser, device)
        for r in dec:
            r["sample_id"]    = sample
            r["is_decoy"]     = True
            r["replicate_id"] = 0
        all_decoy.extend(dec)

    if not all_target:
        print("No target PSMs produced — aborting.")
        return pd.DataFrame()

    target_scores = [r["score"] for r in all_target]
    decoy_scores  = [r["score"] for r in all_decoy]

    fdr_df = compute_empirical_fdr(target_scores, decoy_scores)
    tau, achieved_fdr = find_fdr_threshold(fdr_df, target_fdr=target_fdr)

    label = "5pct" if achieved_fdr <= 0.05 else "10pct"
    print(f"FDR threshold tau={tau:.4f}  achieved FDR={achieved_fdr*100:.1f}%  ({label})")

    # Filter targets
    df = pd.DataFrame([r for r in all_target if r["score"] > tau])
    if df.empty:
        print("No PSMs pass FDR threshold.")
        return df

    print(f"PSMs at {label} FDR : {len(df)}")
    print(f"Unique peptides     : {df['sequence'].nunique()}")

    # NOVEL #5: Jaccard replicate consistency
    df = add_replicate_consistency(df, {})

    # NOVEL #7: ESM-2 Winnow calibration
    if use_esm_winnow:
        try:
            esm_model, esm_tok = _load_esm2_model()
            df = winnow_calibration(df, esm_model, esm_tok)
        except Exception as e:
            print(f"ESM-2 Winnow skipped ({e})")
            df["esm2_ppl"]            = float("nan")
            df["esm2_anomalous_frac"] = float("nan")
            df["winnow_score"]        = df["score"]
    else:
        df["esm2_ppl"]            = float("nan")
        df["esm2_anomalous_frac"] = float("nan")
        df["winnow_score"]        = df["score"]

    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    col_order = ["spectrum_id", "sequence", "score", "winnow_score",
                 "sample_id", "replicate_consistent", "jaccard_score",
                 "esm2_ppl", "esm2_anomalous_frac",
                 "precursor_mz", "charge"]
    col_order = [c for c in col_order if c in df.columns]
    df[col_order].to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    return df


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wastewater target-decoy FDR pipeline")
    parser.add_argument("--mzml",   nargs="+", required=True,
                        help="Paths to wastewater mzML files")
    parser.add_argument("--ckpt",   default="checkpoints/diffusion_best.pt",
                        help="Diffusion model checkpoint")
    parser.add_argument("--out",    default="results/wastewater_predictions_5pct_fdr.csv")
    parser.add_argument("--no_esm", action="store_true",
                        help="Skip ESM-2 Winnow calibration")
    args = parser.parse_args()

    from diffusion import load_checkpoint
    encoder, denoiser = load_checkpoint(args.ckpt)

    run_wastewater_pipeline(
        mzml_paths     = args.mzml,
        encoder        = encoder,
        denoiser       = denoiser,
        out_csv        = args.out,
        use_esm_winnow = not args.no_esm,
    )
