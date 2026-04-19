"""
src/esm_scoring.py
S-1 (NOVEL #2): Per-residue ESM-2 pseudo-perplexity scorer
S-2 (NOVEL #6): EV cargo anomaly detection via PPL z-score
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── ESM-2 model ──────────────────────────────────────────────────────────────
_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

def _load_esm2():
    from transformers import AutoTokenizer, EsmForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model     = EsmForMaskedLM.from_pretrained(_MODEL_NAME)
    model.eval()
    return model, tokenizer


# ── NOVEL #2: Per-residue scorer ──────────────────────────────────────────────
def score_per_residue(sequence: str, model, tokenizer) -> tuple:
    """
    One-fell-swoop masked forward pass (Kantroo 2024):
    mask ALL positions simultaneously, single forward pass.

    Returns:
        ppl_scalar      (float)       — exp(-mean log-prob)
        log_probs       (list[float]) — per-position log-probability
        anomalous_pos   (list[int])   — positions where log_prob < -3
    """
    if len(sequence) < 5:
        return None, None, None

    inputs    = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs["input_ids"]          # (1, L+2)

    masked              = input_ids.clone()
    masked[0, 1:-1]     = tokenizer.mask_token_id   # mask all residues

    with torch.no_grad():
        logits = model(masked).logits        # (1, L+2, vocab)

    log_probs = []
    for i in range(len(sequence)):
        true_id = input_ids[0, i + 1].item()
        lp      = F.log_softmax(logits[0, i + 1], dim=-1)[true_id].item()
        log_probs.append(lp)

    ppl_scalar     = math.exp(-sum(log_probs) / len(sequence))
    anomalous_pos  = [i for i, lp in enumerate(log_probs) if lp < -3.0]

    return ppl_scalar, log_probs, anomalous_pos


def score_sequences(sequences, model, tokenizer) -> list:
    """
    Score a list of (sequence, model_label, is_correct, spectrum_id) tuples.
    Returns a list of row dicts ready for DataFrame construction.
    """
    rows = []
    for seq, model_label, is_correct, spectrum_id in sequences:
        if not seq or len(seq) < 5:
            continue
        ppl_scalar, log_probs, anomalous_pos = score_per_residue(seq, model, tokenizer)
        if ppl_scalar is None:
            continue
        rows.append({
            "sequence":          seq,
            "ppl_scalar":        ppl_scalar,
            "ppl_per_residue":   json.dumps(log_probs),
            "model":             model_label,
            "is_correct":        is_correct,
            "spectrum_id":       spectrum_id,
            "anomalous_positions": json.dumps(anomalous_pos),
        })
    return rows


# ── NOVEL #6: EV cargo anomaly detection ─────────────────────────────────────
def detect_ev_cargo_anomalies(df: pd.DataFrame,
                               gt_sequences: list,
                               model,
                               tokenizer,
                               z_thresh: float = 2.5,
                               log_threshold: float = -5.0) -> pd.DataFrame:
    """
    Compute PPL z-score vs ground-truth E. coli EV reference distribution.
    Flag predictions where z > z_thresh AND spectral_logprob > log_threshold.

    Adds columns: ppl_zscore (float), ev_cargo_anomaly (bool).
    """
    # Build reference distribution from ground-truth sequences
    ref_ppls = []
    for seq in gt_sequences:
        if seq and len(seq) >= 5:
            ppl, _, _ = score_per_residue(seq, model, tokenizer)
            if ppl is not None:
                ref_ppls.append(ppl)

    if len(ref_ppls) < 2:
        df["ppl_zscore"]       = float("nan")
        df["ev_cargo_anomaly"] = False
        return df

    ppl_mean = float(np.mean(ref_ppls))
    ppl_std  = float(np.std(ref_ppls)) or 1.0

    z_scores  = []
    anomalies = []

    # spectral_logprob column may not exist for all rows (e.g. ESM-only scoring)
    has_lp = "spectral_logprob" in df.columns

    for _, row in df.iterrows():
        ppl = row["ppl_scalar"]
        z   = (ppl - ppl_mean) / ppl_std
        z_scores.append(z)
        if has_lp:
            sp_lp    = row.get("spectral_logprob", float("-inf"))
            is_anom  = bool(z > z_thresh and sp_lp > log_threshold)
        else:
            is_anom  = bool(z > z_thresh)
        anomalies.append(is_anom)

    df = df.copy()
    df["ppl_zscore"]       = z_scores
    df["ev_cargo_anomaly"] = anomalies
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_esm_scoring(diffusion_csv: str,
                    lstm_csv:     str   = None,
                    gru_csv:      str   = None,
                    gt_sequences: list  = None,
                    out_csv:      str   = "results/esm2_scores.csv",
                    figures_dir:  str   = "figures") -> pd.DataFrame:
    """
    Score diffusion, LSTM, GRU predictions + ground-truth + random sequences.
    Saves results/esm2_scores.csv, Figure 1 (violin), Figure 2 (heatmap).
    """
    print("Loading ESM-2 model …")
    model, tokenizer = _load_esm2()

    # (seq, model_label, is_correct, spectrum_id, spectral_logprob)
    sequences_to_score = []
    # spectral_logprob lookup keyed by (model, spectrum_id) for anomaly detection
    sp_lp_lookup: dict = {}

    # ── diffusion predictions ─────────────────────────────────────────────────
    # CSV columns: spectrum_id, sequence, spectral_logprob, gate_confidence
    if os.path.exists(diffusion_csv):
        df_diff = pd.read_csv(diffusion_csv)
        for idx, row in df_diff.iterrows():
            sid  = int(row.get("spectrum_id", idx))
            slp  = float(row.get("spectral_logprob", float("nan")))
            seq  = str(row["sequence"])
            sequences_to_score.append((seq, "diffusion", False, sid, slp))
            sp_lp_lookup[("diffusion", sid)] = slp

    # ── LSTM predictions ──────────────────────────────────────────────────────
    # CSV columns (from 03_baseline.ipynb): predicted_sequence, true_sequence, correct
    if lstm_csv and os.path.exists(lstm_csv):
        df_lstm = pd.read_csv(lstm_csv)
        for idx, row in df_lstm.iterrows():
            pred_seq = str(row.get("predicted_sequence", row.get("sequence", "")))
            is_ok    = bool(row.get("correct", row.get("is_correct", False)))
            sid      = int(row.get("spectrum_id", idx))
            sequences_to_score.append((pred_seq, "lstm", is_ok, sid, float("nan")))
            # collect ground-truth sequences from this CSV
            gt_seq = str(row.get("true_sequence", ""))
            if gt_seq and len(gt_seq) >= 5 and gt_seq not in (gt_sequences or []):
                if gt_sequences is None:
                    gt_sequences = []
                if gt_seq not in gt_sequences:
                    gt_sequences.append(gt_seq)

    # ── GRU predictions ───────────────────────────────────────────────────────
    # CSV columns (from 03_baseline.ipynb): predicted_sequence, true_sequence, correct
    if gru_csv and os.path.exists(gru_csv):
        df_gru = pd.read_csv(gru_csv)
        for idx, row in df_gru.iterrows():
            pred_seq = str(row.get("predicted_sequence", row.get("sequence", "")))
            is_ok    = bool(row.get("correct", row.get("is_correct", False)))
            sid      = int(row.get("spectrum_id", idx))
            sequences_to_score.append((pred_seq, "gru", is_ok, sid, float("nan")))

    # ── Ground-truth sequences ────────────────────────────────────────────────
    if gt_sequences:
        for i, seq in enumerate(gt_sequences):
            sequences_to_score.append((seq, "ground_truth", True, i, float("nan")))

    # ── Random baseline ───────────────────────────────────────────────────────
    import random
    aa_chars = list("ACDEFGHIKLMNPQRSTVWY")
    random.seed(0)
    for i in range(50):
        rand_seq = "".join(random.choices(aa_chars, k=random.randint(8, 20)))
        sequences_to_score.append((rand_seq, "random", False, i, float("nan")))

    print(f"Scoring {len(sequences_to_score)} sequences …")
    # score_sequences only takes 4-tuples; strip spectral_logprob before passing
    rows = score_sequences([(s, m, c, sid) for s, m, c, sid, _ in sequences_to_score],
                           model, tokenizer)

    df = pd.DataFrame(rows)

    # Attach spectral_logprob back for diffusion rows (used by anomaly detection)
    def _get_slp(row):
        return sp_lp_lookup.get((row["model"], row["spectrum_id"]), float("nan"))
    df["spectral_logprob"] = df.apply(_get_slp, axis=1)

    # ── NOVEL #6: anomaly detection ───────────────────────────────────────────
    if gt_sequences:
        df = detect_ev_cargo_anomalies(df, gt_sequences, model, tokenizer)
    else:
        df["ppl_zscore"]       = float("nan")
        df["ev_cargo_anomaly"] = False

    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows → {out_csv}")

    _make_figures(df, figures_dir)
    return df


# ── Figures ───────────────────────────────────────────────────────────────────
def _make_figures(df: pd.DataFrame, figures_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available — skipping figures.")
        return

    os.makedirs(figures_dir, exist_ok=True)

    # Figure 1: violin plot — PPL scalar by model group
    fig, ax = plt.subplots(figsize=(8, 5))
    model_order = [m for m in ["ground_truth", "diffusion", "lstm", "gru", "random"]
                   if m in df["model"].unique()]
    sns.violinplot(data=df, x="model", y="ppl_scalar", order=model_order,
                   cut=0, ax=ax)
    ax.set_title("ESM-2 Perplexity by Model (Figure 1)")
    ax.set_xlabel("Model"); ax.set_ylabel("PPL scalar")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "figure1_esm2_violin.png"), dpi=150)
    plt.close(fig)
    print(f"Saved Figure 1 → {figures_dir}/figure1_esm2_violin.png")

    # Figure 2: per-position PPL heatmap (first 10 test spectra, 3 models)
    target_models = ["diffusion", "gru", "lstm"]
    available     = [m for m in target_models if m in df["model"].unique()]
    if not available:
        return

    first_ids = sorted(df[df["model"] == available[0]]["spectrum_id"].unique())[:10]
    sub       = df[df["spectrum_id"].isin(first_ids) & df["model"].isin(available)]

    max_len = 0
    for _, row in sub.iterrows():
        lp = json.loads(row["ppl_per_residue"])
        max_len = max(max_len, len(lp))

    if max_len == 0:
        return

    n_rows = len(available) * len(first_ids)
    heat   = np.full((n_rows, max_len), np.nan)
    ylabels = []

    row_idx = 0
    for sid in first_ids:
        for mdl in available:
            r = sub[(sub["spectrum_id"] == sid) & (sub["model"] == mdl)]
            if r.empty:
                ylabels.append(f"{mdl}|{sid}")
                row_idx += 1
                continue
            lp = json.loads(r.iloc[0]["ppl_per_residue"])
            heat[row_idx, :len(lp)] = lp
            ylabels.append(f"{mdl}|{sid}")
            row_idx += 1

    fig, ax = plt.subplots(figsize=(max(8, max_len // 2), max(4, n_rows // 2)))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn",
                   vmin=-5, vmax=0, interpolation="nearest")
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Residue position"); ax.set_title("Per-position log-prob heatmap (Figure 2)")
    fig.colorbar(im, ax=ax, label="log-prob")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "figure2_esm2_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"Saved Figure 2 → {figures_dir}/figure2_esm2_heatmap.png")


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESM-2 per-residue scorer")
    parser.add_argument("--diffusion_csv", default="results/diffusion_predictions.csv")
    parser.add_argument("--lstm_csv",      default="results/lstm_predictions.csv")
    parser.add_argument("--gru_csv",       default="results/gru_predictions.csv")
    parser.add_argument("--out_csv",       default="results/esm2_scores.csv")
    parser.add_argument("--figures_dir",   default="figures")
    args = parser.parse_args()

    run_esm_scoring(
        diffusion_csv=args.diffusion_csv,
        lstm_csv=args.lstm_csv,
        gru_csv=args.gru_csv,
        out_csv=args.out_csv,
        figures_dir=args.figures_dir,
    )
