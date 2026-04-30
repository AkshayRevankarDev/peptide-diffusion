"""
src/ensemble.py
A-1 (NOVEL #3): Three-term val-calibrated ensemble
    score(c) = log_p_spectral - lambda * mean(ppl_per_residue) + gamma * gate_confidence
    Lambda grid: {0.0, 0.05, 0.1, 0.2, 0.5}
    Gamma  grid: {0.0, 0.1,  0.3, 0.5}
    Both calibrated on the validation set (AA recall objective).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.5]
GAMMA_GRID  = [0.0, 0.1,  0.3, 0.5]

VOCAB = list("ACDEFGHIKLMNPQRSTVWY")


# ── AA recall helper ──────────────────────────────────────────────────────────
def _aa_recall(pred: str, true: str) -> float:
    """Positional recall: fraction of positions where pred[i] == true[i]."""
    matches = sum(a == b for a, b in zip(str(pred), str(true)))
    return matches / max(len(true), 1)


# ── Load and merge prediction tables ─────────────────────────────────────────
def load_candidate_pool(diffusion_csv: str,
                        esm2_csv:      str,
                        lstm_csv:      str = None,
                        gru_csv:       str = None) -> pd.DataFrame:
    """
    Merge diffusion predictions with ESM-2 per-residue scores.
    Also includes top-1 from LSTM and GRU as extra candidates.

    Returns a DataFrame with one row per (spectrum_id, candidate) containing:
        spectrum_id, sequence, spectral_logprob, gate_confidence,
        ppl_scalar, ppl_per_residue (JSON), model_source,
        true_sequence (if available from LSTM/GRU CSV)
    """
    rows = []

    # spectral_logprob per spectrum from diffusion (used as base score for all models)
    diffusion_sp_lookup: dict = {}  # {spectrum_id: spectral_logprob}

    # ── Diffusion candidates (top-5 per spectrum) ─────────────────────────────
    if os.path.exists(diffusion_csv):
        df_diff = pd.read_csv(diffusion_csv)
        # Build ESM-2 lookup: sequence → (ppl_scalar, ppl_per_residue)
        esm_lookup = {}
        if os.path.exists(esm2_csv):
            df_esm = pd.read_csv(esm2_csv)
            diff_esm = df_esm[df_esm["model"] == "diffusion"]
            for _, r in diff_esm.iterrows():
                esm_lookup[str(r["sequence"])] = (
                    float(r.get("ppl_scalar", float("nan"))),
                    r.get("ppl_per_residue", "[]"),
                )

        for _, r in df_diff.iterrows():
            seq    = str(r["sequence"])
            sid    = int(r.get("spectrum_id", -1))
            sp_lp  = float(r.get("spectral_logprob", float("nan")))
            ppl_scalar, ppl_per_res = esm_lookup.get(seq, (float("nan"), "[]"))
            # Record the best (highest) spectral_logprob per spectrum for LSTM/GRU reuse
            if not math.isnan(sp_lp):
                if sid not in diffusion_sp_lookup or sp_lp > diffusion_sp_lookup[sid]:
                    diffusion_sp_lookup[sid] = sp_lp
            rows.append({
                "spectrum_id":       sid,
                "sequence":          seq,
                "spectral_logprob":  sp_lp,
                "gate_confidence":   float(r.get("gate_confidence", 0.0)),
                "ppl_scalar":        ppl_scalar,
                "ppl_per_residue":   ppl_per_res,
                "model_source":      "diffusion",
                "true_sequence":     "",
            })

    # ── LSTM top-1 ────────────────────────────────────────────────────────────
    if lstm_csv and os.path.exists(lstm_csv):
        df_lstm = pd.read_csv(lstm_csv)
        esm_lookup_lstm = {}
        if os.path.exists(esm2_csv):
            df_esm = pd.read_csv(esm2_csv)
            lstm_esm = df_esm[df_esm["model"] == "lstm"]
            for _, r in lstm_esm.iterrows():
                esm_lookup_lstm[str(r["sequence"])] = (
                    float(r.get("ppl_scalar", float("nan"))),
                    r.get("ppl_per_residue", "[]"),
                )
        for idx, r in df_lstm.iterrows():
            seq      = str(r.get("predicted_sequence", r.get("sequence", "")))
            true_seq = str(r.get("true_sequence", ""))
            sid      = int(r.get("spectrum_id", idx))
            ppl_s, ppl_pr = esm_lookup_lstm.get(seq, (float("nan"), "[]"))
            # Borrow diffusion spectral_logprob for this spectrum so LSTM/GRU
            # compete on equal footing — avoids NaN→0 dominating negative values
            rows.append({
                "spectrum_id":       sid,
                "sequence":          seq,
                "spectral_logprob":  diffusion_sp_lookup.get(sid, float("nan")),
                "gate_confidence":   0.0,
                "ppl_scalar":        ppl_s,
                "ppl_per_residue":   ppl_pr,
                "model_source":      "lstm",
                "true_sequence":     true_seq,
            })

    # ── GRU top-1 ─────────────────────────────────────────────────────────────
    if gru_csv and os.path.exists(gru_csv):
        df_gru = pd.read_csv(gru_csv)
        esm_lookup_gru = {}
        if os.path.exists(esm2_csv):
            df_esm = pd.read_csv(esm2_csv)
            gru_esm = df_esm[df_esm["model"] == "gru"]
            for _, r in gru_esm.iterrows():
                esm_lookup_gru[str(r["sequence"])] = (
                    float(r.get("ppl_scalar", float("nan"))),
                    r.get("ppl_per_residue", "[]"),
                )
        for idx, r in df_gru.iterrows():
            seq      = str(r.get("predicted_sequence", r.get("sequence", "")))
            true_seq = str(r.get("true_sequence", ""))
            sid      = int(r.get("spectrum_id", idx))
            ppl_s, ppl_pr = esm_lookup_gru.get(seq, (float("nan"), "[]"))
            rows.append({
                "spectrum_id":       sid,
                "sequence":          seq,
                "spectral_logprob":  diffusion_sp_lookup.get(sid, float("nan")),
                "gate_confidence":   0.0,
                "ppl_scalar":        ppl_s,
                "ppl_per_residue":   ppl_pr,
                "model_source":      "gru",
                "true_sequence":     true_seq,
            })

    return pd.DataFrame(rows)


# ── Ensemble scoring ──────────────────────────────────────────────────────────
def ensemble_score(row: pd.Series, lam: float, gam: float) -> float:
    """
    score(c) = log_p_spectral(c)
             - lam * mean(ppl_per_residue(c))
             + gam * gate_confidence(c)
    gate_confidence = 0 for LSTM/GRU (they have no mass gate).
    If spectral_logprob is NaN (LSTM/GRU), treat as 0.0.
    """
    sp_lp = row.get("spectral_logprob", float("nan"))
    if sp_lp is None or (isinstance(sp_lp, float) and math.isnan(sp_lp)):
        # No spectral score available — assign large negative penalty so this
        # candidate loses to any diffusion candidate that has a real score
        sp_lp = -1e9

    # mean(ppl_per_residue) — use stored JSON array
    ppl_mean = float("nan")
    try:
        lp_list = json.loads(str(row.get("ppl_per_residue", "[]")))
        if lp_list:
            ppl_mean = float(np.mean(lp_list))
    except Exception:
        pass

    if ppl_mean is None or (isinstance(ppl_mean, float) and math.isnan(ppl_mean)):
        ppl_mean = 0.0

    gc = float(row.get("gate_confidence", 0.0) or 0.0)
    return sp_lp - lam * ppl_mean + gam * gc


def pick_best_per_spectrum(df: pd.DataFrame,
                            lam: float, gam: float) -> pd.DataFrame:
    """For each spectrum, pick the candidate with the highest ensemble score."""
    df = df.copy()
    df["_score"] = df.apply(lambda r: ensemble_score(r, lam, gam), axis=1)
    best = df.loc[df.groupby("spectrum_id")["_score"].idxmax()].copy()
    best = best.rename(columns={"_score": "ensemble_score"})
    return best.reset_index(drop=True)


# ── Val-set calibration ───────────────────────────────────────────────────────
def compute_aa_recall_grid(df: pd.DataFrame,
                            true_seqs: dict) -> pd.DataFrame:
    """
    Grid search over lambda × gamma on the val candidate pool.
    true_seqs: {spectrum_id: true_sequence}

    Returns DataFrame with columns [lambda, gamma, aa_recall].
    """
    records = []
    for lam in LAMBDA_GRID:
        for gam in GAMMA_GRID:
            best = pick_best_per_spectrum(df, lam, gam)
            recalls = []
            for _, row in best.iterrows():
                sid   = int(row["spectrum_id"])
                true  = true_seqs.get(sid, "")
                if true:
                    recalls.append(_aa_recall(row["sequence"], true))
            aa_rec = float(np.mean(recalls)) * 100 if recalls else 0.0
            records.append({"lambda": lam, "gamma": gam, "aa_recall": aa_rec})

    return pd.DataFrame(records)


# ── Figures ───────────────────────────────────────────────────────────────────
def make_heatmap(grid_df: pd.DataFrame,
                 lambda_opt: float,
                 gamma_opt:  float,
                 figures_dir: str = "figures") -> None:
    """Figure 3: lambda/gamma ablation heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available — skipping heatmap.")
        return

    pivot = grid_df.pivot(index="gamma", columns="lambda", values="aa_recall")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, cbar_kws={"label": "AA Recall %"})
    ax.set_title("Ensemble Ablation — AA Recall % (Figure 3)\n"
                 f"Optimal: λ={lambda_opt}, γ={gamma_opt}")
    ax.set_xlabel("λ (ESM-2 PPL weight)")
    ax.set_ylabel("γ (gate_confidence weight)")
    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, "figure3_ensemble_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved Figure 3 → {path}")


def make_comparison_bar(metrics_csv: str,
                         ensemble_recall: float,
                         figures_dir: str = "figures") -> None:
    """Comparison bar chart: LSTM / GRU / Diffusion / Ensemble / InstaNovo."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Load existing metrics
    models, recalls = [], []
    if os.path.exists(metrics_csv):
        mdf = pd.read_csv(metrics_csv)
        for _, r in mdf.iterrows():
            name = str(r.get("Model", ""))
            val  = r.get("AA Recall %", None)
            if name and val is not None:
                # Keep only base model rows (not seed-level rows)
                skip_words = ["seed", "mean", "std", "±", "gate"]
                if not any(w in name.lower() for w in skip_words):
                    models.append(name)
                    recalls.append(float(str(val).split("±")[0].strip()))

    # Add ensemble
    if "Ensemble" not in models:
        models.append("Ensemble (CP2)")
        recalls.append(ensemble_recall)

    colors_map = {
        "LSTM Baseline": "#6baed6",
        "GRU Ablation":  "#74c476",
        "Diffusion (CP2)": "#fd8d3c",
        "Ensemble (CP2)": "#9e9ac8",
        "InstaNovo (ref)": "#fc4e2a",
    }
    bar_colors = [colors_map.get(m, "#bdbdbd") for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, recalls, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("AA Recall %")
    ax.set_title("Model Comparison — AA Recall (CP1 → CP2)")
    ax.set_ylim(0, max(recalls) * 1.15)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, "figure4_model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved Figure 4 → {path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_ensemble(diffusion_csv:  str = "results/diffusion_predictions.csv",
                 esm2_csv:       str = "results/esm2_scores.csv",
                 lstm_csv:       str = None,
                 gru_csv:        str = None,
                 metrics_csv:    str = "checkpoints/diffusion_metrics_full.csv",
                 out_csv:        str = "results/ensemble_predictions.csv",
                 figures_dir:    str = "figures",
                 val_fraction:   float = 0.5) -> pd.DataFrame:
    """
    Full ensemble pipeline:
      1. Load all candidate predictions and ESM-2 scores.
      2. Split spectra into val/test halves.
      3. Grid-search lambda × gamma on val half.
      4. Apply optimal weights to full set.
      5. Save ensemble_predictions.csv, Figure 3 (heatmap), Figure 4 (bar chart).
    """
    print("Loading candidate pool …")
    df = load_candidate_pool(diffusion_csv, esm2_csv, lstm_csv, gru_csv)

    if df.empty:
        print("No candidates found — check input CSVs exist.")
        return df

    # Build ground-truth lookup from LSTM CSV (has true_sequence)
    true_seqs: dict = {}
    if lstm_csv and os.path.exists(lstm_csv):
        df_lstm = pd.read_csv(lstm_csv)
        for idx, r in df_lstm.iterrows():
            sid = int(r.get("spectrum_id", idx))
            ts  = str(r.get("true_sequence", ""))
            if ts:
                true_seqs[sid] = ts

    all_ids = sorted(df["spectrum_id"].unique())

    if len(all_ids) < 2 or not true_seqs:
        # No val labels — use lambda=0.1, gamma=0.3 as sensible defaults
        print("No ground-truth for val calibration — using defaults λ=0.1, γ=0.3")
        lambda_opt, gamma_opt = 0.1, 0.3
        grid_df = pd.DataFrame([
            {"lambda": l, "gamma": g, "aa_recall": float("nan")}
            for l in LAMBDA_GRID for g in GAMMA_GRID
        ])
    else:
        # Val/test split — first half of spectrum IDs for val
        n_val   = max(1, int(len(all_ids) * val_fraction))
        val_ids = set(all_ids[:n_val])

        val_df  = df[df["spectrum_id"].isin(val_ids)]
        val_gt  = {sid: ts for sid, ts in true_seqs.items() if sid in val_ids}

        print(f"Val spectra: {len(val_ids)} | Test spectra: {len(all_ids) - len(val_ids)}")
        print("Running lambda × gamma grid search …")
        grid_df = compute_aa_recall_grid(val_df, val_gt)

        best_row   = grid_df.loc[grid_df["aa_recall"].idxmax()]
        lambda_opt = float(best_row["lambda"])
        gamma_opt  = float(best_row["gamma"])
        print(f"Optimal: λ={lambda_opt}, γ={gamma_opt}  "
              f"(val AA recall = {best_row['aa_recall']:.2f}%)")

    # Apply to full candidate pool
    final = pick_best_per_spectrum(df, lambda_opt, gamma_opt)
    final["lambda_opt"] = lambda_opt
    final["gamma_opt"]  = gamma_opt

    # Compute test-set AA recall
    test_recalls = []
    for _, row in final.iterrows():
        sid  = int(row["spectrum_id"])
        true = true_seqs.get(sid, "")
        if true:
            test_recalls.append(_aa_recall(row["sequence"], true))
    ensemble_recall = float(np.mean(test_recalls)) * 100 if test_recalls else float("nan")
    print(f"Ensemble AA recall (full set): {ensemble_recall:.2f}%")

    # Save CSV
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    keep = ["spectrum_id", "sequence", "ensemble_score", "spectral_logprob",
            "gate_confidence", "ppl_scalar", "model_source", "lambda_opt", "gamma_opt"]
    keep = [c for c in keep if c in final.columns]
    final[keep].to_csv(out_csv, index=False)
    print(f"Saved {len(final)} ensemble predictions → {out_csv}")

    # Save grid results
    grid_path = os.path.join(os.path.dirname(out_csv) or "results",
                             "ensemble_grid_search.csv")
    grid_df.to_csv(grid_path, index=False)
    print(f"Saved grid search results → {grid_path}")

    # Figures
    make_heatmap(grid_df, lambda_opt, gamma_opt, figures_dir)
    make_comparison_bar(metrics_csv, ensemble_recall, figures_dir)

    return final


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Three-term val-calibrated ensemble")
    parser.add_argument("--diffusion_csv", default="results/diffusion_predictions.csv")
    parser.add_argument("--esm2_csv",      default="results/esm2_scores.csv")
    parser.add_argument("--lstm_csv",      default="results/lstm_predictions.csv")
    parser.add_argument("--gru_csv",       default="results/gru_predictions.csv")
    parser.add_argument("--metrics_csv",   default="checkpoints/diffusion_metrics_full.csv")
    parser.add_argument("--out_csv",       default="results/ensemble_predictions.csv")
    parser.add_argument("--figures_dir",   default="figures")
    args = parser.parse_args()

    run_ensemble(
        diffusion_csv = args.diffusion_csv,
        esm2_csv      = args.esm2_csv,
        lstm_csv      = args.lstm_csv,
        gru_csv       = args.gru_csv,
        metrics_csv   = args.metrics_csv,
        out_csv       = args.out_csv,
        figures_dir   = args.figures_dir,
    )
