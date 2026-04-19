"""
src/diffusion.py
V-1 (Base):    TransformerDenoiser
V-2 (NOVEL#1): Entropy-adaptive mass gate + gate_confidence export
V-3 (NOVEL#4): Spectral noise augmentation during training
"""
import math
import os
import glob
import sys
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from baseline import Encoder
from preprocessing import (
    load_labeled_spectra, preprocess_spectrum, encode_peptide,
    VOCAB, CHAR_TO_IDX,
)

# ── Constants ──────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 23       # PAD=0, SOS=1, EOS=2, AAs=3..22
SEQ_LEN     = 32       # 30 residues + SOS + EOS
T_STEPS     = 200
BETA_START  = 1e-3
BETA_END    = 2e-2
PROTON_MASS = 1.007276
WATER_MASS  = 18.010565

# Entropy-adaptive gate hyperparameters (NOVEL #1)
GATE_BASE_TOL = 0.02   # Da — tight tolerance when model is confident
GATE_ALPHA    = 2.0    # scales how much entropy relaxes the gate

# Spectral noise augmentation hyperparameters (NOVEL #4)
AUG_PROB      = 0.4    # probability of augmenting a training spectrum
AUG_NOISE_STD = 0.05   # fraction of per-spectrum std to add as noise

# VOCAB = "ACDEFGHIKLMNPQRSTVWY"  →  token[i+3] = VOCAB[i]
_MONO_MASSES = [
    71.03711,   # A  token 3
    103.00919,  # C  token 4
    115.02694,  # D  token 5
    129.04259,  # E  token 6
    147.06841,  # F  token 7
    57.02146,   # G  token 8
    137.05891,  # H  token 9
    113.08406,  # I  token 10
    128.09496,  # K  token 11
    113.08406,  # L  token 12
    131.04049,  # M  token 13
    114.04293,  # N  token 14
    97.05276,   # P  token 15
    128.05858,  # Q  token 16
    156.10111,  # R  token 17
    87.03203,   # S  token 18
    101.04768,  # T  token 19
    99.06841,   # V  token 20
    186.07931,  # W  token 21
    163.06333,  # Y  token 22
]
RESIDUE_MASS = torch.zeros(VOCAB_SIZE)
for _i, _m in enumerate(_MONO_MASSES):
    RESIDUE_MASS[_i + 3] = _m

IDX_TO_CHAR = {i + 3: c for i, c in enumerate(VOCAB)}

# ── Diffusion Schedule ─────────────────────────────────────────────────────────
_betas      = torch.linspace(BETA_START, BETA_END, T_STEPS)
_alpha_bars = torch.cumprod(1.0 - _betas, dim=0)

# Accelerated inference anchors: 20 evenly-spaced steps (avoids error accumulation)
_INFER_STEPS = list(range(T_STEPS - 1, -1, -(T_STEPS // 20)))
if _INFER_STEPS[-1] != 0:
    _INFER_STEPS.append(0)


def set_seed(seed: int):
    """Set all RNG seeds for reproducibility across training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def q_sample(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Forward process: keep x0 token with prob ᾱ_t, else Uniform(VOCAB_SIZE)."""
    abar = _alpha_bars[t.cpu()].to(x0.device).unsqueeze(1)
    keep  = torch.bernoulli(abar.expand_as(x0.float())).bool()
    noise = torch.randint(0, VOCAB_SIZE, x0.shape, device=x0.device)
    return torch.where(keep, x0, noise)


# ── NOVEL #4: Spectral Noise Augmentation ─────────────────────────────────────
def augment_spectrum(spec: torch.Tensor, p: float = AUG_PROB,
                     noise_frac: float = AUG_NOISE_STD) -> torch.Tensor:
    """
    With probability p, add Gaussian noise scaled to noise_frac * per-spectrum std.
    Forces encoder to learn noise-robust features (generalises clean Orbitrap → wastewater).
    spec: (B, 20000)
    """
    if not torch.is_grad_enabled():          # inference — no augmentation
        return spec
    mask = torch.rand(spec.shape[0], device=spec.device) < p  # (B,)
    if not mask.any():
        return spec
    noise_scale = spec.std(dim=1, keepdim=True) * noise_frac  # (B, 1)
    noise       = torch.randn_like(spec) * noise_scale
    augmented   = (spec + noise).clamp(0.0, 1.0)
    return torch.where(mask.unsqueeze(1), augmented, spec)


# ── NOVEL #1: Entropy-Adaptive Mass Gate ──────────────────────────────────────
def entropy_adaptive_gate(logits: torch.Tensor,
                           precursor_masses: torch.Tensor,
                           base_tol: float = GATE_BASE_TOL,
                           alpha: float    = GATE_ALPHA) -> tuple:
    """
    Per-position tolerance = base_tol * (1 + alpha * H_t(i) / log(23)).
    High entropy (uncertain) → relax gate.  Low entropy (confident) → keep tight.

    logits:           (B, L, V)   — modified in-place clone
    precursor_masses: (B,)        — neutral peptide mass in Da
    Returns: (gated_logits, gate_confidence)
        gate_confidence: (B, L) — 1 = tight gate held, 0 = gate relaxed
    """
    B, L, V = logits.shape
    mass_lut = RESIDUE_MASS.to(logits.device)   # (V,)
    out      = logits.clone()

    # Skip gate entirely for spectra with unknown precursor mass
    valid = precursor_masses > 1.0              # (B,)

    # Per-position entropy → per-position tolerance
    with torch.no_grad():
        probs   = F.softmax(logits, dim=-1)                          # (B, L, V)
        entropy = -(probs * (probs + 1e-9).log()).sum(-1)            # (B, L)
        tol     = base_tol * (1 + alpha * entropy / math.log(V))    # (B, L)

        gate_conf = torch.ones(B, L, device=logits.device)          # 1 = tight

        # Best-guess token at each position for the "other positions" mass estimate
        best   = logits.argmax(-1)          # (B, L)
        best_m = mass_lut[best]             # (B, L)
        is_aa  = (best >= 3).float()        # (B, L)

        for pos in range(L):
            # Mass from all OTHER real-AA positions
            other = (best_m * is_aa).sum(1) \
                    - best_m[:, pos] * is_aa[:, pos] \
                    + WATER_MASS                                     # (B,)

            # Candidate total mass for each token at this position
            cand     = other.unsqueeze(1) + mass_lut.unsqueeze(0)   # (B, V)
            pos_tol  = tol[:, pos].unsqueeze(1)                     # (B, 1)
            feasible = (cand - precursor_masses.unsqueeze(1)).abs() < pos_tol

            feasible[:, :3] = True          # PAD/SOS/EOS always pass
            feasible[~valid] = True         # skip gate for unknown-mass spectra

            needs_relax = ~feasible.any(dim=-1)                     # (B,)
            gate_conf[:, pos] = (~needs_relax).float()

            # If even relaxed gate (0.1 Da) zeros everything, leave logits unchanged
            if needs_relax.any():
                relax_feasible = (cand - precursor_masses.unsqueeze(1)).abs() < 0.1
                relax_feasible[:, :3] = True
                relax_feasible[~valid] = True
                final_mask = torch.where(
                    needs_relax.unsqueeze(1), relax_feasible, feasible
                )
            else:
                final_mask = feasible

            out[:, pos].masked_fill_(~final_mask, float('-inf'))

    return out, gate_conf


# ── Model Components ───────────────────────────────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer('freq', freq)
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(1)
        x = t * self.freq.unsqueeze(0)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.proj(x)


class TransformerDenoiser(nn.Module):
    """Predicts clean x_0 logits from noisy x_t, timestep t, spectrum context s."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, nhead=8,
                 dim_ff=512, num_layers=4, seq_len=SEQ_LEN):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(seq_len, d_model)
        self.time_emb  = SinusoidalEmbedding(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, xt: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        B, L = xt.shape
        pos = torch.arange(L, device=xt.device).unsqueeze(0).expand(B, -1)
        x   = self.token_emb(xt) + self.pos_emb(pos)
        x   = x + self.time_emb(t).unsqueeze(1)
        mem = context.unsqueeze(1)
        return self.out(self.transformer(x, mem))   # (B, L, V)


# ── Dataset ────────────────────────────────────────────────────────────────────
class DiffusionDataset(Dataset):
    def __init__(self, X, y, precursor_masses):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.long)
        self.masses = torch.tensor(precursor_masses, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masses[idx]


def build_diffusion_dataset(mzml_path: str, xlsx_path: str, max_spectra: int = 5000):
    """Returns X (N,20000), y (N,32), neutral_masses (N,)."""
    spectra = load_labeled_spectra(mzml_path, xlsx_path, max_spectra)
    X, y, masses = [], [], []
    for s in spectra:
        X.append(preprocess_spectrum(s['mz'], s['intensity']))
        y.append(encode_peptide(s['peptide']))
        prec_mz = s.get('precursor_mz') or 0.0
        charge  = s.get('charge') or 0
        neutral = float(charge) * (float(prec_mz) - PROTON_MASS) if charge else 0.0
        masses.append(neutral)
    return np.array(X), np.array(y), np.array(masses, dtype=np.float32)


# ── Training ───────────────────────────────────────────────────────────────────
def train_diffusion(mzml_paths, xlsx_paths, checkpoint_dir='checkpoints',
                    epochs=50, batch_size=32, lr=1e-3, device=None, seed=42):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    print(f"Device: {device}  |  Seed: {seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    Xs, ys, ms = [], [], []
    for mzml, xlsx in zip(mzml_paths, xlsx_paths):
        X, y, m = build_diffusion_dataset(mzml, xlsx)
        Xs.append(X); ys.append(y); ms.append(m)
    X = np.concatenate(Xs); y = np.concatenate(ys); masses = np.concatenate(ms)
    print(f"Total spectra: {len(X)}")

    N   = len(X)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    n_tr = int(0.70 * N); n_va = int(0.15 * N)
    tr, va = idx[:n_tr], idx[n_tr:n_tr + n_va]
    te = idx[n_tr + n_va:]

    train_dl = DataLoader(DiffusionDataset(X[tr], y[tr], masses[tr]),
                          batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(DiffusionDataset(X[va], y[va], masses[va]),
                          batch_size=batch_size, drop_last=True)

    encoder  = Encoder().to(device)
    denoiser = TransformerDenoiser().to(device)
    params   = list(encoder.parameters()) + list(denoiser.parameters())
    opt      = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        encoder.train(); denoiser.train()
        tr_loss = 0.0
        for spec, seq, _ in train_dl:
            spec, seq = spec.to(device), seq.to(device)
            B = seq.shape[0]

            # NOVEL #4: spectral noise augmentation
            spec = augment_spectrum(spec)

            t      = torch.randint(0, T_STEPS, (B,), device=device)
            xt     = q_sample(seq, t)
            logits = denoiser(xt, t, encoder(spec))
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), seq.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_loss += loss.item() * B

        encoder.eval(); denoiser.eval()
        va_loss = 0.0
        with torch.no_grad():
            for spec, seq, _ in val_dl:
                spec, seq = spec.to(device), seq.to(device)
                t      = torch.randint(0, T_STEPS, (seq.shape[0],), device=device)
                xt     = q_sample(seq, t)
                logits = denoiser(xt, t, encoder(spec))
                va_loss += criterion(logits.reshape(-1, VOCAB_SIZE),
                                     seq.reshape(-1)).item() * spec.shape[0]

        tr_avg = tr_loss / len(train_dl.dataset)
        va_avg = va_loss / len(val_dl.dataset)
        print(f"Epoch {epoch:3d} | train {tr_avg:.4f} | val {va_avg:.4f}")

        if epoch % 10 == 0:
            path = os.path.join(checkpoint_dir, f'diffusion_ckpt_{epoch}.pt')
            torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                        'denoiser': denoiser.state_dict()}, path)
            print(f"  Saved {path}")

        if va_avg < best_val:
            best_val = va_avg
            torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                        'denoiser': denoiser.state_dict()},
                       os.path.join(checkpoint_dir, 'diffusion_best.pt'))

    return encoder, denoiser, (X[te], y[te], masses[te])


# ── Inference ──────────────────────────────────────────────────────────────────
def decode_tokens(tokens) -> str:
    result = []
    for tok in tokens:
        if int(tok) == 2:
            break
        if int(tok) >= 3:
            result.append(IDX_TO_CHAR.get(int(tok), '?'))
    return ''.join(result)


@torch.no_grad()
def generate_sequences(encoder, denoiser, spectra, precursor_masses,
                        n_candidates: int = 5, T_sample: float = 0.8,
                        t_infer: int = 100, device=None, use_gate: bool = True):
    """
    One-shot inference: run denoiser once at t=t_infer with random noise,
    apply entropy-adaptive gate (NOVEL #1), argmax → sequence.
    Repeated n_candidates times with different noise draws.

    Iterative chains collapsed for this model because re-noising injected
    EOS/PAD tokens that the model locked in at t→0. One-shot at t=100
    consistently produces non-empty, meaningful sequences.

    Returns:
        sequences:    list[N] of list[n_candidates] strings
        spectral_lps: list[N] of list[n_candidates] floats  (log-prob at t=0)
        gate_confs:   list[N] of list[n_candidates] floats  (mean gate_confidence)
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval(); denoiser.eval()

    spec_t  = torch.tensor(spectra, dtype=torch.float32, device=device)
    mass_t  = torch.tensor(precursor_masses, dtype=torch.float32, device=device)
    context = encoder(spec_t)
    N       = len(spectra)

    sequences    = [[] for _ in range(N)]
    spectral_lps = [[] for _ in range(N)]
    gate_confs   = [[] for _ in range(N)]

    t_vec  = torch.full((N,), t_infer, dtype=torch.long, device=device)
    t_zero = torch.zeros(N, dtype=torch.long, device=device)

    for _ in range(n_candidates):
        # Random noise input — different each candidate draw
        xt = torch.randint(0, VOCAB_SIZE, (N, SEQ_LEN), device=device)

        # Predict x0 — NO gate here: at t=100 the argmax has arbitrary total
        # mass, so the coordinate-wise gate would zero out all amino acids.
        x0_hat = denoiser(xt, t_vec, context).argmax(-1)   # (N, L)

        # NOVEL #1: entropy-adaptive gate applied at t=0 where x0_hat is a
        # real peptide sequence with a well-defined total mass.
        # gate_confidence = fraction of positions where tight 0.02 Da gate held.
        logits_0 = denoiser(x0_hat, t_zero, context)                 # (N, L, V)
        if use_gate:
            logits_0_gated, gc = entropy_adaptive_gate(logits_0, mass_t)
        else:
            logits_0_gated = logits_0
            gc = torch.ones(N, SEQ_LEN, device=device)  # ablation: gate disabled

        # Spectral log-prob over gated logits (used by ensemble.py)
        log_fin = F.log_softmax(logits_0_gated, dim=-1)
        sp_lp   = log_fin.gather(-1, x0_hat.unsqueeze(-1)).squeeze(-1)
        aa_mask = (x0_hat >= 3).float()
        sp_lp   = (sp_lp * aa_mask).sum(-1) / aa_mask.sum(-1).clamp(min=1)

        for i, seq in enumerate(x0_hat.cpu().numpy()):
            sequences[i].append(decode_tokens(seq))
            spectral_lps[i].append(float(sp_lp[i].cpu()))
            gate_confs[i].append(float(gc[i].mean().cpu()))

    return sequences, spectral_lps, gate_confs


# ── Save Predictions CSV (deliverable) ────────────────────────────────────────
def save_predictions(sequences, spectral_lps, gate_confs,
                     out_path='results/diffusion_predictions.csv'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for i, (seqs, lps, gcs) in enumerate(zip(sequences, spectral_lps, gate_confs)):
        for seq, lp, gc in zip(seqs, lps, gcs):
            rows.append({'spectrum_id': i, 'sequence': seq,
                         'spectral_logprob': lp, 'gate_confidence': gc})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} predictions → {out_path}")
    return df


# ── Evaluation ─────────────────────────────────────────────────────────────────
def aa_recall(pred: str, true: str) -> float:
    from collections import Counter
    p, t = Counter(pred), Counter(true)
    return sum((p & t).values()) / max(len(true), 1)


def evaluate_aa_recall(encoder, denoiser, X_test, y_test, masses_test,
                        batch_size=32, results_dir='results', device=None,
                        use_gate=True):
    if device is None:
        device = next(encoder.parameters()).device

    all_seqs, all_lps, all_gcs = [], [], []
    recalls, pep_correct = [], []

    for i in range(0, len(X_test), batch_size):
        bs  = X_test[i:i+batch_size]
        bm  = masses_test[i:i+batch_size]
        byt = y_test[i:i+batch_size]
        seqs, lps, gcs = generate_sequences(encoder, denoiser, bs, bm,
                                             n_candidates=1, device=device,
                                             use_gate=use_gate)
        for pred_list, lp_list, gc_list, true_tok in zip(seqs, lps, gcs, byt):
            all_seqs.append(pred_list); all_lps.append(lp_list)
            all_gcs.append(gc_list)
            recalls.append(aa_recall(pred_list[0], decode_tokens(true_tok)))
            pep_correct.append(pred_list[0] == decode_tokens(true_tok))

    aa_rec  = float(np.mean(recalls)) * 100
    pep_acc = float(np.mean(pep_correct)) * 100
    print(f"AA Recall  : {aa_rec:.2f}%")
    print(f"Peptide Acc: {pep_acc:.2f}%")

    os.makedirs(results_dir, exist_ok=True)
    save_predictions(all_seqs, all_lps, all_gcs,
                     os.path.join(results_dir, 'diffusion_predictions.csv'))
    return aa_rec, pep_acc


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────
def load_checkpoint(path: str, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder  = Encoder().to(device)
    denoiser = TransformerDenoiser().to(device)
    encoder.load_state_dict(ckpt['encoder'])
    denoiser.load_state_dict(ckpt['denoiser'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return encoder, denoiser


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(__file__), '..', 'Data', 'E coli EV proteomics')
    mzml_paths = sorted(glob.glob(os.path.join(BASE, '*.mzML')))
    xlsx_paths = sorted(glob.glob(os.path.join(BASE, 'Database search output*.xlsx')))
    if not mzml_paths:
        raise FileNotFoundError(f"No mzML files in {BASE}")
    encoder, denoiser, (X_te, y_te, m_te) = train_diffusion(
        mzml_paths, xlsx_paths, checkpoint_dir='checkpoints', epochs=50)
    evaluate_aa_recall(encoder, denoiser, X_te, y_te, m_te)
