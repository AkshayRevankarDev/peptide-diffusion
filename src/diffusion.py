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
VOCAB_SIZE  = 24       # PAD=0, SOS=1, EOS=2, AAs=3..22, MASK=23
MASK_TOK    = 23       # absorbing state — never predicted, only used in q_sample
SEQ_LEN     = 32       # 30 residues + SOS + EOS
T_STEPS     = 200
BETA_START  = 1e-3
BETA_END    = 2e-2
PROTON_MASS = 1.007276
WATER_MASS  = 18.010565
N_PEAKS     = 200      # top-K peaks kept per spectrum
MAX_MZ      = 2000.0   # m/z normalisation ceiling
BY_SIGMA    = 0.1      # Da — Gaussian half-width for b/y ion proximity bias

# Entropy-adaptive gate hyperparameters (NOVEL #1)
GATE_BASE_TOL = 0.02   # Da — tight tolerance when model is confident
GATE_ALPHA    = 2.0    # scales how much entropy relaxes the gate

# Spectral noise augmentation hyperparameters (NOVEL #4)
AUG_PROB      = 0.4    # probability of augmenting a training spectrum
AUG_NOISE_STD = 0.05   # fraction of per-spectrum std to add as noise

# Mass consistency loss weight (Option A fix)
# Uses relative L1 (|pred-true|/true), so values are in [0,1] range.
# 0.1 weight means mass term contributes ~10% as much as CE when error is ~10%.
MASS_LOSS_WEIGHT = 0.1

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


def extract_top_peaks(mz_arr, int_arr, n_peaks: int = N_PEAKS,
                      max_mz: float = MAX_MZ) -> np.ndarray:
    """Return (n_peaks, 2) array of top-K peaks sorted by m/z, normalised.
    Column 0: m/z / max_mz ∈ [0, 1].  Column 1: intensity / max_intensity ∈ [0, 1].
    Rows beyond the actual peak count are zero-padded."""
    mz  = np.asarray(mz_arr,  dtype=np.float32)
    ity = np.asarray(int_arr, dtype=np.float32)
    valid = (mz > 50) & (mz < max_mz)
    mz, ity = mz[valid], ity[valid]
    if len(ity) == 0:
        return np.zeros((n_peaks, 2), dtype=np.float32)
    ity = ity / (ity.max() + 1e-9)
    if len(ity) > n_peaks:
        top = np.argpartition(ity, -n_peaks)[-n_peaks:]
    else:
        top = np.arange(len(ity))
    mz_s, ity_s = mz[top], ity[top]
    order = np.argsort(mz_s)
    mz_s, ity_s = mz_s[order], ity_s[order]
    out = np.zeros((n_peaks, 2), dtype=np.float32)
    n = len(mz_s)
    out[:n, 0] = mz_s / max_mz
    out[:n, 1] = ity_s
    return out


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
    """Forward process: keep x0 token with prob ᾱ_t, else absorb to MASK_TOK.
    Absorbing diffusion aligns with mask-predict decoding: re-corrupting uncertain
    positions at inference with MASK_TOK now matches the training distribution."""
    abar = _alpha_bars[t.cpu()].to(x0.device).unsqueeze(1)
    keep = torch.bernoulli(abar.expand_as(x0.float())).bool()
    mask = torch.full_like(x0, MASK_TOK)
    return torch.where(keep, x0, mask)


# ── Peak-Level Augmentation ────────────────────────────────────────────────────
def augment_peaks(peaks: torch.Tensor, p: float = AUG_PROB,
                  noise_frac: float = AUG_NOISE_STD) -> torch.Tensor:
    """
    Peak-level augmentation for the PeakEncoder (replaces old augment_spectrum).
    With probability p per spectrum:
      - Add Gaussian noise to intensities (noise_frac fraction of max intensity).
      - Randomly zero out 10% of peaks (simulates missing ions).
    peaks: (B, K, 2) — column 0 = m/z/MAX_MZ, column 1 = intensity.
    Augmentation is skipped at inference (torch.is_grad_enabled() == False).
    """
    if not torch.is_grad_enabled():
        return peaks
    B, K, _ = peaks.shape
    aug_mask = torch.rand(B, device=peaks.device) < p  # (B,) spectra to augment
    if not aug_mask.any():
        return peaks

    out = peaks.clone()
    # Intensity noise
    int_noise = torch.randn(B, K, device=peaks.device) * noise_frac
    out[:, :, 1] = (out[:, :, 1] + int_noise).clamp(0.0, 1.0)
    # Random peak dropout (10%)
    drop = torch.rand(B, K, device=peaks.device) < 0.10
    out[:, :, 0] = out[:, :, 0].masked_fill(drop, 0.0)
    out[:, :, 1] = out[:, :, 1].masked_fill(drop, 0.0)

    aug_mask_2d = aug_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
    return torch.where(aug_mask_2d, out, peaks)


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


class PeakEncoder(nn.Module):
    """
    Transformer encoder over the top-K (m/z, intensity) peak pairs.
    Replaces the binned-spectrum CNN encoder with direct peak-level attention,
    which is how InstaNovo and all top de-novo models encode spectra.
    Prepends a precursor-mass token so the model always knows the target mass.
    """
    def __init__(self, d_model: int = 512, n_peaks: int = N_PEAKS,
                 nhead: int = 8, n_layers: int = 6):
        super().__init__()
        self.mz_emb   = SinusoidalEmbedding(d_model)   # embed m/z as float
        self.int_proj  = nn.Linear(1, d_model)
        self.mass_emb  = SinusoidalEmbedding(d_model)  # embed precursor mass
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, peaks: torch.Tensor,
                masses: torch.Tensor) -> tuple:
        """
        peaks:  (B, K, 2) — (mz/MAX_MZ, intensity)
        masses: (B,)      — neutral precursor mass in Da
        Returns ((B, K+1, d_model), (B, K+1) bool pad mask)
        The pad mask is True for positions that should be ignored (padded peaks).
        """
        B, K, _ = peaks.shape
        mz_scaled = peaks[:, :, 0].reshape(-1) * MAX_MZ        # (B*K,)
        mz_tok    = self.mz_emb(mz_scaled).reshape(B, K, -1)   # (B, K, d)
        int_tok   = self.int_proj(peaks[:, :, 1:])              # (B, K, d)
        peak_tok  = mz_tok + int_tok                            # (B, K, d)
        mass_tok  = self.mass_emb(masses).unsqueeze(1)          # (B, 1, d)
        x = torch.cat([mass_tok, peak_tok], dim=1)              # (B, K+1, d)
        # Mask zero-padded peaks (m/z == 0 means no peak)
        pad = torch.zeros(B, K + 1, dtype=torch.bool, device=peaks.device)
        pad[:, 1:] = (peaks[:, :, 0] == 0)
        out = self.encoder(x, src_key_padding_mask=pad)         # (B, K+1, d)
        return out, pad


def compute_by_pair_bias(xt: torch.Tensor, peaks: torch.Tensor,
                          precursor_masses: torch.Tensor,
                          sigma: float = BY_SIGMA) -> torch.Tensor:
    """
    AlphaFold3-inspired B/Y ion pair bias.
    For each (sequence position i, peak j) pair, compute the Gaussian proximity
    of the theoretical b/y ion at position i to peak j's m/z, weighted by
    peak intensity.  Committed (non-MASK) positions contribute signal;
    masked positions contribute zero — so the model learns to use committed
    positions to guide prediction of uncertain ones.

    Returns (B, L, K+1) — first column is the mass-token (always 0 bias).
    """
    B, L = xt.shape
    K    = peaks.shape[1]
    device = xt.device

    mass_lut = RESIDUE_MASS.to(device)          # (V,)
    # Token masses; MASK and special tokens → 0
    is_aa    = ((xt >= 3) & (xt < MASK_TOK)).float()           # (B, L)
    tok_mass = mass_lut[xt.clamp(0, VOCAB_SIZE - 1)] * is_aa   # (B, L)

    # Cumulative residue mass from position 0 → b-ion series
    cum_mass = torch.cumsum(tok_mass, dim=1)                    # (B, L)
    b_ions   = cum_mass + PROTON_MASS                           # (B, L)

    # y-ion series: precursor_mass − cum_mass (approximate)
    pm     = precursor_masses.unsqueeze(1)                      # (B, 1)
    valid  = (pm > 1.0).float()                                 # skip unknown mass
    y_ions = (pm - cum_mass + WATER_MASS + PROTON_MASS) * valid # (B, L)

    # Observed peak m/z, unnormalised
    mz_obs  = peaks[:, :, 0] * MAX_MZ  # (B, K)
    int_obs = peaks[:, :, 1]            # (B, K)

    # Pairwise Gaussian proximity: (B, L, K)
    b_diff  = b_ions.unsqueeze(2) - mz_obs.unsqueeze(1)        # (B, L, K)
    y_diff  = y_ions.unsqueeze(2) - mz_obs.unsqueeze(1)        # (B, L, K)
    prox    = (torch.exp(-b_diff**2 / (2 * sigma**2)) +
               torch.exp(-y_diff**2 / (2 * sigma**2)))          # (B, L, K)
    prox    = prox * int_obs.unsqueeze(1)                       # weight by intensity
    prox    = prox * is_aa.unsqueeze(2)                         # zero out masked pos

    # Prepend zero column for the mass token: (B, L, K+1)
    return F.pad(prox, (1, 0))


class BYBiasedDecoderLayer(nn.Module):
    """
    Pre-norm Transformer decoder layer with B/Y ion pair bias injected
    into the cross-attention logits (AlphaFold3-style pair bias).
    A learnable scalar `bias_scale` (initialised to 0) lets the model
    start as a standard decoder and gradually lean on the ion signal.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff         = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.nhead      = nhead
        # Starts at 0 so model is identical to unbiased decoder at init
        self.bias_scale = nn.Parameter(torch.zeros(1))

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                by_bias: torch.Tensor | None = None,
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = tgt.shape
        # Self-attention (pre-norm)
        h = self.norm1(tgt)
        h, _ = self.self_attn(h, h, h)
        tgt = tgt + h
        # Cross-attention with optional B/Y pair bias and padding mask
        h = self.norm2(tgt)
        if by_bias is not None:
            # by_bias: (B, L, K+1) → (B*nhead, L, K+1)
            bias = (by_bias.unsqueeze(1)
                          .expand(-1, self.nhead, -1, -1)
                          .reshape(B * self.nhead, L, -1))
            bias = self.bias_scale * bias
            h, _ = self.cross_attn(h, memory, memory, attn_mask=bias,
                                   key_padding_mask=memory_key_padding_mask)
        else:
            h, _ = self.cross_attn(h, memory, memory,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + h
        # Feed-forward (pre-norm)
        h = self.norm3(tgt)
        tgt = tgt + self.ff(h)
        return tgt


class TransformerDenoiser(nn.Module):
    """
    Bidirectional denoiser with peak-level cross-attention.
    memory: (B, K+1, d) sequence of peak embeddings from PeakEncoder.
    B/Y ion pair bias (AF3-inspired) is injected into every cross-attention layer.
    """
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 512,
                 nhead: int = 8, dim_ff: int = 2048,
                 num_layers: int = 6, seq_len: int = SEQ_LEN):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(seq_len, d_model)
        self.time_emb  = SinusoidalEmbedding(d_model)
        self.layers    = nn.ModuleList([
            BYBiasedDecoderLayer(d_model, nhead, dim_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Linear(d_model, vocab_size)

    def forward(self, xt: torch.Tensor, t: torch.Tensor,
                memory: torch.Tensor,
                peaks: torch.Tensor | None = None,
                precursor_masses: torch.Tensor | None = None,
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L = xt.shape
        pos  = torch.arange(L, device=xt.device).unsqueeze(0).expand(B, -1)
        x    = self.token_emb(xt) + self.pos_emb(pos)
        x    = x + self.time_emb(t).unsqueeze(1)
        by_bias = (compute_by_pair_bias(xt, peaks, precursor_masses)
                   if peaks is not None and precursor_masses is not None
                   else None)
        for layer in self.layers:
            x = layer(x, memory, by_bias=by_bias,
                      memory_key_padding_mask=memory_key_padding_mask)
        return self.out(self.norm(x))   # (B, L, V)


# ── Dataset ────────────────────────────────────────────────────────────────────
class DiffusionDataset(Dataset):
    """Stores (peaks, sequence, precursor_mass) triples for peak-level training."""
    def __init__(self, peaks, y, precursor_masses):
        self.peaks  = torch.tensor(peaks,  dtype=torch.float32)  # (N, K, 2)
        self.y      = torch.tensor(y,      dtype=torch.long)
        self.masses = torch.tensor(precursor_masses, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.peaks[idx], self.y[idx], self.masses[idx]


def build_diffusion_dataset(mzml_path: str, xlsx_path: str,
                            max_spectra: int = 5000,
                            n_peaks: int = N_PEAKS,
                            return_raw: bool = False):
    """Returns peaks (N, K, 2), y (N, 32), neutral_masses (N,).
    If return_raw=True also returns raw_peaks: list of (mz_arr, int_arr) for SGIR."""
    spectra = load_labeled_spectra(mzml_path, xlsx_path, max_spectra)
    peaks_list, y, masses, raw_peaks = [], [], [], []
    for s in spectra:
        peaks_list.append(extract_top_peaks(s['mz'], s['intensity'], n_peaks))
        y.append(encode_peptide(s['peptide']))
        prec_mz = s.get('precursor_mz') or 0.0
        charge  = s.get('charge') or 0
        neutral = float(charge) * (float(prec_mz) - PROTON_MASS) if charge else 0.0
        masses.append(neutral)
        if return_raw:
            raw_peaks.append((np.asarray(s['mz'], dtype=np.float32),
                              np.asarray(s['intensity'], dtype=np.float32)))
    out = (np.array(peaks_list, dtype=np.float32),
           np.array(y),
           np.array(masses, dtype=np.float32))
    return out + (raw_peaks,) if return_raw else out


# ── Training ───────────────────────────────────────────────────────────────────
def train_diffusion(mzml_paths, xlsx_paths, checkpoint_dir='checkpoints',
                    epochs=50, batch_size=32, lr=1e-3, device=None, seed=42):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    print(f"Device: {device}  |  Seed: {seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_peaks, ys, ms = [], [], []
    for mzml, xlsx in zip(mzml_paths, xlsx_paths):
        pk, y, m = build_diffusion_dataset(mzml, xlsx)
        all_peaks.append(pk); ys.append(y); ms.append(m)
    peaks  = np.concatenate(all_peaks)
    y      = np.concatenate(ys)
    masses = np.concatenate(ms)
    print(f"Total spectra: {len(peaks)}")

    N   = len(peaks)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    n_tr = int(0.70 * N); n_va = int(0.15 * N)
    tr, va = idx[:n_tr], idx[n_tr:n_tr + n_va]
    te = idx[n_tr + n_va:]

    train_dl = DataLoader(DiffusionDataset(peaks[tr], y[tr], masses[tr]),
                          batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(DiffusionDataset(peaks[va], y[va], masses[va]),
                          batch_size=batch_size, drop_last=True)

    encoder  = PeakEncoder().to(device)
    denoiser = TransformerDenoiser().to(device)
    params   = list(encoder.parameters()) + list(denoiser.parameters())
    opt        = optim.AdamW(params, lr=lr, weight_decay=1e-2)
    warmup_ep  = max(5, epochs // 20)   # 5% warmup
    def _lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        t = (ep - warmup_ep) / max(epochs - warmup_ep, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t)) * (1 - 1e-5 / lr) + 1e-5 / lr
    scheduler = optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        encoder.train(); denoiser.train()
        tr_loss = 0.0
        for pks, seq, mass in train_dl:
            pks, seq, mass = pks.to(device), seq.to(device), mass.to(device)
            B = seq.shape[0]

            t      = torch.randint(0, T_STEPS, (B,), device=device)
            xt     = q_sample(seq, t)
            pks_aug = augment_peaks(pks)
            memory, pad_mask = encoder(pks_aug, mass)
            logits = denoiser(xt, t, memory, peaks=pks_aug, precursor_masses=mass,
                              memory_key_padding_mask=pad_mask)
            ce_loss = criterion(logits.reshape(-1, VOCAB_SIZE), seq.reshape(-1))

            # Mass consistency loss (relative L1 on expected sequence mass)
            valid_m = mass > 1.0
            if valid_m.any():
                mass_lut   = RESIDUE_MASS.to(device)
                probs      = F.softmax(logits, dim=-1)
                exp_mass   = (probs * mass_lut).sum(-1)
                is_aa_soft = 1.0 - probs[:, :, :3].sum(-1)
                pred_mass  = (exp_mass * is_aa_soft).sum(-1) + WATER_MASS
                mass_loss  = ((pred_mass[valid_m] - mass[valid_m]).abs()
                              / mass[valid_m]).mean()
                loss = ce_loss + MASS_LOSS_WEIGHT * mass_loss
            else:
                loss = ce_loss

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_loss += ce_loss.item() * B

        encoder.eval(); denoiser.eval()
        va_loss = 0.0
        with torch.no_grad():
            for pks, seq, mass in val_dl:
                pks, seq, mass = pks.to(device), seq.to(device), mass.to(device)
                t      = torch.randint(0, T_STEPS, (seq.shape[0],), device=device)
                xt     = q_sample(seq, t)
                memory, pad_mask = encoder(pks, mass)
                logits = denoiser(xt, t, memory, peaks=pks, precursor_masses=mass,
                                  memory_key_padding_mask=pad_mask)
                va_loss += criterion(logits.reshape(-1, VOCAB_SIZE),
                                     seq.reshape(-1)).item() * pks.shape[0]

        tr_avg = tr_loss / len(train_dl.dataset)
        va_avg = va_loss / len(val_dl.dataset)
        print(f"Epoch {epoch:3d} | train {tr_avg:.4f} | val {va_avg:.4f}")

        if epoch % 10 == 0:
            path = os.path.join(checkpoint_dir, f'diffusion_ckpt_{epoch}.pt')
            torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                        'denoiser': denoiser.state_dict()}, path)
            print(f"  Saved {path}")

        scheduler.step()

        if va_avg < best_val:
            best_val = va_avg
            torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                        'denoiser': denoiser.state_dict()},
                       os.path.join(checkpoint_dir, 'diffusion_best.pt'))

    # Always save the last epoch — AA recall peaks here, not at best val CE loss
    torch.save({'epoch': epochs, 'encoder': encoder.state_dict(),
                'denoiser': denoiser.state_dict()},
               os.path.join(checkpoint_dir, 'diffusion_final.pt'))
    print(f"  Saved diffusion_final.pt (epoch {epochs})")

    return encoder, denoiser, (peaks[te], y[te], masses[te])


# ── Inference ──────────────────────────────────────────────────────────────────
def _seq_mass(seq: str) -> float:
    """Monoisotopic neutral mass of a peptide string (residues + H2O)."""
    return sum(_MONO_MASSES[VOCAB.index(c)] for c in seq if c in VOCAB) + WATER_MASS


def mass_correct_sequence(seq: str, precursor_mass: float,
                           tol: float = 0.05) -> str:
    """
    Option B: post-hoc mass correction.
    If the predicted sequence mass is outside tol Da of precursor_mass,
    try all single-position amino acid swaps and keep the one that minimises
    the mass error. Returns seq unchanged when precursor_mass is unknown (≤1 Da)
    or already within tolerance.
    """
    if precursor_mass <= 1.0 or not seq:
        return seq
    current_delta = abs(_seq_mass(seq) - precursor_mass)
    if current_delta <= tol:
        return seq

    best_seq, best_delta = seq, current_delta
    for pos in range(len(seq)):
        for aa in VOCAB:
            if aa == seq[pos]:
                continue
            candidate = seq[:pos] + aa + seq[pos + 1:]
            delta = abs(_seq_mass(candidate) - precursor_mass)
            if delta < best_delta:
                best_delta, best_seq = delta, candidate

    # Only apply if the correction actually achieves tolerance — a swap that
    # merely reduces the error without reaching tolerance swaps a correct AA
    # for a wrong one more often than not.
    return best_seq if best_delta <= tol else seq


# ── NOVEL #8: Mass-Constrained Beam Search ────────────────────────────────────
_MIN_RESIDUE_MASS = min(_MONO_MASSES)   # G = 57.02 Da
_MAX_RESIDUE_MASS = max(_MONO_MASSES)   # W = 186.08 Da

def mass_constrained_beam_search(logits_t0: torch.Tensor,
                                  precursor_masses: torch.Tensor,
                                  beam_width: int = 20,
                                  tol: float = 0.1) -> list:
    """
    NOVEL #8: replace independent argmax with left-to-right beam search.
    At each position, expand beams using top-k token log-probs and prune
    branches whose remaining mass budget is infeasible given precursor_mass.

    logits_t0:        (B, L, V)
    precursor_masses: (B,)
    Returns: list[B] of str — best sequence per spectrum.
    """
    B, L, V = logits_t0.shape
    log_probs = F.log_softmax(logits_t0, dim=-1)  # (B, L, V)
    mass_lut  = RESIDUE_MASS.to(logits_t0.device) # (V,)

    results = []
    for b in range(B):
        pm = float(precursor_masses[b].cpu())
        lp = log_probs[b]          # (L, V)

        # beam: list of (score, [token_ids], accumulated_aa_mass)
        beams = [(0.0, [], 0.0)]

        for pos in range(L):
            top_scores, top_toks = lp[pos].topk(beam_width)
            new_beams = []
            for score, toks, acc_mass in beams:
                for s, tok in zip(top_scores.tolist(), top_toks.tolist()):
                    new_score = score + s
                    tok_mass  = float(mass_lut[tok].cpu())
                    new_mass  = acc_mass + tok_mass if tok >= 3 else acc_mass

                    # Remaining positions: [pos+1 .. L-1]
                    remaining = L - pos - 1
                    # Feasibility: can any filling of remaining positions hit pm?
                    lo = new_mass + WATER_MASS + remaining * _MIN_RESIDUE_MASS
                    hi = new_mass + WATER_MASS + remaining * _MAX_RESIDUE_MASS

                    # If tok is EOS, we're done — check mass now
                    if tok == 2:
                        delta = abs(new_mass + WATER_MASS - pm)
                        if pm <= 1.0 or delta <= tol * 3:
                            new_beams.append((new_score, toks + [tok], new_mass))
                        continue

                    # Prune if no remaining filling can reach pm (unless mass unknown)
                    if pm > 1.0 and (lo > pm + tol or hi < pm - tol):
                        continue

                    new_beams.append((new_score, toks + [tok], new_mass))

            if not new_beams:
                # Fallback: keep existing beams without pruning
                new_beams = [(sc + float(lp[pos].max().cpu()),
                              tk + [int(lp[pos].argmax().cpu())], am)
                             for sc, tk, am in beams]

            # Keep top beam_width by score
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

        best_toks = beams[0][1]
        results.append(best_toks)
    return results


# ── NOVEL #11: Mask-Predict Iterative Decoding ────────────────────────────────
@torch.no_grad()
def mask_predict_decode(logits_t0: torch.Tensor,
                        context: torch.Tensor,
                        denoiser: nn.Module,
                        n_iter: int = 4,
                        peaks: torch.Tensor | None = None,
                        precursor_masses: torch.Tensor | None = None,
                        memory_key_padding_mask: torch.Tensor | None = None) -> list:
    """
    NOVEL #11: Mask-Predict iterative decoding for the bidirectional denoiser.
    Correct counterpart to left-to-right beam search: keeps highest-confidence
    positions fixed and re-corrupts the uncertain ones, letting the bidirectional
    model refine them conditioned on the already-committed positions.
    Grounded in Ghazvininejad et al. 2019 (Mask-Predict) and LLaDA 2025.
    No training changes needed — works on existing checkpoints.
    """
    B, L, V = logits_t0.shape
    device  = logits_t0.device

    probs = F.softmax(logits_t0, dim=-1)
    conf  = probs.max(dim=-1).values     # (B, L)
    x0    = logits_t0.argmax(dim=-1)     # (B, L)

    t_zero = torch.zeros(B, dtype=torch.long, device=device)
    # Linear schedule: commit progressively more positions each round
    n_uncertain_schedule = [
        int(L * (1.0 - (i + 1) / n_iter)) for i in range(n_iter)
    ]  # e.g. n_iter=4, L=32 → [24, 16, 8, 0]

    for step in range(n_iter):
        n_uncertain = n_uncertain_schedule[step]
        if n_uncertain == 0:
            break

        # Mask least-confident positions
        rank = conf.argsort(dim=-1, descending=False)   # lowest conf first
        uncertain_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        uncertain_mask.scatter_(1, rank[:, :n_uncertain], True)

        # Corrupt uncertain positions with MASK_TOK — matches absorbing q_sample
        mask_fill = torch.full((B, L), MASK_TOK, dtype=torch.long, device=device)
        xt_new    = torch.where(uncertain_mask, mask_fill, x0)

        # Re-denoise at t=0 — bidirectional model conditions on fixed positions
        logits_new = denoiser(xt_new, t_zero, context,
                              peaks=peaks, precursor_masses=precursor_masses,
                              memory_key_padding_mask=memory_key_padding_mask)
        logits_new[..., MASK_TOK] = float('-inf')   # never output MASK
        probs_new  = F.softmax(logits_new, dim=-1)

        # Update only uncertain positions
        x0   = torch.where(uncertain_mask, logits_new.argmax(dim=-1), x0)
        conf = torch.where(uncertain_mask, probs_new.max(dim=-1).values, conf)

    return [decode_tokens(x0[b].cpu().tolist()) for b in range(B)]


# ── NOVEL #9: Spectrally-Grounded Iterative Refinement ────────────────────────
def _compute_by_ions(seq: str) -> tuple:
    """Compute theoretical b and y ion m/z values (singly charged, +1 proton)."""
    masses = [_MONO_MASSES[VOCAB.index(c)] for c in seq if c in VOCAB]
    if not masses:
        return np.array([]), np.array([])
    b_ions = np.cumsum(masses) + PROTON_MASS
    y_ions = np.cumsum(masses[::-1]) + WATER_MASS + PROTON_MASS
    return b_ions, y_ions


def _peak_support(theoretical_mz: np.ndarray, obs_mz: np.ndarray,
                  obs_int: np.ndarray, tol: float = 0.02) -> float:
    """Intensity-weighted fraction of theoretical ions matched in observed spectrum."""
    if len(theoretical_mz) == 0 or len(obs_mz) == 0:
        return 0.0
    total = 0.0
    for tmz in theoretical_mz:
        delta = np.abs(obs_mz - tmz)
        idx = delta.argmin()
        if delta[idx] <= tol:
            total += float(obs_int[idx])
    denom = obs_int.sum()
    return total / denom if denom > 0 else 0.0


def spectrally_grounded_refine(seq: str, obs_mz: np.ndarray, obs_int: np.ndarray,
                                precursor_mass: float, t0_log_probs: torch.Tensor,
                                max_iter: int = 5, ion_tol: float = 0.02) -> str:
    """
    NOVEL #9: Spectrally-Grounded Iterative Refinement (SGIR).
    For each iteration:
      1. Compute b/y ion support per position.
      2. Find weakest-supported position.
      3. Try all mass-feasible AA substitutions there; pick the one maximising
         support + t=0 log-prob. Stop when no improvement.

    t0_log_probs: (L, V) log-softmax from denoiser at t=0 for this spectrum.
    """
    if not seq or precursor_mass <= 1.0 or len(obs_mz) == 0:
        return seq

    # Normalise observed intensities
    obs_int = obs_int / (obs_int.max() + 1e-9)

    for _ in range(max_iter):
        b_ions, y_ions = _compute_by_ions(seq)
        if len(b_ions) == 0:
            break

        # Per-position support: average of b[i] and y[len-1-i] support
        n = len(seq)
        support = np.zeros(n)
        for i in range(n):
            bi = b_ions[i:i+1] if i < len(b_ions) else np.array([])
            yi = y_ions[n-1-i:n-i] if (n-1-i) < len(y_ions) else np.array([])
            ions = np.concatenate([bi, yi]) if len(bi) or len(yi) else np.array([])
            support[i] = _peak_support(ions, obs_mz, obs_int, ion_tol)

        worst_pos = int(np.argmin(support))
        best_seq, best_score = seq, support[worst_pos]

        for aa in VOCAB:
            if aa == seq[worst_pos]:
                continue
            candidate = seq[:worst_pos] + aa + seq[worst_pos + 1:]
            # Mass feasibility
            if abs(_seq_mass(candidate) - precursor_mass) > 0.1:
                continue
            # New support at worst_pos
            b_c, y_c = _compute_by_ions(candidate)
            if len(b_c) == 0:
                continue
            bi = b_c[worst_pos:worst_pos+1] if worst_pos < len(b_c) else np.array([])
            yi = y_c[n-1-worst_pos:n-worst_pos] if (n-1-worst_pos) < len(y_c) else np.array([])
            ions = np.concatenate([bi, yi]) if len(bi) or len(yi) else np.array([])
            new_support = _peak_support(ions, obs_mz, obs_int, ion_tol)
            # Tiebreak with t=0 log-prob
            tok = CHAR_TO_IDX.get(aa, 0)
            lp_bonus = float(t0_log_probs[worst_pos, tok].cpu()) if tok < t0_log_probs.shape[1] else 0.0
            score = new_support + 0.05 * lp_bonus
            if score > best_score:
                best_score, best_seq = score, candidate

        if best_seq == seq:
            break
        seq = best_seq

    return seq


# ── NOVEL #10: ESM-2 + Spectral Posterior Reranking ──────────────────────────
_esm2_model  = None
_esm2_tokenizer = None

def _get_esm2():
    global _esm2_model, _esm2_tokenizer
    if _esm2_model is None:
        try:
            from transformers import EsmModel, EsmTokenizer
            _esm2_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
            _esm2_model     = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
            _esm2_model.eval()
        except Exception:
            _esm2_model = None
    return _esm2_model, _esm2_tokenizer


def esm2_pseudo_perplexity(seqs: list) -> list:
    """Batch ESM-2 pseudo-perplexity using one-fell-swoop masking."""
    model, tok = _get_esm2()
    if model is None or not seqs:
        return [0.0] * len(seqs)
    ppls = []
    for seq in seqs:
        if not seq:
            ppls.append(999.0)
            continue
        try:
            import torch
            inputs = tok(seq, return_tensors='pt')
            ids    = inputs['input_ids'][0]  # (L+2,)
            L      = len(ids) - 2
            log_prob_sum = 0.0
            with torch.no_grad():
                for i in range(1, L + 1):
                    masked = ids.clone()
                    masked[i] = tok.mask_token_id
                    out = model(input_ids=masked.unsqueeze(0),
                                attention_mask=inputs['attention_mask'])
                    logits = out.last_hidden_state[0, i]
                    lp = float(torch.log_softmax(logits, dim=-1)[ids[i]])
                    log_prob_sum += lp
            ppls.append(float(np.exp(-log_prob_sum / L)))
        except Exception:
            ppls.append(999.0)
    return ppls


def rank_candidates(candidates: list, spectral_lps: list, obs_mz: np.ndarray,
                    obs_int: np.ndarray, precursor_mass: float,
                    lam: float = 0.05, use_esm: bool = True) -> str:
    """
    NOVEL #10: score each candidate by:
      score(c) = spectral_logprob(c) - lam * esm2_ppl(c) + spectral_support(c)
    Returns the best candidate string.
    """
    if not candidates:
        return ''
    if len(candidates) == 1:
        return candidates[0]

    obs_int_norm = obs_int / (obs_int.max() + 1e-9) if len(obs_int) > 0 else obs_int
    ppls = esm2_pseudo_perplexity(candidates) if use_esm else [0.0]*len(candidates)

    best_seq, best_score = candidates[0], float('-inf')
    for seq, sp_lp, ppl in zip(candidates, spectral_lps, ppls):
        b_ions, y_ions = _compute_by_ions(seq)
        all_ions = np.concatenate([b_ions, y_ions]) if len(b_ions) > 0 else np.array([])
        supp = _peak_support(all_ions, obs_mz, obs_int_norm) if len(all_ions) > 0 else 0.0
        score = sp_lp - lam * ppl + supp
        if score > best_score:
            best_score, best_seq = score, seq
    return best_seq


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
                        n_candidates: int = 5, T_sample: float = 1.0,
                        t_infer: int = 100, device=None, use_gate: bool = False,
                        use_beam: bool = False, use_cfid: bool = False):
    """
    Iterative reverse diffusion over _INFER_STEPS (20 accelerated steps T→0).

    use_beam=True: final step uses mass-constrained beam search (NOVEL #8).
    T_sample < 1.0: final step samples with temperature instead of argmax
                    (used for multi-candidate generation for reranking).

    Returns:
        sequences:    list[N] of list[n_candidates] strings
        spectral_lps: list[N] of list[n_candidates] floats
        gate_confs:   list[N] of list[n_candidates] floats (1.0 = gate not applied)
        t0_logits:    list[N] of (L, V) tensor at t=0 (last candidate, for SGIR)
    """
    if device is None:
        device = next(encoder.parameters()).device
    encoder.eval(); denoiser.eval()

    spec_t  = torch.tensor(spectra,          dtype=torch.float32, device=device)
    mass_t  = torch.tensor(precursor_masses, dtype=torch.float32, device=device)
    context, peak_pad_mask = encoder(spec_t, mass_t)   # (N, K+1, d), (N, K+1)
    N       = len(spectra)

    sequences    = [[] for _ in range(N)]
    spectral_lps = [[] for _ in range(N)]
    gate_confs   = [[] for _ in range(N)]
    t0_logits    = [None] * N

    n_steps = len(_INFER_STEPS)

    for cand_idx in range(n_candidates):
        # Start from fully masked sequence (absorbing diffusion: t=T ≡ all-MASK)
        xt = torch.full((N, SEQ_LEN), MASK_TOK, dtype=torch.long, device=device)

        for step_idx, t_cur in enumerate(_INFER_STEPS):
            t_vec  = torch.full((N,), t_cur, dtype=torch.long, device=device)
            logits = denoiser(xt, t_vec, context,
                              peaks=spec_t, precursor_masses=mass_t,
                              memory_key_padding_mask=peak_pad_mask)   # (N, L, V)
            logits[..., MASK_TOK] = float('-inf')   # never predict MASK as output

            is_final = (step_idx == n_steps - 1)

            if not is_final:
                x0_hat = logits.argmax(-1)           # (N, L) — EOS/PAD allowed
                t_next     = _INFER_STEPS[step_idx + 1]
                t_next_vec = torch.full((N,), t_next, dtype=torch.long, device=device)
                xt = q_sample(x0_hat, t_next_vec)
            else:
                log_fin = F.log_softmax(logits, dim=-1)  # (N, L, V)

                # Store t=0 logits from last candidate for SGIR
                for i in range(N):
                    t0_logits[i] = log_fin[i].cpu()

                if use_beam:
                    # NOVEL #8: mass-constrained beam search
                    beam_toks_list = mass_constrained_beam_search(
                        logits, mass_t, beam_width=20)
                    for bi, toks in enumerate(beam_toks_list):
                        toks_pad = (toks + [0] * SEQ_LEN)[:SEQ_LEN]
                        tok_t    = torch.tensor(toks_pad, dtype=torch.long, device=device)
                        lp       = log_fin[bi].gather(-1, tok_t.unsqueeze(-1)).squeeze(-1)
                        aa_mask  = (tok_t >= 3).float()
                        sp_lp_i  = (lp * aa_mask).sum() / aa_mask.sum().clamp(min=1)
                        sequences[bi].append(decode_tokens(toks_pad))
                        spectral_lps[bi].append(float(sp_lp_i.cpu()))
                        gate_confs[bi].append(1.0)
                elif T_sample != 1.0 and n_candidates > 1:
                    # Temperature sampling: diverse candidates for reranking
                    scaled = logits / max(T_sample, 1e-6)
                    probs  = F.softmax(scaled, dim=-1)
                    x0_samp = torch.multinomial(
                        probs.reshape(-1, VOCAB_SIZE), 1).reshape(N, SEQ_LEN)
                    sp_lp = log_fin.gather(-1, x0_samp.unsqueeze(-1)).squeeze(-1)
                    aa_mask = (x0_samp >= 3).float()
                    sp_lp   = (sp_lp * aa_mask).sum(-1) / aa_mask.sum(-1).clamp(min=1)
                    for i, seq in enumerate(x0_samp.cpu().numpy()):
                        sequences[i].append(decode_tokens(seq))
                        spectral_lps[i].append(float(sp_lp[i].cpu()))
                        gate_confs[i].append(1.0)
                elif use_cfid:
                    # NOVEL #11: Mask-Predict iterative decoding
                    cfid_seqs = mask_predict_decode(logits, context, denoiser, n_iter=4,
                                                    peaks=spec_t, precursor_masses=mass_t,
                                                    memory_key_padding_mask=peak_pad_mask)
                    for bi, seq in enumerate(cfid_seqs):
                        tok_ids = [(CHAR_TO_IDX.get(c, 0) if c in CHAR_TO_IDX else 0)
                                   for c in seq]
                        tok_t   = torch.tensor((tok_ids + [0] * SEQ_LEN)[:SEQ_LEN],
                                               dtype=torch.long, device=device)
                        lp      = log_fin[bi].gather(-1, tok_t.unsqueeze(-1)).squeeze(-1)
                        aa_mask = (tok_t >= 3).float()
                        sp_lp_i = (lp * aa_mask).sum() / aa_mask.sum().clamp(min=1)
                        sequences[bi].append(seq)
                        spectral_lps[bi].append(float(sp_lp_i.cpu()))
                        gate_confs[bi].append(1.0)
                else:
                    # Default: argmax
                    x0_final = logits.argmax(-1)                         # (N, L)
                    sp_lp    = log_fin.gather(-1, x0_final.unsqueeze(-1)).squeeze(-1)
                    aa_mask  = (x0_final >= 3).float()
                    sp_lp    = (sp_lp * aa_mask).sum(-1) / aa_mask.sum(-1).clamp(min=1)
                    for i, seq in enumerate(x0_final.cpu().numpy()):
                        sequences[i].append(decode_tokens(seq))
                        spectral_lps[i].append(float(sp_lp[i].cpu()))
                        gate_confs[i].append(1.0)

    return sequences, spectral_lps, gate_confs, t0_logits


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
    """Positional recall: fraction of positions where pred[i] == true[i]."""
    matches = sum(a == b for a, b in zip(pred, true))
    return matches / max(len(true), 1)


def evaluate_aa_recall(encoder, denoiser, X_test, y_test, masses_test,
                        batch_size=32, results_dir='results', device=None,
                        use_gate=False, use_beam=False, use_cfid=False,
                        use_rerank=False, n_rerank=30, T_sample=0.8,
                        use_sgir=False, raw_peaks=None, use_esm=False,
                        use_mass_correct=False):
    """
    Evaluate de-novo sequencing on a held-out test set.

    Flags (can be combined):
      use_beam         – NOVEL #8: mass-constrained beam search at final step
      use_cfid         – NOVEL #11: Mask-Predict iterative decoding (bidirectional)
      use_rerank       – NOVEL #10: generate n_rerank candidates, ESM-2+spectral rerank
      use_sgir         – NOVEL #9: spectrally-grounded iterative refinement post-decode
                         requires raw_peaks: list[(mz_arr, int_arr)] aligned with X_test
      use_mass_correct – Option B: post-hoc single-swap mass correction
    """
    if device is None:
        device = next(encoder.parameters()).device

    n_cands = n_rerank if use_rerank else 1
    t_samp  = T_sample if use_rerank else 1.0

    all_seqs, all_lps, all_gcs = [], [], []
    recalls, pep_correct = [], []

    for i in range(0, len(X_test), batch_size):
        bs   = X_test[i:i+batch_size]
        bm   = masses_test[i:i+batch_size]
        byt  = y_test[i:i+batch_size]
        brp  = raw_peaks[i:i+batch_size] if raw_peaks is not None else None

        seqs, lps, gcs, t0_lgs = generate_sequences(
            encoder, denoiser, bs, bm,
            n_candidates=n_cands, T_sample=t_samp,
            device=device, use_gate=use_gate, use_beam=use_beam,
            use_cfid=use_cfid)

        for j, (pred_list, lp_list, gc_list, true_tok) in enumerate(
                zip(seqs, lps, gcs, byt)):
            pm    = float(bm[j])
            obs_mz  = brp[j][0] if brp is not None else np.array([])
            obs_int = brp[j][1] if brp is not None else np.array([])

            # NOVEL #10: rerank candidates
            if use_rerank and len(pred_list) > 1:
                best = rank_candidates(pred_list, lp_list, obs_mz, obs_int, pm,
                                       use_esm=use_esm)
                best_lp = lp_list[pred_list.index(best)] if best in pred_list else lp_list[0]
            else:
                best    = pred_list[0]
                best_lp = lp_list[0]

            # NOVEL #9: SGIR refinement
            if use_sgir and brp is not None and t0_lgs[j] is not None:
                best = spectrally_grounded_refine(
                    best, obs_mz, obs_int, pm, t0_lgs[j])

            # Option B: post-hoc single-swap mass correction
            if use_mass_correct:
                best = mass_correct_sequence(best, pm)

            all_seqs.append([best]); all_lps.append([best_lp])
            all_gcs.append([gc_list[0]])
            true_seq = decode_tokens(true_tok)
            recalls.append(aa_recall(best, true_seq))
            pep_correct.append(best == true_seq)

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
    encoder  = PeakEncoder().to(device)
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
