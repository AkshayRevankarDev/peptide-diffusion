"""
Regenerate figure6_ablation.png and figure8_mass_constraints.png from
results/novels_ablation_v2.csv.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(REPO_DIR, 'figures')
CSV_PATH = os.path.join(REPO_DIR, 'results', 'novels_ablation_v2.csv')

df = pd.read_csv(CSV_PATH)
means = df.groupby('config')[['AA Recall %', 'Pep Acc %']].mean()
stds  = df.groupby('config')[['AA Recall %', 'Pep Acc %']].std()

INSTANOVO_AA  = 72.9
INSTANOVO_PEP = 33.1

# ── Figure 6: main ablation (7 original configs) ──────────────────────────────
MAIN_CONFIGS = [
    ('Baseline (argmax)',        'argmax'),
    ('NOVEL#9 SGIR',             'SGIR'),
    ('NOVEL#11 CFID',            'CFID'),
    ('NOVEL#11 CFID + SGIR',     'CFID+SGIR'),
    ('NOVEL#11 CFID + mass-corr','CFID+mass'),
    ('Option B mass-correct',    'mass-corr'),
    ('NOVEL#10 rerank-spectral', 'rerank'),
]
BEST = 'NOVEL#11 CFID + SGIR'
NEG  = {'NOVEL#11 CFID + mass-corr', 'Option B mass-correct'}

def bar_color(cfg):
    if cfg == BEST:
        return '#2ecc71'
    if cfg in NEG:
        return '#e74c3c'
    return '#3498db'

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
fig.suptitle('Ablation: 7 Inference Strategies × 3 Seeds  |  Mean ± Std',
             fontsize=13, fontweight='bold', y=1.01)

for ax, metric, instanovo_val, ylabel, ylim in [
    (axes[0], 'AA Recall %',  INSTANOVO_AA,  'AA Recall (%)',       (70, 79)),
    (axes[1], 'Pep Acc %',    INSTANOVO_PEP, 'Peptide Accuracy (%)', (30, 68)),
]:
    labels = [lbl for _, lbl in MAIN_CONFIGS]
    vals   = [means.loc[cfg, metric] for cfg, _ in MAIN_CONFIGS]
    errs   = [stds.loc[cfg, metric]  for cfg, _ in MAIN_CONFIGS]
    colors = [bar_color(cfg) for cfg, _ in MAIN_CONFIGS]
    x = np.arange(len(labels))

    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors,
                  edgecolor='white', linewidth=0.6, error_kw={'elinewidth': 1.2})
    ax.axhline(instanovo_val, color='#e74c3c', linestyle='--',
               linewidth=1.3, label=f'InstaNovo {instanovo_val}%', zorder=3)

    best_idx = [i for i, (cfg, _) in enumerate(MAIN_CONFIGS) if cfg == BEST][0]
    ax.annotate('★ Best', xy=(best_idx, vals[best_idx] + errs[best_idx] + 0.15),
                ha='center', va='bottom', fontsize=9, color='#27ae60', fontweight='bold')

    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v - e - 0.15, f'{v:.2f}', ha='center', va='top',
                fontsize=7.5, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=8)
    title = 'AA Recall' if metric == 'AA Recall %' else 'Peptide Accuracy'
    ax.set_title(f'{title} — Inference Ablation', fontsize=11, pad=8)

plt.tight_layout()
out6 = os.path.join(FIG_DIR, 'figure6_ablation.png')
fig.savefig(out6, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out6}')


# ── Figure 8: mass constraint strategies ─────────────────────────────────────
MASS_CONFIGS = [
    ('Option B mass-correct',        'post-hoc\nswap'),
    ('NOVEL#8 beam',                 'beam\nsearch'),
    ('NOVEL#8 beam+SGIR',            'beam\n+SGIR'),
    ('NOVEL#1 gate',                 'gate\nalone'),
    ('NOVEL#1 gate + CFID',          'gate\n+CFID'),
    ('NOVEL#1 gate+CFID+SGIR',       'gate\n+CFID+SGIR'),
    ('NOVEL#11 CFID + SGIR',         'CFID+SGIR\n(best, no gate)'),
]

MASS_COLORS = {
    'Option B mass-correct':    '#e67e22',
    'NOVEL#8 beam':             '#c0392b',
    'NOVEL#8 beam+SGIR':        '#c0392b',
    'NOVEL#1 gate':             '#8e44ad',
    'NOVEL#1 gate + CFID':      '#2980b9',
    'NOVEL#1 gate+CFID+SGIR':   '#2980b9',
    'NOVEL#11 CFID + SGIR':     '#2ecc71',
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
fig.suptitle('Mass Constraint Strategies vs. Best Baseline  |  Mean ± Std (3 seeds)',
             fontsize=13, fontweight='bold', y=1.01)

for ax, metric, instanovo_val, ylabel in [
    (axes[0], 'AA Recall %',  INSTANOVO_AA,  'AA Recall (%)'),
    (axes[1], 'Pep Acc %',    INSTANOVO_PEP, 'Peptide Accuracy (%)'),
]:
    labels = [lbl for _, lbl in MASS_CONFIGS]
    vals   = [means.loc[cfg, metric] for cfg, _ in MASS_CONFIGS]
    errs   = [stds.loc[cfg, metric]  for cfg, _ in MASS_CONFIGS]
    colors = [MASS_COLORS[cfg] for cfg, _ in MASS_CONFIGS]
    x = np.arange(len(labels))

    ax.bar(x, vals, yerr=errs, capsize=4, color=colors,
           edgecolor='white', linewidth=0.6, error_kw={'elinewidth': 1.2})
    ax.axhline(instanovo_val, color='#e74c3c', linestyle='--',
               linewidth=1.3, label=f'InstaNovo {instanovo_val}%', zorder=3)

    for i, (v, e) in enumerate(zip(vals, errs)):
        offset = e + 0.8
        ax.text(i, v + offset, f'{v:.1f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=8)
    title = 'AA Recall' if metric == 'AA Recall %' else 'Peptide Accuracy'
    ax.set_title(f'{title} — Mass Constraint Strategies', fontsize=11, pad=8)

legend_patches = [
    mpatches.Patch(color='#e67e22', label='post-hoc swap'),
    mpatches.Patch(color='#c0392b', label='beam search (incompatible direction)'),
    mpatches.Patch(color='#8e44ad', label='entropy-adaptive gate (alone)'),
    mpatches.Patch(color='#2980b9', label='gate + CFID (subsumes gate benefit)'),
    mpatches.Patch(color='#2ecc71', label='CFID+SGIR baseline'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           fontsize=8, bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

plt.tight_layout()
out8 = os.path.join(FIG_DIR, 'figure8_mass_constraints.png')
fig.savefig(out8, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out8}')
