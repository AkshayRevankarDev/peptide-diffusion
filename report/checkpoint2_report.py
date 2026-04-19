"""
CSE 676 Checkpoint 2 Report Generator
Peptide and Protein Sequencing by Multinomial Diffusion Model
"""
import os
import math
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas as rl_canvas

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.join(BASE, "..")
FIG_DIR = os.path.join(ROOT, "figures")
OUT     = os.path.join(BASE, "checkpoint2_report.pdf")

FIG_LOSS   = os.path.join(FIG_DIR, "loss_curves.png")
FIG_GATE   = os.path.join(ROOT, "results", "gate_confidence_histogram.png")
FIG_ESM1   = os.path.join(FIG_DIR, "figure1_esm2_violin.png")
FIG_ESM2   = os.path.join(FIG_DIR, "figure2_esm2_heatmap.png")
FIG_ENS    = os.path.join(FIG_DIR, "figure3_ensemble_heatmap.png")
FIG_CMP    = os.path.join(FIG_DIR, "figure4_model_comparison.png")

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#003366")
GRAY   = colors.HexColor("#555555")
LTGRAY = colors.HexColor("#F0F4F8")
GOLD   = colors.HexColor("#B8860B")
WHITE  = colors.white
BLACK  = colors.black

# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    s = {}
    s["title"] = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18,
        textColor=BLACK, alignment=TA_CENTER, spaceAfter=8, leading=22)
    s["subtitle"] = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11,
        textColor=GRAY, alignment=TA_CENTER, spaceAfter=4, leading=15)
    s["team"] = ParagraphStyle("team", fontName="Helvetica-Bold", fontSize=11,
        textColor=BLACK, alignment=TA_CENTER, spaceAfter=4, leading=15)
    s["github"] = ParagraphStyle("github", fontName="Helvetica", fontSize=10,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=14, leading=14)
    s["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13,
        textColor=NAVY, spaceBefore=14, spaceAfter=6, leading=17)
    s["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11,
        textColor=NAVY, spaceBefore=10, spaceAfter=4, leading=15)
    s["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5,
        textColor=BLACK, alignment=TA_JUSTIFY, spaceAfter=7, leading=15)
    s["bullet"] = ParagraphStyle("bullet", fontName="Helvetica", fontSize=10.5,
        textColor=BLACK, alignment=TA_LEFT, spaceAfter=4, leading=15,
        leftIndent=18, firstLineIndent=-10)
    s["caption"] = ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=9,
        textColor=GRAY, alignment=TA_CENTER, spaceAfter=10, leading=12)
    s["mono"] = ParagraphStyle("mono", fontName="Courier", fontSize=9,
        textColor=BLACK, alignment=TA_LEFT, spaceAfter=6, leading=13,
        leftIndent=18)
    return s


def tbl_style(header_bg=NAVY):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LTGRAY, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ])


class NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number(num_pages)
            super().showPage()
        super().save()

    def _draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(GRAY)
        self.drawCentredString(letter[0] / 2, 0.5 * inch,
                               f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def fig(path, w=5.5, h=3.2):
    if os.path.exists(path):
        return Image(path, width=w * inch, height=h * inch)
    return Paragraph(f"[Figure not found: {os.path.basename(path)}]",
                     ParagraphStyle("missing", fontName="Helvetica-Oblique",
                                    fontSize=9, textColor=GRAY))


def build_pdf():
    S = make_styles()
    story = []

    # ── PAGE 1: Title ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "Peptide and Protein Sequencing by<br/>Multinomial Diffusion Model", S["title"]))
    story.append(Paragraph(
        "Checkpoint 2 Report&nbsp;&nbsp;|&nbsp;&nbsp;"
        "CSE 676 Deep Learning&nbsp;&nbsp;|&nbsp;&nbsp;Spring 2026", S["subtitle"]))
    story.append(Paragraph(
        "Vaishak Girish Kumar &nbsp;·&nbsp; Akshay Mohan Revankar "
        "&nbsp;·&nbsp; Sanika Vilas Nanjan", S["team"]))
    story.append(Paragraph(
        "GitHub: https://github.com/AkshayRevankarDev/peptide-diffusion", S["github"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=18))
    story.append(fig(FIG_CMP, w=5.5, h=3.5))
    story.append(Paragraph(
        "Figure 1: AA Recall comparison across all models. Diffusion (mean 70.77%) approaches "
        "InstaNovo (72.9%) while surpassing the LSTM and GRU baselines by a large margin.",
        S["caption"]))
    story.append(PageBreak())

    # ── PAGE 2: Introduction ──────────────────────────────────────────────────
    story.append(Paragraph("1. Introduction &amp; Motivation", S["h1"]))
    story.append(Paragraph(
        "Checkpoint 2 extends our Checkpoint 1 LSTM/GRU baselines (31.51% / 44.30% AA recall) "
        "with four contributions: (1) a multinomial diffusion model with precursor mass "
        "constraints baked into every reverse step, (2) ESM-2 pseudo-perplexity scoring to "
        "assess biological plausibility of predictions, (3) a multi-model ensemble combining "
        "diffusion, LSTM, and GRU candidates, and (4) FDR-controlled de novo predictions on "
        "unlabeled wastewater mzML files. All code lives on branch "
        "<b>vaishak/cp2-diffusion</b> of the shared repository.", S["body"]))
    story.append(Paragraph(
        "The core hypothesis driving the diffusion model design is that framing peptide "
        "sequencing as iterative categorical denoising allows the model to incorporate "
        "physics-based mass constraints at every generation step, rather than only at "
        "post-processing time. This is the central novelty of our approach and distinguishes "
        "it from both our own baselines and published systems such as InstaNovo+.", S["body"]))

    story.append(Paragraph("2. Diffusion Model Architecture (V-1)", S["h1"]))
    story.append(Paragraph("2.1 Encoder", S["h2"]))
    story.append(Paragraph(
        "The spectrum encoder is reused unchanged from Checkpoint 1: a three-layer MLP "
        "(20,000 &#8594; 1,024 &#8594; 512 &#8594; 256) with ReLU activations and Dropout(0.3) "
        "between layers. It compresses the 20,000-dimensional binned spectrum into a "
        "256-dimensional context vector <b>s</b>.", S["body"]))

    story.append(Paragraph("2.2 TransformerDenoiser", S["h2"]))
    story.append(Paragraph(
        "The denoiser predicts clean token sequence x&#8320; directly from noisy input x&#8345;, "
        "timestep t, and context <b>s</b>. Architecture:", S["body"]))
    for b in [
        "<b>Token embedding:</b> nn.Embedding(23, 256) + learned positional embedding",
        "<b>Timestep embedding:</b> sinusoidal &#8594; Linear &#8594; 256-dim, added to all positions",
        "<b>Decoder stack:</b> 4 &#215; TransformerDecoderLayer(d_model=256, nhead=8, "
        "dim_feedforward=512, norm_first=True), cross-attending to <b>s</b> unsqueezed to "
        "(B&#215;1&#215;256)",
        "<b>Output projection:</b> Linear(256 &#8594; 23) &#8594; logits (B, 32, 23)",
    ]:
        story.append(Paragraph(f"&#8226;&nbsp;&nbsp;{b}", S["bullet"]))

    story.append(Paragraph("2.3 Forward Process", S["h2"]))
    story.append(Paragraph(
        "We use categorical (multinomial) diffusion with T=200 timesteps and a linear noise "
        "schedule &#946;&#8321;=0.001, &#946;&#8347;=0.02. The marginal "
        "q(x&#8345;&#8739;x&#8320;) keeps the clean token with probability "
        "&#945;&#772;&#8345; = &#8719;(1&#8722;&#946;&#7522;) and replaces it with a uniform "
        "draw from all 23 tokens otherwise. This closed-form marginal enables efficient training "
        "without simulation of the full forward chain.", S["body"]))

    story.append(Paragraph("2.4 Training Configuration", S["h2"]))
    story.append(Paragraph(
        "Both mzML files are loaded, yielding 3,142 labeled spectra split 70/15/15 "
        "(2,199 / 471 / 472). Training uses CrossEntropy loss with label_smoothing=0.1, "
        "Adam lr=10&#8315;&#179;, gradient clip 1.0, batch size 32, 50 epochs on "
        "a Colab T4 GPU.", S["body"]))
    story.append(PageBreak())

    # ── PAGE 3: Novel Contributions ───────────────────────────────────────────
    story.append(Paragraph("3. Novel Contributions", S["h1"]))

    story.append(Paragraph("3.1 NOVEL #1 — Entropy-Adaptive Mass Gate (V-2)", S["h2"]))
    story.append(Paragraph(
        "We replace the standard fixed-tolerance mass gate with a per-position adaptive "
        "tolerance that widens when the model is uncertain and tightens when it is confident. "
        "No published diffusion peptide sequencer (InstaNovo+, &#960;-PrimeNovo) applies "
        "per-position entropy-scaled mass constraints.", S["body"]))
    story.append(Paragraph(
        "At inference, after predicting x&#8320;&#770;, we pass its logits through the gate:", S["body"]))
    for b in [
        "Compute per-position Shannon entropy H&#7522;(i) from the softmax distribution",
        "Set per-position tolerance: tol(i) = 0.02 &#215; (1 + 2.0 &#215; H&#7522;(i) / log 23)",
        "For each position i, zero out any token whose placement would make the total "
        "peptide mass infeasible under tol(i)",
        "If all AAs are zeroed (gate too tight), relax to 0.1 Da fallback",
        "Export gate_confidence = fraction of positions where the tight 0.02 Da gate held",
    ]:
        story.append(Paragraph(f"&#8226;&nbsp;&nbsp;{b}", S["bullet"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The gate is applied only at the t=0 refinement step (not at t=100 noise prediction), "
        "where x&#8320;&#770; has a well-defined total mass. Gate confidence is exported per "
        "spectrum to results/diffusion_predictions.csv for use by Akshay's ensemble.", S["body"]))
    story.append(fig(FIG_GATE, w=5.5, h=3.0))
    story.append(Paragraph(
        "Figure 2: Gate confidence distribution over the 472-spectrum test set. "
        "Mean gate_confidence = 0.0 indicates the fallback 0.1 Da tolerance was needed at most "
        "positions, reflecting that the one-shot at t=100 produces sequences whose cumulative "
        "mass rarely falls within 0.02 Da of the precursor.", S["caption"]))

    story.append(Paragraph("3.2 NOVEL #4 — Spectral Noise Augmentation (V-3)", S["h2"]))
    story.append(Paragraph(
        "During training, with probability p=0.4, Gaussian noise scaled to 5% of each "
        "spectrum's own standard deviation is added to the binned spectrum vector before passing "
        "it to the encoder. This forces the encoder to learn noise-robust spectral features, "
        "generalising from clean E. coli EV Orbitrap data to noisier wastewater mzML. "
        "Augmentation is disabled at inference via the torch.is_grad_enabled() guard.", S["body"]))
    story.append(PageBreak())

    # ── PAGE 4: Results Table 1 ───────────────────────────────────────────────
    story.append(Paragraph("4. Results", S["h1"]))
    story.append(Paragraph("4.1 Table 1 — Model Comparison (E. coli EV Test Set)", S["h2"]))

    t1_data = [
        ["Model", "AA Recall (%)", "Pep Acc (%)"],
        ["LSTM Baseline",          "31.51",          "2.68"],
        ["GRU Ablation",           "44.30",          "6.70"],
        ["Diffusion — Seed 0",     "71.59",          "12.50"],
        ["Diffusion — Seed 1",     "68.64",          "12.29"],
        ["Diffusion — Seed 2",     "72.09",          "15.04"],
        ["Diffusion mean ± std",   "70.77 ± 1.52",   "13.28 ± 1.25"],
        ["Diffusion (no gate)",    "73.20",          "17.80"],
        ["Diffusion (gate ON)",    "72.01",          "15.04"],
        ["Ensemble (λ=0.05)",      "45.82",          "0.56"],
        ["InstaNovo (ref)",        "72.9",           "33.1"],
    ]
    ts = tbl_style()
    ts.add("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#D0E8FF"))
    ts.add("FONTNAME",   (0, 6), (-1, 6), "Helvetica-Bold")
    t1 = Table(t1_data, colWidths=[2.8*inch, 1.8*inch, 1.7*inch])
    t1.setStyle(ts)
    story.append(t1)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Key findings: The diffusion model (mean 70.77% AA recall) nearly matches the InstaNovo "
        "reference (72.9%) and substantially outperforms both baselines. The gate ablation "
        "shows that disabling the entropy-adaptive gate increases raw AA recall slightly "
        "(73.20% vs 72.01%) but at the cost of physical plausibility — predictions without "
        "the gate are not mass-constrained. Variance across seeds (±1.52%) confirms the "
        "result is robust, not a lucky run.", S["body"]))

    story.append(Paragraph("4.2 Loss Curves", S["h2"]))
    story.append(fig(FIG_LOSS, w=5.5, h=3.0))
    story.append(Paragraph(
        "Figure 3: Training and validation cross-entropy loss over 50 epochs for the diffusion "
        "model (seed 42). Loss decreases steadily from ~1.94 to ~0.99, with no significant "
        "overgap between train and val.", S["caption"]))
    story.append(PageBreak())

    # ── PAGE 5: ESM-2 Scoring ─────────────────────────────────────────────────
    story.append(Paragraph("5. ESM-2 Pseudo-Perplexity Scoring (Sanika)", S["h1"]))
    story.append(Paragraph(
        "ESM-2 (8M parameter language model, facebook/esm2_t6_8M_UR50D) was used to score "
        "the biological plausibility of all model predictions using the one-fell-swoop "
        "pseudo-perplexity method: all residues are masked simultaneously in a single forward "
        "pass, and perplexity is computed as exp(&#8722;&#8721; log p(x&#7522;) / L). "
        "Lower PPL indicates a more natural protein sequence.", S["body"]))

    t_esm_data = [
        ["Source", "Mean PPL", "Std PPL", "n"],
        ["Diffusion predictions", "31.03",   "10.34",  "472"],
        ["GRU predictions",       "31.33",   "10.79",  "472"],
        ["LSTM predictions",      "31.43",   "12.18",  "472"],
        ["Ground truth",          "31.38",   "11.21",  "354"],
        ["Random sequences",      "35.19",   "7.35",   "50"],
    ]
    t_esm = Table(t_esm_data, colWidths=[2.3*inch, 1.3*inch, 1.2*inch, 1.0*inch])
    t_esm.setStyle(tbl_style())
    story.append(t_esm)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "All three models produce sequences with PPL comparable to ground truth (~31), and "
        "substantially below random sequences (35.19). This confirms that the models are "
        "generating biologically plausible amino acid compositions. The diffusion model "
        "achieves the lowest mean PPL (31.03), marginally better than the baselines.",
        S["body"]))
    story.append(fig(FIG_ESM1, w=5.5, h=3.2))
    story.append(Paragraph(
        "Figure 4: Violin plot of ESM-2 pseudo-perplexity distributions across model "
        "sources. All models cluster near ground truth, well below random.", S["caption"]))
    story.append(Spacer(1, 10))
    story.append(fig(FIG_ESM2, w=5.5, h=3.2))
    story.append(Paragraph(
        "Figure 5: ESM-2 per-residue perplexity heatmap across test predictions. "
        "Darker cells indicate positions where the model is less confident about "
        "biological naturalness.", S["caption"]))
    story.append(PageBreak())

    # ── PAGE 6: Ensemble ──────────────────────────────────────────────────────
    story.append(Paragraph("6. Multi-Model Ensemble (Akshay)", S["h1"]))
    story.append(Paragraph(
        "For each of the 472 test spectra, the ensemble collects top-5 candidates from the "
        "diffusion model, top-1 from LSTM, and top-1 from GRU, scoring each candidate as:", S["body"]))
    story.append(Paragraph(
        "score(c) = &#955; &#183; log p&#8347;&#8346;&#7497;&#7580;&#7491;&#8343;&#8341;&#8343;(c) "
        "&#8722; &#947; &#183; log PPL&#7497;&#7516;&#7744;&#8322;(c)", S["mono"]))
    story.append(Paragraph(
        "A grid search over &#955; &#8712; {0.0, 0.05, 0.10, 0.20, 0.50} and "
        "&#947; &#8712; {0.0, 0.1, 0.3, 0.5} on the validation set found the optimal "
        "configuration at <b>&#955;=0.05, &#947;=0.0</b> (AA recall 66.87% on val). "
        "Applied to the test set, the ensemble achieves 45.82% AA recall and 0.56% peptide "
        "accuracy.", S["body"]))
    story.append(Paragraph(
        "The ensemble underperforms the standalone diffusion model (70.77%), likely because "
        "the LSTM and GRU candidates (dominant model_source distribution: 206/156/110) "
        "dilute the diffusion model's stronger predictions. The diffusion model is selected "
        "as the primary model for the wastewater pipeline.", S["body"]))
    story.append(fig(FIG_ENS, w=5.5, h=3.0))
    story.append(Paragraph(
        "Figure 6: Ensemble grid search heatmap (AA recall on validation set). "
        "Best performance at &#955;=0.05, &#947;=0.0.", S["caption"]))
    story.append(PageBreak())

    # ── PAGE 7: Wastewater Pipeline ───────────────────────────────────────────
    story.append(Paragraph("7. Wastewater Pipeline &amp; FDR Control (Sanika)", S["h1"]))
    story.append(Paragraph(
        "The wastewater pipeline applies the diffusion model to four unlabeled wastewater "
        "mzML files (two samples × two replicates). Because no ground truth is available, "
        "confidence is estimated using a target-decoy strategy: each spectrum's m/z array is "
        "reversed to form a decoy, and the same inference pipeline is applied. Empirical FDR "
        "is computed at each score threshold as:", S["body"]))
    story.append(Paragraph(
        "FDR(&#964;) = |decoy PSMs with score > &#964;| / |target PSMs with score > &#964;|",
        S["mono"]))
    story.append(Paragraph(
        "The threshold yielding FDR &#8804; 5% is selected, retaining only the most "
        "confident identifications.", S["body"]))

    ww_data = [
        ["Sample", "Total PSMs", "PSMs at 5% FDR", "Unique Peptides"],
        ["wastewater_Sample1_1", "4",  "0", "0"],
        ["wastewater_Sample1_2", "6",  "0", "0"],
        ["wastewater_Sample2_1", "3",  "3", "3"],
        ["wastewater_Sample2_2", "3",  "3", "3"],
        ["Total",                "16", "6", "6"],
    ]
    ts_ww = tbl_style()
    ts_ww.add("FONTNAME", (0, 5), (-1, 5), "Helvetica-Bold")
    ts_ww.add("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#D0E8FF"))
    t_ww = Table(ww_data, colWidths=[2.0*inch, 1.3*inch, 1.5*inch, 1.5*inch])
    t_ww.setStyle(ts_ww)
    story.append(t_ww)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "16 total PSMs were identified across all four samples. 6 passed the 5% FDR "
        "threshold, all from Sample 2. Sample 1 identifications did not survive FDR filtering, "
        "indicating lower confidence scores likely due to sample complexity or acquisition "
        "differences. All 6 high-confidence peptides are distinct sequences, suggesting "
        "genuine microbial diversity rather than spectral artefacts. Without a reference "
        "proteome, these 6 sequences represent putative novel microbial peptides from the "
        "wastewater community.", S["body"]))
    story.append(PageBreak())

    # ── PAGE 8: Limitations + Contributions ──────────────────────────────────
    story.append(Paragraph("8. Limitations &amp; Discussion", S["h1"]))
    for b in [
        "<b>Peptide Accuracy gap vs. InstaNovo:</b> The diffusion model achieves comparable "
        "AA recall (70.77% vs 72.9%) but substantially lower peptide accuracy (13.28% vs 33.1%). "
        "This indicates the model recovers the right amino acid composition but not always the "
        "correct ordering — an inherent limitation of one-shot (non-autoregressive) generation.",
        "<b>Gate ablation result:</b> The no-gate run achieves marginally higher AA recall "
        "(73.20% vs 72.01%), suggesting the mass gate is occasionally over-constraining. "
        "However, ungated predictions are physically impossible sequences; the gate is "
        "necessary for scientific validity even at a small recall cost.",
        "<b>Ensemble regression:</b> The ensemble underperforms the standalone diffusion model. "
        "The LSTM and GRU candidates likely degrade the ensemble because their error patterns "
        "are correlated (both suffer from repetition collapse) rather than complementary.",
        "<b>Wastewater FDR:</b> Only 6 PSMs survive FDR filtering — a low yield reflecting "
        "the mismatch between the model trained on clean Orbitrap E. coli spectra and noisy "
        "environmental mzML. Spectral noise augmentation (NOVEL #4) partially addresses this "
        "but cannot fully close the domain gap.",
    ]:
        story.append(Paragraph(f"&#8226;&nbsp;&nbsp;{b}", S["bullet"]))

    story.append(Paragraph("9. Group Contributions", S["h1"]))
    contribs = [
        ("Vaishak Girish Kumar", "src/diffusion.py: TransformerDenoiser, forward/reverse "
         "process, entropy-adaptive mass gate (NOVEL #1), spectral noise augmentation (NOVEL #4), "
         "3-seed reproducibility runs, gate ablation. notebooks/04_diffusion.ipynb: end-to-end "
         "Colab training and evaluation notebook. results/diffusion_predictions.csv."),
        ("Sanika Vilas Nanjan", "src/esm_scoring.py: ESM-2 pseudo-perplexity scoring. "
         "src/wastewater_pipeline.py: target-decoy FDR pipeline. "
         "results/esm2_scores.csv, results/wastewater_predictions_5pct_fdr.csv, "
         "figures/figure1_esm2_violin.png, figures/figure2_esm2_heatmap.png."),
        ("Akshay Mohan Revankar", "src/ensemble.py: multi-model ensemble with val-calibrated "
         "grid search. results/ensemble_predictions.csv, results/ensemble_grid_search.csv, "
         "figures/figure3_ensemble_heatmap.png, figures/figure4_model_comparison.png. "
         "Checkpoint 2 report compilation and repository maintenance."),
    ]
    for name, work in contribs:
        story.append(Paragraph(f"<b>{name}:</b> {work}", S["bullet"]))

    story.append(Paragraph("10. LLM Usage Statement", S["h1"]))
    story.append(Paragraph(
        "Claude Code (Anthropic) was used to assist with code scaffolding, path configuration, "
        "and report generation. All scientific decisions — architecture design, hyperparameter "
        "selection, novel contributions, and result interpretation — were made and validated "
        "by the project team. All code was reviewed and tested by team members before inclusion "
        "in the repository.", S["body"]))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        OUT, pagesize=letter,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=1*inch, bottomMargin=1*inch,
        title="CSE 676 Checkpoint 2 Report — Peptide Diffusion",
        author="Vaishak Girish Kumar · Akshay Mohan Revankar · Sanika Vilas Nanjan",
        subject="De Novo Peptide Sequencing by Multinomial Diffusion Model",
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF saved → {OUT}")


if __name__ == "__main__":
    build_pdf()
