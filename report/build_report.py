"""
CSE 676 Checkpoint 1 Report Generator
Peptide and Protein Sequencing by Multinomial Diffusion Model
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

# ── Paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
FIG    = os.path.join(BASE, "..", "figures", "eda")
OUT    = os.path.join(BASE, "checkpoint1_report.pdf")

FIG1 = os.path.join(FIG, "amino_acid_freq.png")
FIG2 = os.path.join(FIG, "spectrum_1636.png")
FIG3 = os.path.join(FIG, "training_loss.png")
FIG4 = os.path.join(FIG, "aa_recall_curve.png")
FIG5 = os.path.join(FIG, "confusion_matrix.png")
FIG6 = os.path.join(FIG, "aa_recall_by_length.png")

# ── Colours ──────────────────────────────────────────────────────────────────
NAVY     = colors.HexColor("#003366")
GRAY     = colors.HexColor("#555555")
LTGRAY   = colors.HexColor("#F0F4F8")
WHITE    = colors.white
BLACK    = colors.black
NAVYBG   = colors.HexColor("#003366")

# ── Styles ───────────────────────────────────────────────────────────────────
def make_styles():
    s = {}

    s["title"] = ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=18,
        textColor=BLACK, alignment=TA_CENTER, spaceAfter=8, leading=22
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", fontName="Helvetica", fontSize=11,
        textColor=GRAY, alignment=TA_CENTER, spaceAfter=4, leading=15
    )
    s["team"] = ParagraphStyle(
        "team", fontName="Helvetica-Bold", fontSize=11,
        textColor=BLACK, alignment=TA_CENTER, spaceAfter=4, leading=15
    )
    s["github"] = ParagraphStyle(
        "github", fontName="Helvetica", fontSize=10,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=14, leading=14
    )
    s["h1"] = ParagraphStyle(
        "h1", fontName="Helvetica-Bold", fontSize=13,
        textColor=NAVY, spaceBefore=14, spaceAfter=6, leading=17
    )
    s["h2"] = ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=11,
        textColor=NAVY, spaceBefore=10, spaceAfter=4, leading=15
    )
    s["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10.5,
        textColor=BLACK, alignment=TA_JUSTIFY,
        spaceAfter=7, leading=15, firstLineIndent=0
    )
    s["bullet"] = ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=10.5,
        textColor=BLACK, alignment=TA_LEFT,
        spaceAfter=4, leading=15, leftIndent=18, firstLineIndent=-10
    )
    s["caption"] = ParagraphStyle(
        "caption", fontName="Helvetica-Oblique", fontSize=9,
        textColor=GRAY, alignment=TA_CENTER, spaceAfter=10, leading=12
    )
    s["toc_normal"] = ParagraphStyle(
        "toc_normal", fontName="Helvetica", fontSize=10,
        textColor=BLACK, alignment=TA_LEFT, leading=14
    )
    return s

# ── Page-number canvas ────────────────────────────────────────────────────────
class NumberedCanvas:
    """Stores pages so we can write 'Page X of N' after build."""
    pass

def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(letter[0] / 2, 0.55 * inch,
                             f"Page {page_num} of %(total)s")
    canvas.restoreState()

class ReportDocTemplate(SimpleDocTemplate):
    """Subclass that injects total-page-count after first pass."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_pages = 0

    def handle_pageEnd(self):
        self._total_pages = self.page
        super().handle_pageEnd()

    def afterPage(self):
        pass

def build_pdf():
    S = make_styles()

    # We'll do a two-pass approach: build once to count pages, then patch.
    # For simplicity we just use onLaterPages / onFirstPage callbacks.
    story = []

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — Title Page
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "Peptide and Protein Sequencing by<br/>Multinomial Diffusion Model", S["title"]
    ))
    story.append(Paragraph(
        "Checkpoint 1 Report&nbsp;&nbsp;|&nbsp;&nbsp;"
        "CSE 676 Deep Learning&nbsp;&nbsp;|&nbsp;&nbsp;Spring 2026",
        S["subtitle"]
    ))
    story.append(Paragraph(
        "Team: Akshay Mohan Revankar, Sanika Nanjan, Vaishak Girish Kumar",
        S["team"]
    ))
    story.append(Paragraph(
        "GitHub: https://github.com/AkshayRevankarDev/peptide-diffusion",
        S["github"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=18))

    if os.path.exists(FIG1):
        story.append(Image(FIG1, width=5.5 * inch, height=3.2 * inch))
    story.append(Paragraph(
        "Figure 1: Amino Acid Frequency Distribution across 1,495 labeled E. coli EV peptides.",
        S["caption"]
    ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — Introduction + Dataset Description
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Introduction &amp; Problem Statement", S["h1"]))
    story.append(Paragraph(
        "De novo peptide sequencing is the computational task of determining a peptide's amino "
        "acid sequence directly from its tandem mass spectrometry (MS/MS) fragmentation pattern, "
        "without referencing a protein sequence database. This approach is essential for analyzing "
        "complex biological samples — such as wastewater proteomics — where the microbial community "
        "contains organisms with unsequenced genomes, rendering traditional database search methods "
        "ineffective.", S["body"]
    ))
    story.append(Paragraph(
        "Current state-of-the-art de novo sequencing tools (e.g., DeepNovo, Casanovo) rely on "
        "transformer or LSTM architectures that treat sequencing as a supervised sequence "
        "generation problem. While effective on well-characterised proteomes, these models are "
        "fragile to out-of-distribution spectra and lack physical grounding: they do not enforce "
        "that predicted peptide masses match the observed precursor ion mass.", S["body"]
    ))
    story.append(Paragraph(
        "This project proposes a novel approach: framing de novo peptide sequencing as a discrete "
        "denoising problem using a Multinomial Diffusion Model. By iteratively denoising a sequence "
        "of categorical amino acid tokens, the model can incorporate mass constraints naturally into "
        "the reverse diffusion process. Checkpoint 1 establishes the full data pipeline, "
        "exploratory analysis, and an encoder-decoder LSTM baseline against which the diffusion "
        "model will be benchmarked.", S["body"]
    ))

    story.append(Paragraph("2. Dataset Description", S["h1"]))
    story.append(Paragraph(
        "The primary dataset consists of E. coli EV (Extracellular Vesicle) proteomics data from "
        "the ProteomeXchange repository, comprising raw tandem MS/MS spectra in mzML format paired "
        "with peptide sequence identifications from a database search in xlsx format. The database "
        "search results serve as ground-truth labels for supervised training of the baseline model.",
        S["body"]
    ))
    story.append(Paragraph(
        "EDA was performed on the first 5,000 spectra streamed from Ecoli_EV_1.mzML, matched to "
        "the database search results by scan number. Key statistics from the labeled subset are "
        "summarised below:", S["body"]
    ))

    # Summary stats table
    stats_data = [
        ["Metric", "Value"],
        ["Total Identified Peptides", "1,495"],
        ["Unique Peptide Sequences", "793"],
        ["Average Peaks per Spectrum", "3,435.6"],
        ["Average Precursor m/z", "646.9 Da"],
        ["Average Peptide Length", "13.3 amino acids"],
        ["Dominant Charge States", "+2, +3, +4"],
        ["Peptide Length Range", "5 – 30 amino acids"],
    ]
    stats_style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  NAVYBG),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LTGRAY, WHITE]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ])
    stats_table = Table(stats_data, colWidths=[3.0 * inch, 2.8 * inch])
    stats_table.setStyle(stats_style)
    story.append(stats_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "The peptide length distribution peaks around 12–14 residues, consistent with typical "
        "tryptic digestion. Leucine (L), Alanine (A), and Glutamic Acid (E) are the most frequent "
        "amino acids, reflecting the E. coli proteome composition. Spectra from charge states "
        "greater than +4 were excluded from training, as their complex fragmentation patterns are "
        "disproportionately rare and noisy. A secondary dataset of unlabeled wastewater MS/MS "
        "spectra will serve as the inference target domain in Checkpoint 2.", S["body"]
    ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — Preprocessing Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Preprocessing Pipeline", S["h1"]))
    story.append(Paragraph(
        "Raw MS/MS spectra are variable-length lists of (m/z, intensity) coordinate pairs and "
        "must be converted into fixed-size vectors for use with neural networks. Our preprocessing "
        "pipeline consists of four sequential steps, implemented in src/preprocessing.py:",
        S["body"]
    ))
    bullets = [
        "<b>Step 1 — Intensity Thresholding:</b> Peaks with intensity below 0.1% of the "
        "spectrum's base peak intensity are removed. This eliminates baseline electronic noise "
        "while preserving chemically meaningful fragment ions.",
        "<b>Step 2 — Top-K Peak Selection:</b> After thresholding, only the top 200 most intense "
        "peaks are retained. This standard denoising strategy reduces the feature space while "
        "preserving the dominant fragmentation pattern.",
        "<b>Step 3 — Intensity Normalization:</b> All surviving peak intensities are normalised "
        "to the range [0, 1] by dividing by the maximum intensity in the spectrum. This removes "
        "instrument-level gain variation and makes spectra comparable across acquisition runs.",
        "<b>Step 4 — m/z Binning:</b> The m/z axis is discretised into fixed-width bins of "
        "0.1 Da spanning the range [0, 2000] Da, producing a dense vector of exactly 20,000 "
        "dimensions. Intensities are accumulated into bins using np.add.at for "
        "resolution-preserving bin assignment.",
    ]
    for b in bullets:
        story.append(Paragraph(f"&#8226;&nbsp;&nbsp;{b}", S["bullet"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Peptide sequences are encoded as integer token arrays using a 23-token vocabulary: "
        "20 standard amino acids (indices 3–22) plus PAD (0), SOS (1), and EOS (2). Each encoded "
        "sequence has a fixed length of 32 tokens (SOS + up to 30 AA tokens + EOS + "
        "zero-padding), implemented in encode_peptide() in src/preprocessing.py.", S["body"]
    ))

    if os.path.exists(FIG2):
        story.append(Image(FIG2, width=5.5 * inch, height=3.0 * inch))
    story.append(Paragraph(
        "Figure 2: Annotated MS/MS spectrum for scan 1636 with theoretical b-ions (blue, dashed) "
        "and y-ions (red, dashed) overlaid for the identified peptide. Grey peaks are unassigned "
        "fragments.", S["caption"]
    ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — Baseline Model & Results
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Baseline Model &amp; Results", S["h1"]))
    story.append(Paragraph(
        "We implemented a sequence-to-sequence Encoder-Decoder architecture as the Checkpoint 1 "
        "baseline, defined in src/baseline.py. The architecture is deliberately classical to serve "
        "as a lower-bound benchmark for the diffusion model.", S["body"]
    ))

    story.append(Paragraph("Architecture", S["h2"]))
    story.append(Paragraph(
        "The Encoder is a three-layer fully-connected network: Linear(20000&#8594;1024) &#8594; "
        "ReLU &#8594; Dropout(0.3) &#8594; Linear(1024&#8594;512) &#8594; ReLU &#8594; "
        "Dropout(0.3) &#8594; Linear(512&#8594;256). This compresses the 20,000-dimensional "
        "binned spectrum into a 256-dimensional latent context vector.", S["body"]
    ))
    story.append(Paragraph(
        "The Decoder is a 2-layer LSTM with hidden dimension 256. At each decoding step, the "
        "embedded previous amino acid token (embedding dimension 64) is concatenated with the "
        "context vector and fed to the LSTM. A final Linear(256&#8594;23) projection produces "
        "logits over the 23-token vocabulary. Teacher forcing is used during training.", S["body"]
    ))

    story.append(Paragraph("Training Configuration", S["h2"]))
    story.append(Paragraph(
        "The 1,488 labeled spectra are split into Train / Val / Test sets at a 70/15/15 ratio "
        "(1,041 / 223 / 224 samples). Both models are trained for 15 epochs using:", S["body"]
    ))
    train_bullets = [
        "Loss: CrossEntropyLoss with ignore_index=0 (PAD) and label_smoothing=0.1",
        "Optimizer: Adam with lr=1e-3 and weight_decay=1e-5",
        "Gradient clipping: max_norm=1.0",
        "Batch size: 32",
        "Hardware: Apple MPS (M-series GPU)",
    ]
    for b in train_bullets:
        story.append(Paragraph(f"&#8226;&nbsp;&nbsp;{b}", S["bullet"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Ablation Study: GRU vs. LSTM Decoder", S["h2"]))
    story.append(Paragraph(
        "To assess the impact of gating complexity, we replaced the LSTM decoder with an "
        "architecturally identical GRU decoder (hidden_dim=256, num_layers=2) and trained it "
        "under identical conditions. Results on the held-out test set are reported below:",
        S["body"]
    ))

    # Results table
    results_data = [
        ["Model Variant", "Train Loss", "Val Loss", "Test AA Recall", "Test Peptide Acc"],
        ["LSTM Baseline",  "2.0184",    "2.4372",   "31.51%",         "2.68%"],
        ["GRU Ablation",   "1.4938",    "2.2379",   "44.30%",         "6.70%"],
    ]
    results_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVYBG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LTGRAY, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ])
    results_table = Table(
        results_data,
        colWidths=[1.45 * inch, 1.0 * inch, 1.0 * inch, 1.25 * inch, 1.3 * inch]
    )
    results_table.setStyle(results_style)
    story.append(results_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "The GRU ablation achieves notably higher token recall (44.30% vs. 31.51%) and peptide "
        "accuracy (6.70% vs. 2.68%) with lower final training loss, suggesting that the simpler "
        "gating mechanism generalises better under teacher-forcing on this small dataset. Both "
        "models exhibit the same failure mode: qualitative inspection of predictions reveals mode "
        "collapse on longer sequences, where the decoder repeats high-frequency tokens (e.g., "
        "Lysine 'K', Leucine 'L') at tail positions. This is consistent with the cross-entropy "
        "objective's inability to penalise position-specific errors, and motivates the diffusion "
        "model's structured generation approach.", S["body"]
    ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5 — Loss & Recall Curves
    # ─────────────────────────────────────────────────────────────────────────
    if os.path.exists(FIG3):
        story.append(Image(FIG3, width=5.5 * inch, height=3.1 * inch))
    story.append(Paragraph(
        "Figure 3: Training and Validation loss curves for both LSTM and GRU decoders over "
        "15 epochs.", S["caption"]
    ))
    story.append(Spacer(1, 14))
    if os.path.exists(FIG4):
        story.append(Image(FIG4, width=5.5 * inch, height=3.1 * inch))
    story.append(Paragraph(
        "Figure 4: Validation Amino Acid Recall per epoch for the LSTM decoder. "
        "Recall stabilises around epoch 10, indicating early convergence.", S["caption"]
    ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 6 — Confusion Matrix + AA Recall by Length
    # ─────────────────────────────────────────────────────────────────────────
    if os.path.exists(FIG5):
        story.append(Image(FIG5, width=5.0 * inch, height=3.8 * inch))
    story.append(Paragraph(
        "Figure 5: Amino Acid confusion matrix on the test set (LSTM decoder). "
        "Rows = ground truth, Columns = predicted. Strong diagonal entries for L, A, E "
        "reflect dataset frequency bias.", S["caption"]
    ))
    story.append(Spacer(1, 12))
    if os.path.exists(FIG6):
        story.append(Image(FIG6, width=5.0 * inch, height=2.8 * inch))
    story.append(Paragraph(
        "Figure 6: Per-bucket AA Recall by peptide length on the test set. "
        "Short peptides (&lt;10 AA) are predicted more accurately; recall degrades "
        "monotonically for longer sequences.", S["caption"]
    ))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 7 — Limitations + Checkpoint 2 Plan
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Limitations", S["h1"]))
    lim_bullets = [
        "<b>Dataset Size:</b> Only 1,488 labeled spectra are available for supervised training — "
        "orders of magnitude fewer than industrial sequencing pipelines that leverage millions of "
        "identified spectra. At this scale the model cannot learn robust sequence-to-spectrum "
        "mappings and instead overfits to amino acid frequency priors, producing Leu/Lys-heavy "
        "predictions regardless of the input spectrum.",
        "<b>Greedy Decoding:</b> The current pipeline uses argmax (greedy) decoding at inference "
        "time. Production de novo sequencers employ beam search or Viterbi-style decoding with "
        "branching width &#8805;5 to explore the token distribution more thoroughly. Without "
        "beam search, a single poor early prediction is never corrected.",
        "<b>No Mass Constraint:</b> The most fundamental limitation is the absence of a mass "
        "constraint. Any valid prediction must satisfy: sum(residue_masses) &#8776; "
        "precursor_mass. The current model ignores this invariant entirely, producing physically "
        "impossible sequences. This constraint will be enforced in the diffusion model's reverse "
        "process.",
        "<b>No Positional Encoding of m/z:</b> The binned spectrum vector is fed as a flat "
        "20,000-dimensional input with no explicit encoding of which bin corresponds to which "
        "fragment ion type (b vs. y ion series). This prevents the encoder from exploiting the "
        "structural relationship between the input spectrum and the output sequence.",
    ]
    for i, b in enumerate(lim_bullets, 1):
        story.append(Paragraph(f"{i}.&nbsp;&nbsp;{b}", S["bullet"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("6. Plan for Checkpoint 2 (Due: April 19, 2026)", S["h1"]))
    story.append(Paragraph(
        "The primary objective of Checkpoint 2 is to replace the LSTM decoder with a Multinomial "
        "Diffusion Model for discrete sequence generation. Concretely, we will:", S["body"]
    ))
    cp2_bullets = [
        "<b>Implement Forward Diffusion:</b> Design a categorical noise schedule that gradually "
        "corrupts a ground-truth peptide sequence toward a uniform token distribution over "
        "T=200 timesteps.",
        "<b>Implement Reverse Denoising Network:</b> Train a transformer-based denoiser "
        "conditioned on the encoded spectrum context and the current noisy sequence to predict "
        "the clean token at each position.",
        "<b>Mass-Constrained Decoding:</b> Implement a reject-sampling or guided-diffusion "
        "mechanism that enforces the precursor mass constraint during the reverse diffusion "
        "process, discarding or re-weighting candidate sequences whose total residue mass "
        "deviates from the observed precursor m/z by more than 0.02 Da.",
        "<b>Evaluation:</b> Report AA Recall and Peptide Accuracy on both the E. coli EV test "
        "set and a held-out wastewater sample set, comparing the diffusion model directly against "
        "the LSTM and GRU baselines from Checkpoint 1.",
    ]
    for i, b in enumerate(cp2_bullets, 1):
        story.append(Paragraph(f"{i}.&nbsp;&nbsp;{b}", S["bullet"]))
    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 8 — Individual Contributions + LLM Statement
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("7. Individual Contributions", S["h1"]))
    contribs = [
        ("<b>Akshay Mohan Revankar:</b> Designed and implemented the full preprocessing pipeline "
         "(src/preprocessing.py and src/data_loader.py), including mzML streaming, "
         "scan-number-based label joining, peak filtering, m/z binning, and peptide encoding. "
         "Configured the project repository structure, requirements.txt, and data path "
         "abstraction across all notebooks."),
        ("<b>Sanika Nanjan:</b> Led the Exploratory Data Analysis (notebooks/01_eda.ipynb), "
         "producing all distribution plots (peaks per spectrum, precursor m/z, charge state, "
         "peptide length, amino acid frequency), the 3 annotated example spectrum visualisations "
         "with theoretical b/y ion overlays, and the database search score distribution. "
         "Ensured all figures were saved to figures/eda/ with consistent styling."),
        ("<b>Vaishak Girish Kumar:</b> Designed and trained the PyTorch encoder-decoder "
         "architecture (src/baseline.py and notebooks/03_baseline.ipynb), including the LSTM "
         "decoder, GRU ablation study, evaluate_test() evaluation function, confusion matrix "
         "generation, and AA recall-by-length error analysis. Authored the analytical sections "
         "of this report and compiled the final PDF."),
    ]
    for c in contribs:
        story.append(Paragraph(c, S["body"]))
        story.append(Spacer(1, 4))

    story.append(Paragraph("8. LLM Usage Statement", S["h1"]))
    story.append(Paragraph(
        "Claude (Anthropic) was used to assist with boilerplate code generation, pipeline design "
        "scaffolding, and initial report drafting. All scientific and experimental decisions — "
        "including architecture choices, hyperparameter selection, evaluation metrics, and result "
        "interpretation — were made, validated, and verified by the project team. All code was "
        "reviewed and tested by team members before inclusion in the repository.", S["body"]
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Build PDF with "Page X of N" via NumberedCanvas pattern
    # ─────────────────────────────────────────────────────────────────────────
    from reportlab.pdfgen import canvas as rl_canvas

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
            self.drawCentredString(
                letter[0] / 2, 0.5 * inch,
                f"Page {self._pageNumber} of {page_count}"
            )
            self.restoreState()

    doc = SimpleDocTemplate(
        OUT, pagesize=letter,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=1*inch, bottomMargin=1*inch,
        title="CSE 676 Checkpoint 1 Report — Peptide Diffusion",
        author="Akshay Mohan Revankar, Sanika Nanjan, Vaishak Girish Kumar",
        subject="De Novo Peptide Sequencing by Multinomial Diffusion Model",
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF saved → {OUT}")


if __name__ == "__main__":
    build_pdf()
