"""
Figure 1 — Input-format contrast: how the three time-encoding approaches represent
"patient had disease X at age Y".

Panels:
  A  Setting 2  / Delphi     — ordinal-token stream with causal attention
  B  Setting 3/5 / Prose     — text serialisation → single vector
  C  Setting 4  / Trajectory — decoupled age (sin/cos) + ICD encoder + concat/pool

Run:
    conda run -n delphi python evaluation/plot_input_formats.py
Output:
    evaluation/figures/figure_input_formats.png
    evaluation/figures/figure_input_formats.pdf
    manuscript/figures/figure_input_formats.png
    manuscript/figures/figure_input_formats.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedding.trajectory_embedding import AgeEncoder

OUT_DIR = PROJECT_ROOT / "evaluation" / "figures"
MS_DIR  = PROJECT_ROOT / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MS_DIR.mkdir(parents=True, exist_ok=True)

# ── palette (mirrors plot_method_overview.py) ──────────────────────────────────
MC = [
    "#3A5A85",   # 0  Delphi         — slate blue
    "#4A5C66",   # 1  Baseline CoxPH — charcoal slate
    "#3A7050",   # 2  Disease Text   — forest sage
    "#2E637A",   # 3  Trajectory     — deep teal
    "#7A4A3A",   # 4  Clinical Summ  — terracotta
]
LIGHT_BG = "#F7F8FA"
DARK_T   = "#1A1A1A"
MID_T    = "#444444"
ARR_C    = "#606060"

# ── generate real AgeEncoder embeddings ────────────────────────────────────────
encoder = AgeEncoder(n_embd=128)
AGE_SAMPLES = [20.0, 30.0, 40.0, 50.0, 60.0]
age_embs = encoder.encode_batch(np.array(AGE_SAMPLES))   # (5, 128)
age_embs_vis = age_embs[:, :32]                 # first 32 dims for heatmap

# ── canvas ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("white")

for ax in axes:
    ax.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# helper: remove top/right spines (used for heatmap sub-axes)
def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Setting 2: Delphi — Ordinal-token stream
# ─────────────────────────────────────────────────────────────────────────────
def draw_panel_A(ax):
    col = MC[0]
    col_light = "#C8D5E8"

    # ── panel title ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97, "(A)  Setting 2: Delphi", ha="center", va="top",
            fontsize=11, fontweight="bold", color=DARK_T, transform=ax.transAxes)
    ax.text(0.5, 0.91, "Ordinal-token stream", ha="center", va="top",
            fontsize=9.5, color=MID_T, style="italic", transform=ax.transAxes)

    # ── tokens ─────────────────────────────────────────────────────────────────
    tokens = ["44yr\nJ45", "52yr\nF32", "58yr\nE11", "···"]
    n_tok = len(tokens)
    tok_w, tok_h = 0.16, 0.14
    # x-centres of the 4 tokens, evenly spaced across 0.1 → 0.90
    tok_x = np.linspace(0.13, 0.87, n_tok)
    tok_y_bottom = 0.62   # bottom edge of token boxes

    for i, (tx, label) in enumerate(zip(tok_x, tokens)):
        lx = tx - tok_w / 2
        ly = tok_y_bottom
        alpha = 0.3 if label == "···" else 1.0
        rect = FancyBboxPatch((lx, ly), tok_w, tok_h,
                              boxstyle="round,pad=0.01",
                              facecolor=col_light, edgecolor=col,
                              linewidth=1.5, alpha=alpha,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(tx, tok_y_bottom + tok_h / 2, label,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=col, transform=ax.transAxes)

    # ── right-arrow between tokens ─────────────────────────────────────────────
    for i in range(n_tok - 1):
        x_start = tok_x[i] + tok_w / 2 + 0.01
        x_end   = tok_x[i + 1] - tok_w / 2 - 0.01
        y_mid   = tok_y_bottom + tok_h / 2
        ax.annotate("", xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=ARR_C, lw=1.2))

    # ── causal attention arcs (below the tokens) ───────────────────────────────
    arc_y = tok_y_bottom - 0.04   # baseline for arcs
    for src in range(n_tok - 1):          # last token "···" doesn't draw arcs
        for dst in range(src + 1, n_tok - 1):   # skip "···" as target
            x0, x1 = tok_x[src], tok_x[dst]
            # arc height proportional to span
            height = 0.06 + 0.04 * (dst - src - 1)
            # Bezier-style arc via curved annotation
            ax.annotate("",
                        xy=(x1, arc_y), xytext=(x0, arc_y),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(
                            arrowstyle="->",
                            color=col,
                            lw=0.9,
                            alpha=0.55,
                            connectionstyle=f"arc3,rad=-{height:.2f}"
                        ))

    # ── "sinusoidal" label under first two tokens ──────────────────────────────
    ax.annotate("Age fused into token\nembedding (additive sinusoidal)",
                xy=(tok_x[1], tok_y_bottom),
                xycoords="axes fraction",
                xytext=(0.38, 0.22),
                textcoords="axes fraction",
                fontsize=7.5, ha="center", color=col,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8,
                                connectionstyle="arc3,rad=0.15"))

    # ── bottom annotation ──────────────────────────────────────────────────────
    ax.text(0.5, 0.10,
            "Ordering preserved;\nexact inter-event duration implicit",
            ha="center", va="center", fontsize=8, color=MID_T,
            style="italic", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — Setting 3/5: Prose — Text serialization
# ─────────────────────────────────────────────────────────────────────────────
def draw_panel_B(ax):
    col = MC[2]
    col_light = "#C6DDD4"

    # ── panel title ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97, "(B)  Setting 3/5: Prose", ha="center", va="top",
            fontsize=11, fontweight="bold", color=DARK_T, transform=ax.transAxes)
    ax.text(0.5, 0.91, "Text serialisation", ha="center", va="top",
            fontsize=9.5, color=MID_T, style="italic", transform=ax.transAxes)

    # ── text block ────────────────────────────────────────────────────────────
    prose_lines = (
        '"At age 44.0, patient was\n'
        ' diagnosed with J45\n'
        ' vasomotor rhinitis.\n\n'
        ' At age 52.0, patient was\n'
        ' diagnosed with F32\n'
        ' major depression."'
    )
    text_box_rect = FancyBboxPatch((0.07, 0.47), 0.56, 0.36,
                                   boxstyle="round,pad=0.015",
                                   facecolor="white", edgecolor=col,
                                   linewidth=1.5,
                                   transform=ax.transAxes, clip_on=False)
    ax.add_patch(text_box_rect)
    ax.text(0.35, 0.65, prose_lines,
            ha="center", va="center", fontsize=7.8,
            color=DARK_T, family="monospace",
            transform=ax.transAxes)

    # ── arrow: text block → encoder ───────────────────────────────────────────
    ax.annotate("", xy=(0.72, 0.65), xytext=(0.64, 0.65),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=ARR_C, lw=1.5))

    # ── encoder box ───────────────────────────────────────────────────────────
    enc_rect = FancyBboxPatch((0.72, 0.56), 0.21, 0.18,
                              boxstyle="round,pad=0.01",
                              facecolor=col_light, edgecolor=col,
                              linewidth=1.5,
                              transform=ax.transAxes, clip_on=False)
    ax.add_patch(enc_rect)
    ax.text(0.825, 0.65, "BERT /\nLLM\nencoder",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=col, transform=ax.transAxes)

    # ── arrow: encoder → single vector ────────────────────────────────────────
    ax.annotate("", xy=(0.825, 0.50), xytext=(0.825, 0.56),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=ARR_C, lw=1.5))

    # ── single vector box ─────────────────────────────────────────────────────
    vec_rect = FancyBboxPatch((0.72, 0.38), 0.21, 0.12,
                              boxstyle="round,pad=0.01",
                              facecolor="#E8F4F0", edgecolor=col,
                              linewidth=1.2,
                              transform=ax.transAxes, clip_on=False)
    ax.add_patch(vec_rect)
    ax.text(0.825, 0.44, "single\nvector",
            ha="center", va="center", fontsize=7.5,
            color=col, transform=ax.transAxes)

    # ── annotation: "44.0" in prose ───────────────────────────────────────────
    ax.annotate("Age encoded as text\nscalar '44.0'",
                xy=(0.20, 0.74),
                xycoords="axes fraction",
                xytext=(0.08, 0.30),
                textcoords="axes fraction",
                fontsize=7.5, ha="left", color=col,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))

    # ── bottom annotation ──────────────────────────────────────────────────────
    ax.text(0.5, 0.10,
            "Temporal interval invisible\nto attention",
            ha="center", va="center", fontsize=8, color=MID_T,
            style="italic", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))


# ─────────────────────────────────────────────────────────────────────────────
# Panel C — Setting 4: Trajectory — Decoupled age + event
# ─────────────────────────────────────────────────────────────────────────────
def draw_panel_C(ax, fig):
    col = MC[3]
    col_light = "#B8D5E0"

    # ── panel title ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97, "(C)  Setting 4: Trajectory", ha="center", va="top",
            fontsize=11, fontweight="bold", color=DARK_T, transform=ax.transAxes)
    ax.text(0.5, 0.91, "Decoupled age + event", ha="center", va="top",
            fontsize=9.5, color=MID_T, style="italic", transform=ax.transAxes)

    # ── shared event rows ──────────────────────────────────────────────────────
    ages  = ["44.0", "52.0", "58.0"]
    icds  = ["J45",  "F32",  "E11"]
    n_ev  = len(ages)

    row_ys   = [0.74, 0.63, 0.52]   # y-centres of the three event rows
    age_x    = 0.10   # centre of age column
    icd_x    = 0.46   # centre of ICD column
    box_w, box_h = 0.14, 0.075

    # Age column header
    ax.text(age_x, 0.84, "Age (yrs)", ha="center", va="bottom",
            fontsize=7.5, color=col, fontweight="bold",
            transform=ax.transAxes)
    # ICD column header
    ax.text(icd_x, 0.84, "ICD code", ha="center", va="bottom",
            fontsize=7.5, color=col, fontweight="bold",
            transform=ax.transAxes)

    for ry, age_lbl, icd_lbl in zip(row_ys, ages, icds):
        # age box
        age_rect = FancyBboxPatch((age_x - box_w/2, ry - box_h/2), box_w, box_h,
                                  boxstyle="round,pad=0.01",
                                  facecolor=col_light, edgecolor=col, linewidth=1.2,
                                  transform=ax.transAxes, clip_on=False)
        ax.add_patch(age_rect)
        ax.text(age_x, ry, age_lbl, ha="center", va="center",
                fontsize=8, fontweight="bold", color=col,
                transform=ax.transAxes)
        # ICD box
        icd_rect = FancyBboxPatch((icd_x - box_w/2, ry - box_h/2), box_w, box_h,
                                  boxstyle="round,pad=0.01",
                                  facecolor="#DDEEDD", edgecolor=MC[2], linewidth=1.2,
                                  transform=ax.transAxes, clip_on=False)
        ax.add_patch(icd_rect)
        ax.text(icd_x, ry, icd_lbl, ha="center", va="center",
                fontsize=8, fontweight="bold", color=MC[2],
                transform=ax.transAxes)

    # ── AgeEncoder block ───────────────────────────────────────────────────────
    age_enc_y_bot = 0.36
    ae_rect = FancyBboxPatch((0.01, age_enc_y_bot), 0.22, 0.10,
                             boxstyle="round,pad=0.01",
                             facecolor=col_light, edgecolor=col, linewidth=1.5,
                             transform=ax.transAxes, clip_on=False)
    ax.add_patch(ae_rect)
    ax.text(0.12, age_enc_y_bot + 0.05, "AgeEncoder\n(sin/cos 128-d)",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color=col, transform=ax.transAxes)

    # arrows: age boxes → AgeEncoder
    for ry in row_ys:
        ax.annotate("", xy=(0.12, age_enc_y_bot + 0.10),
                    xytext=(age_x, ry - box_h/2),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9,
                                   connectionstyle="arc3,rad=0.1"))

    # ── TextEncoder block ──────────────────────────────────────────────────────
    te_y_bot = 0.36
    te_rect = FancyBboxPatch((0.37, te_y_bot), 0.22, 0.10,
                             boxstyle="round,pad=0.01",
                             facecolor="#DDEEDD", edgecolor=MC[2], linewidth=1.5,
                             transform=ax.transAxes, clip_on=False)
    ax.add_patch(te_rect)
    ax.text(0.48, te_y_bot + 0.05, "TextEncoder\n(BCB / Qwen)",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color=MC[2], transform=ax.transAxes)

    # arrows: ICD boxes → TextEncoder
    for ry in row_ys:
        ax.annotate("", xy=(0.48, te_y_bot + 0.10),
                    xytext=(icd_x, ry - box_h/2),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=MC[2], lw=0.9,
                                   connectionstyle="arc3,rad=-0.1"))

    # ── concat + pool block ────────────────────────────────────────────────────
    pool_x = 0.285
    pool_y_bot = 0.20
    pool_rect = FancyBboxPatch((pool_x - 0.115, pool_y_bot), 0.23, 0.10,
                               boxstyle="round,pad=0.01",
                               facecolor="#E8ECF5", edgecolor="#555577", linewidth=1.4,
                               transform=ax.transAxes, clip_on=False)
    ax.add_patch(pool_rect)
    ax.text(pool_x, pool_y_bot + 0.05, "Concat + Pool\n→ patient vector",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#333355", transform=ax.transAxes)

    # arrows: AgeEncoder + TextEncoder → Concat
    ax.annotate("", xy=(pool_x - 0.05, pool_y_bot + 0.10),
                xytext=(0.12, age_enc_y_bot),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2,
                                connectionstyle="arc3,rad=-0.15"))
    ax.annotate("", xy=(pool_x + 0.05, pool_y_bot + 0.10),
                xytext=(0.48, te_y_bot),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MC[2], lw=1.2,
                                connectionstyle="arc3,rad=0.15"))

    # ── inset heatmap: real sin/cos encodings ──────────────────────────────────
    # Use axes fraction coordinates: place inset at right side of panel C
    inset_ax = ax.inset_axes([0.62, 0.20, 0.36, 0.60])
    im = inset_ax.imshow(age_embs_vis, aspect="auto", cmap="RdBu_r",
                         vmin=-1, vmax=1, interpolation="nearest")
    inset_ax.set_yticks(range(len(AGE_SAMPLES)))
    inset_ax.set_yticklabels([f"{int(a)}" for a in AGE_SAMPLES], fontsize=6.5)
    inset_ax.set_xlabel("Dim (first 32 of 128)", fontsize=6.5)
    inset_ax.set_ylabel("Age", fontsize=6.5)
    inset_ax.set_title("128-dim sinusoidal\nage encoding", fontsize=7,
                        fontweight="bold", color=col)
    inset_ax.tick_params(axis="both", labelsize=6, length=2)
    # Add a minimal colorbar
    cbar = fig.colorbar(im, ax=inset_ax, fraction=0.046, pad=0.08,
                        orientation="vertical")
    cbar.ax.tick_params(labelsize=5.5)
    clean_spines(inset_ax)

    # ── bottom annotation: geometric separation ────────────────────────────────
    ax.text(0.30, 0.09,
            "Explicit geometric separation:\nage 44 vs 52 = ±0.008 rad/yr different",
            ha="center", va="center", fontsize=7.5, color=MID_T,
            style="italic", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))


# ── draw all panels ────────────────────────────────────────────────────────────
draw_panel_A(axes[0])
draw_panel_B(axes[1])
draw_panel_C(axes[2], fig)

# ── tight layout & save ────────────────────────────────────────────────────────
fig.tight_layout(pad=1.2)

for stem in [OUT_DIR, MS_DIR]:
    fig.savefig(stem / "figure_input_formats.png", dpi=200, bbox_inches="tight")
    fig.savefig(stem / "figure_input_formats.pdf", bbox_inches="tight")

plt.close(fig)
print("Saved:")
for stem in [OUT_DIR, MS_DIR]:
    print(f"  {stem / 'figure_input_formats.png'}")
    print(f"  {stem / 'figure_input_formats.pdf'}")
