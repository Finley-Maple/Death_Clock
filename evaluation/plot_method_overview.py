"""Methods overview figure — redesigned for Nature/Lancet publication.
5 parallel vertical columns, per-column data blocks, desaturated professional palette.
Compact single-column layout for LNCS proceedings (~16cm max width).

Run from evaluation/ directory:
    python plot_method_overview.py
Output: figures/figure_methods.png, figures/figure_methods.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})

os.makedirs("figures", exist_ok=True)

# ── Canvas ─────────────────────────────────────────────────────────────────────
FW, FH = 22, 20
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Palette ────────────────────────────────────────────────────────────────────
MC = [
    "#3A5A85",   # 0  Delphi         — slate blue
    "#4A5C66",   # 1  Baseline CoxPH — charcoal slate
    "#3A7050",   # 2  Disease Text   — forest sage
    "#2E637A",   # 3  Trajectory     — deep teal
    "#7A4A3A",   # 4  Clinical Summ  — terracotta
]
DC = {
    "demo":  "#5B8DB8",
    "labs":  "#C0784A",
    "flags": "#5A9E94",
    "icd":   "#5A8E5A",
}
RAW_C    = "#7A8FA8"
FUSED_C  = "#7A6A9A"
LIGHT_BG = "#F7F8FA"
DARK_T   = "#1A1A1A"
MID_T    = "#444444"
DIV_C    = "#D0D5DB"
ARR_C    = "#606060"

# ── Column geometry ────────────────────────────────────────────────────────────
LM    = 2.2            # left-margin for row labels
COL_W = 3.5
COL_G = 0.45
COL_X = [LM + COL_W / 2 + i * (COL_W + COL_G) for i in range(5)]
PAD   = 0.12

# ── Row y-positions (bottom edge) and heights ──────────────────────────────────
Y_HDR  = 16.8;  H_HDR  = 1.3
Y_DATA = 12.5;  H_DATA = 3.6
Y_FORM = 10.1;  H_FORM = 1.8
Y_ENC  =  4.8;  H_ENC  = 4.6
Y_OUT  =  3.0;  H_OUT  = 1.5

# ── Helpers ────────────────────────────────────────────────────────────────────
def box(cx, ybot, w, h, text, fc, tc=DARK_T, fs=12.0, bold=False,
        ec="#AAAAAA", lw=0.9, alpha=1.0, zorder=2, rpad=0.06):
    rx = cx - w / 2 + PAD
    r  = FancyBboxPatch((rx, ybot), w - 2 * PAD, h,
                        boxstyle=f"round,pad={rpad}",
                        facecolor=fc, edgecolor=ec,
                        linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(r)
    if text:
        ax.text(cx, ybot + h / 2, text,
                ha="center", va="center",
                fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal",
                multialignment="center", linespacing=1.35,
                zorder=zorder + 1)


def dn_arrow(cx, y_from, y_to, col=ARR_C, lw=1.8):
    ax.annotate("",
                xy=(cx, y_to), xytext=(cx, y_from),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=14),
                zorder=6)


def row_lbl(y_ctr, text):
    ax.text(1.1, y_ctr, text,
            ha="center", va="center",
            fontsize=11, color="#666666",
            style="italic", multialignment="center", linespacing=1.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Method column headers
# ═══════════════════════════════════════════════════════════════════════════════
TITLES = [
    "Delphi\n(Zero-shot LLM)",
    "Baseline\nCoxPH",
    "S3 Prose\nEmb + CoxPH",
    "S4 Trajectory\nEmb + CoxPH",
    "S3-Full\nFull Prose",
]
for i, (cx, t) in enumerate(zip(COL_X, TITLES)):
    box(cx, Y_HDR, COL_W, H_HDR, t,
        fc=MC[i], tc="white", fs=14, bold=True,
        ec=MC[i], lw=0, zorder=4, rpad=0.09)

# ═══════════════════════════════════════════════════════════════════════════════
# Section A — Input Data
# ═══════════════════════════════════════════════════════════════════════════════
row_lbl(Y_DATA + H_DATA / 2, "Input\nData")

TILE_M = 0.14

# Col 0 — Delphi: ICD History only
box(COL_X[0], Y_DATA + TILE_M, COL_W, H_DATA - 2 * TILE_M,
    "Temporal ICD History\nage-stamped diagnosis sequence\nJ45 · F32 · E11 · · · → [DEAD]?",
    fc=DC["icd"], tc="white", fs=12.0,
    ec=DC["icd"], lw=0, alpha=0.93, zorder=3)

# Col 1 — Baseline: 3 stacked tiles
_avail  = H_DATA - 2 * TILE_M           # available height
_th     = (_avail - 2 * 0.16) / 3       # per tile
_tg     = 0.16
_stk1   = [
    ("Demographics & Lifestyle\nage · sex · BMI · smoking",     DC["demo"]),
    ("Blood Biomarkers × 21\nHbA1c · HDL/LDL · CRP …",        DC["labs"]),
    ("Comorbidity Flags × 11\nT2DM · CKD · hypertension …",   DC["flags"]),
]
for j, (t, fc) in enumerate(_stk1):
    box(COL_X[1],
        Y_DATA + TILE_M + j * (_th + _tg), COL_W, _th,
        t, fc=fc, tc="white", fs=11.0,
        ec=fc, lw=0, alpha=0.93, zorder=3, rpad=0.06)

# Cols 2 & 3 — ICD primary (encoder input) + three-segment raw features band
_raw_h   = 0.85
_icd_h   = _avail - _raw_h - 0.14
_seg_gap = 0.10
_avail_w = COL_W - 2 * PAD                     # drawable width
_seg_w   = (_avail_w - 2 * _seg_gap) / 3       # per segment
_raw_segs = [
    (DC["demo"],  "Demo"),
    (DC["labs"],  "Labs ×21"),
    (DC["flags"], "Flags ×11"),
]
for col_i in [2, 3]:
    cx = COL_X[col_i]
    # ICD tile — top, larger: this is the encoder's actual text input
    box(cx, Y_DATA + TILE_M + _raw_h + 0.14, COL_W, _icd_h,
        "Temporal ICD History\nage-stamped diagnosis sequence\nJ45 · F32 · E11 · · ·",
        fc=DC["icd"], tc="white", fs=11.5,
        ec=DC["icd"], lw=0, alpha=0.93, zorder=3)
    # Three colored segments (Demo / Labs / Flags) — raw features concatenated later
    x0 = cx - COL_W / 2 + PAD
    for m, (fc, lbl) in enumerate(_raw_segs):
        sx = x0 + m * (_seg_w + _seg_gap)
        r  = FancyBboxPatch((sx, Y_DATA + TILE_M), _seg_w, _raw_h,
                            boxstyle="round,pad=0.03",
                            facecolor=fc, edgecolor="white",
                            linewidth=0.6, alpha=0.93, zorder=3)
        ax.add_patch(r)
        ax.text(sx + _seg_w / 2, Y_DATA + TILE_M + _raw_h / 2, lbl,
                ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold", zorder=4)

# Col 4 — Clinical Summary: 2×2 grid of all 4 modalities
_t2h  = (_avail - 0.16) / 2    # per cell
_t2w  = 1.7
_t2g  = 0.16
_cx4  = COL_X[4]
_xL   = _cx4 - (_t2w / 2 - PAD + _t2g / 2)   # left tile center
_xR   = _cx4 + (_t2w / 2 - PAD + _t2g / 2)   # right tile center
_grid = [
    (DC["demo"],  "Demographics\n& Lifestyle",  _xL, Y_DATA + TILE_M + _t2h + _t2g),
    (DC["labs"],  "Biomarkers\n× 21",           _xR, Y_DATA + TILE_M + _t2h + _t2g),
    (DC["flags"], "Comorbidity\nFlags × 11",    _xL, Y_DATA + TILE_M),
    (DC["icd"],   "ICD\nHistory",               _xR, Y_DATA + TILE_M),
]
for fc, t, xc, yb in _grid:
    box(xc, yb, _t2w, _t2h, t,
        fc=fc, tc="white", fs=11.0,
        ec=fc, lw=0, alpha=0.93, zorder=3, rpad=0.06)

# Arrows: data → input format
for i, cx in enumerate(COL_X):
    dn_arrow(cx, Y_DATA, Y_FORM + H_FORM, MC[i])

# ═══════════════════════════════════════════════════════════════════════════════
# Section B — Input Format
# ═══════════════════════════════════════════════════════════════════════════════
row_lbl(Y_FORM + H_FORM / 2, "Input\nFormat")

FORM_TXT = [
    "Discrete ICD token sequence\n[D₁  D₂  …  Dₙ  ⊕DEAD?]",
    "39 numeric features\n[age · sex · BMI\n21 labs · 11 flags]",
    "Disease-only prose text\n\"At age 44, asthma (J45) …\"",
    "Temporal token stream\n\"44yr:J45 → 52yr:F32 → …\"",
    "Full clinical prose\n(demographics + labs\n+ disease history)",
]
for i, (cx, txt) in enumerate(zip(COL_X, FORM_TXT)):
    box(cx, Y_FORM, COL_W, H_FORM, txt,
        fc=LIGHT_BG, tc=DARK_T, fs=11.5,
        ec=MC[i], lw=1.8, zorder=2)
    dn_arrow(cx, Y_FORM, Y_ENC + H_ENC, MC[i])

# ═══════════════════════════════════════════════════════════════════════════════
# Section C — Encoder / Architecture
# ═══════════════════════════════════════════════════════════════════════════════
row_lbl(Y_ENC + H_ENC / 2, "Encoder /\nArchitecture")

for i, cx in enumerate(COL_X):
    box(cx, Y_ENC, COL_W, H_ENC, "",
        fc=LIGHT_BG, ec=MC[i], lw=1.8, zorder=2)


def draw_delphi(cx):
    ax.text(cx, Y_ENC + H_ENC - 0.50,
            "Autoregressive LLM (Delphi)",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=MC[0], zorder=4)
    tokens = ["D₁", "D₂", "…", "Dₙ", "⊕"]
    tclrs  = [DC["icd"]] * 4 + ["#B03030"]
    tw     = (COL_W - 0.70) / len(tokens)
    ty     = Y_ENC + H_ENC / 2 + 0.15
    x0     = cx - (COL_W / 2 - 0.35)
    for k, (tok, fc) in enumerate(zip(tokens, tclrs)):
        txc = x0 + (k + 0.5) * tw
        r = FancyBboxPatch((txc - tw * 0.40, ty - 0.38),
                           tw * 0.80, 0.76,
                           boxstyle="round,pad=0.05",
                           facecolor=fc, edgecolor="white",
                           linewidth=0.8, alpha=0.92, zorder=4)
        ax.add_patch(r)
        ax.text(txc, ty, tok, ha="center", va="center",
                fontsize=10.0, color="white", fontweight="bold", zorder=5)
        if k < len(tokens) - 1:
            ax.annotate("",
                        xy=(txc + tw * 0.45, ty), xytext=(txc + tw * 0.40, ty),
                        arrowprops=dict(arrowstyle="-|>", color="#999",
                                        lw=0.8, mutation_scale=8), zorder=5)
    ax.text(cx, Y_ENC + 0.55,
            "Causal attention  ·  death-token logit\n→ sigmoid hazard chain → S(t)",
            ha="center", va="center",
            fontsize=10.5, color=MID_T, zorder=4)


def draw_baseline(cx):
    ax.text(cx, Y_ENC + H_ENC - 0.50,
            "No Encoder (Direct CoxPH Fit)",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=MC[1], zorder=4)
    np.random.seed(42)
    bx    = cx - 1.40
    bcols = [DC["demo"], DC["labs"], DC["labs"], DC["labs"],
             DC["flags"], DC["labs"], DC["demo"], DC["flags"]]
    for k in range(8):
        bh = np.random.uniform(0.35, 1.20)
        r  = FancyBboxPatch((bx + k * 0.36, Y_ENC + 0.75),
                            0.26, bh,
                            boxstyle="square,pad=0.01",
                            facecolor=bcols[k], edgecolor="white",
                            linewidth=0.5, alpha=0.82, zorder=4)
        ax.add_patch(r)
    ax.text(cx, Y_ENC + 0.45,
            "39 raw numerics → Cox partial likelihood",
            ha="center", va="center",
            fontsize=10.5, color=MID_T, zorder=4)


QWEN_SUB = [
    "Disease-only prose → mean-pool → z ∈ ℝ⁴⁰⁹⁶",
    "Temporal token stream → mean-pool → z ∈ ℝ⁴⁰⁹⁶",
    "Full clinical prose → mean-pool → z ∈ ℝ⁴⁰⁹⁶",
]


def draw_qwen(cx, col_i, subtitle):
    mc   = MC[col_i]
    lw_b = COL_W * 0.55    # layer block width
    bx   = cx - lw_b / 2
    n, lh, gap = 4, 0.52, 0.09
    for k in range(n):
        shade = 0.42 + k * 0.14
        c  = matplotlib.colors.to_rgba(mc, alpha=shade)
        ly = Y_ENC + 1.10 + k * (lh + gap)
        r  = FancyBboxPatch((bx, ly), lw_b, lh,
                            boxstyle="round,pad=0.04",
                            facecolor=c, edgecolor="white",
                            linewidth=0.7, zorder=4)
        ax.add_patch(r)
        ax.text(bx + lw_b / 2, ly + lh / 2,
                f"Transformer Block {k + 1}",
                ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold", zorder=5)
    # Embedding indicator
    emb_cx = cx + lw_b / 2 + 0.10 + 0.38    # right of layer stack
    emb_yb = Y_ENC + 1.60
    emb_h  = 0.95
    emb_w  = 0.70
    box(emb_cx, emb_yb, emb_w, emb_h, "z\n4096",
        fc=mc, tc="white", fs=10.0, bold=True,
        ec=mc, lw=0, zorder=5, rpad=0.05)
    ax.annotate("",
                xy=(emb_cx - emb_w / 2 + PAD, emb_yb + emb_h / 2),
                xytext=(bx + lw_b, emb_yb + emb_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=mc,
                                lw=1.1, mutation_scale=10), zorder=5)
    ax.text(cx, Y_ENC + H_ENC - 0.50,
            "Qwen3-8B  (Encoder Mode)",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=mc, zorder=4)
    ax.text(cx, Y_ENC + 0.45, subtitle,
            ha="center", va="center",
            fontsize=10.5, color=MID_T, zorder=4)


draw_delphi(COL_X[0])
draw_baseline(COL_X[1])
for i, sub in enumerate(QWEN_SUB):
    draw_qwen(COL_X[2 + i], 2 + i, sub)

# Section C → Section E arrows (Section D removed)
for i, cx in enumerate(COL_X):
    dn_arrow(cx, Y_ENC, Y_OUT + H_OUT, MC[i])

# ═══════════════════════════════════════════════════════════════════════════════
# Section E — Output
# ═══════════════════════════════════════════════════════════════════════════════
row_lbl(Y_OUT + H_OUT / 2, "Output")

OUT_TXT = [
    "S(t) survival curve\n(zero-shot)",
    "Risk score\nranking",
    "Risk score\nranking",
    "Risk score\nranking",
    "Risk score\nranking",
]
for i, (cx, txt) in enumerate(zip(COL_X, OUT_TXT)):
    box(cx, Y_OUT, COL_W, H_OUT, txt,
        fc=MC[i], tc="white", fs=13, bold=True,
        ec=MC[i], lw=0, zorder=3)

# ═══════════════════════════════════════════════════════════════════════════════
# Structural dividers
# ═══════════════════════════════════════════════════════════════════════════════
for i in range(4):
    xd = (COL_X[i] + COL_X[i + 1]) / 2
    ax.axvline(xd,
               ymin=(Y_OUT - 0.1) / FH,
               ymax=(Y_HDR + H_HDR + 0.12) / FH,
               color=DIV_C, linewidth=0.9, linestyle="--", zorder=0)

# ═══════════════════════════════════════════════════════════════════════════════
# Data modality legend
# ═══════════════════════════════════════════════════════════════════════════════
LEG = [
    (DC["demo"],  "Demographics & Lifestyle"),
    (DC["labs"],  "Blood Biomarkers"),
    (DC["flags"], "Comorbidity Flags"),
    (DC["icd"],   "Temporal ICD History"),
    (FUSED_C,     "All Modalities (fused into prose)"),
]
lx, ly = 0.4, 0.75
ax.text(lx + 0.1, ly, "Data modality colour key:",
        fontsize=12, va="center", color=DARK_T, fontweight="bold")
for k, (fc, lbl) in enumerate(LEG):
    xi = lx + 3.2 + k * 3.5
    r  = FancyBboxPatch((xi - 0.22, ly - 0.18), 0.44, 0.36,
                        boxstyle="round,pad=0.04",
                        facecolor=fc, edgecolor="none", alpha=0.93, zorder=3)
    ax.add_patch(r)
    ax.text(xi + 0.34, ly, lbl, fontsize=12, va="center", color=DARK_T)

# ── Save ───────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
plt.savefig("figures/figure_methods.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.savefig("figures/figure_methods.pdf", bbox_inches="tight",
            facecolor="white")
print("Saved: figures/figure_methods.png  figures/figure_methods.pdf")
