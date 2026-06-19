"""Visualise the 128-dim sinusoidal age encoding used in Setting 4.

Two panels:
  (A) Heatmap — encoding matrix for ages 20–80, dims 0–127
  (B) Line plot — selected age curves across dimensions 0–31

Run from evaluation/ directory:
    python plot_age_encoding.py

Outputs: figures/figure_age_encoding.png, figures/figure_age_encoding.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})
from matplotlib.colors import TwoSlopeNorm

os.makedirs("figures", exist_ok=True)

D_MODEL = 128
AGES    = np.linspace(20, 80, 300)      # continuous age range for heatmap
AGES_SPARSE = np.arange(20, 81, 10)    # for x-tick labels

HIGHLIGHT_AGES = [30, 45, 60, 75]      # curves shown in panel B
HIGHLIGHT_COLORS = ["#5B8DB8", "#E07B54", "#4C9B8E", "#7B5EA7"]


def sinusoidal_encoding(ages, d_model=D_MODEL):
    """Return (len(ages), d_model) matrix of sinusoidal age encodings."""
    i   = np.arange(d_model)
    div = np.power(10000.0, (2 * (i // 2)) / d_model)   # (d_model,)
    enc = np.zeros((len(ages), d_model))
    enc[:, 0::2] = np.sin(ages[:, None] / div[0::2])    # even dims → sin
    enc[:, 1::2] = np.cos(ages[:, None] / div[1::2])    # odd  dims → cos
    return enc


def main():
    matrix = sinusoidal_encoding(AGES)                   # (300, 128)
    curves = sinusoidal_encoding(np.array(HIGHLIGHT_AGES))  # (4, 128)

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38,
                            width_ratios=[1.6, 1])

    # ── Panel A: heatmap ─────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax_heat.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[0, D_MODEL, AGES[0], AGES[-1]],
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.03)
    cbar.set_label("Encoding value", fontsize=9)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    ax_heat.set_xlabel("Dimension index", fontsize=10)
    ax_heat.set_ylabel("Age (years)", fontsize=10)
    ax_heat.set_title("")
    ax_heat.set_xticks([0, 16, 32, 48, 64, 80, 96, 112, 128])
    ax_heat.set_yticks([20, 30, 40, 50, 60, 70, 80])

    # Annotate separation between two ages as an inset arrow pair
    for age, color in zip([45, 65], ["#E07B54", "#4C9B8E"]):
        ax_heat.axhline(age, color=color, linewidth=1.0,
                        linestyle="--", alpha=0.7)

    ax_heat.text(-0.10, 1.04, "A", transform=ax_heat.transAxes,
                 fontsize=13, fontweight="bold", va="top")

    # ── Panel B: selected age curves (dims 0–31) ─────────────────────────────
    ax_line = fig.add_subplot(gs[1])
    dims = np.arange(32)
    for age, color, curve in zip(HIGHLIGHT_AGES, HIGHLIGHT_COLORS, curves):
        ax_line.plot(dims, curve[:32], color=color, linewidth=1.8,
                     label=f"age {age}")

    ax_line.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_line.set_xlabel("Dimension index (first 32)", fontsize=10)
    ax_line.set_ylabel("Encoding value", fontsize=10)
    ax_line.set_title("")
    ax_line.set_xlim(0, 31)
    ax_line.set_ylim(-1.15, 1.15)
    ax_line.legend(fontsize=9, loc="upper right", framealpha=0.85,
                   edgecolor="lightgray")
    ax_line.grid(axis="y", linestyle="--", alpha=0.35)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.text(-0.18, 1.04, "B", transform=ax_line.transAxes,
                 fontsize=13, fontweight="bold", va="top")

    for ext in ("png", "pdf"):
        out = f"figures/figure_age_encoding.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
