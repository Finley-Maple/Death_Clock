"""Figure 2a: 3-panel barplot for the 5 reference configurations.

Panels: C-index | IBS | Mean TD-AUC
Only the reference config for each setting is shown (5 bars per panel).

Run from evaluation/ directory:
    python plot_results_bars.py

Outputs: figures/figure_2a.png, figures/figure_2a.pdf
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# ── Colour palette (consistent with plot_comparison.py) ──────────────────────
COLORS = {
    "S1": "#7f7f7f",
    "S2": "#5B8DB8",
    "S3": "#E07B54",
    "S4": "#4C9B8E",
    "S5": "#7B5EA7",
}

# ── Reference configuration data sources ─────────────────────────────────────
REF_CONFIGS = [
    {
        "label":       "S1 CoxPH",
        "color":       COLORS["S1"],
        "file":        "benchmarking_results/benchmarking_results.json",
        "is_baseline": True,
    },
    {
        "label":       "S2 Delphi",
        "color":       COLORS["S2"],
        "file":        "delphi_test_results.json",
    },
    {
        "label":       "S3 Prose",
        "color":       COLORS["S3"],
        "file":        "embedding_results/s3_qwen_concat_results.json",
    },
    {
        "label":       "S4 Decoupled",
        "color":       COLORS["S4"],
        "file":        "embedding_results/s4_qwen_concat_results.json",
    },
    {
        "label":       "S5 Unified",
        "color":       COLORS["S5"],
        "file":        "embedding_results/s5_qwen_results.json",
    },
]


def load_refs():
    rows = []
    for cfg in REF_CONFIGS:
        with open(cfg["file"]) as f:
            d = json.load(f)
        test = d["splits"]["test"]
        rows.append({
            "label":       cfg["label"],
            "color":       cfg["color"],
            "is_baseline": cfg.get("is_baseline", False),
            "c_index":     test["c_index"],
            "ibs":         test["ibs"],
            "mean_tdauc":  test["mean_td_auc"],
        })
    return rows


def bar_panel(ax, rows, field, ylabel, title, letter, lower_better=False):
    vals   = [r[field] for r in rows]
    colors = [r["color"] for r in rows]
    labels = [r["label"] for r in rows]
    x      = np.arange(len(rows))

    bars = ax.bar(x, vals, width=0.6, color=colors, edgecolor="white", linewidth=0.6)

    # Dotted baseline reference line
    for r in rows:
        if r["is_baseline"]:
            ax.axhline(r[field], color=r["color"], linestyle=":",
                       linewidth=1.4, alpha=0.75, zorder=3)
            break

    span = max(vals) - min(vals)
    pad  = span * 0.015 if span > 0 else 1e-4
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7.5,
        )

    lo = min(vals) - span * 0.35
    hi = max(vals) + span * 0.55
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    arrow = "↓" if lower_better else "↑"
    ax.set_title(f"{title}  ({arrow})", fontsize=9.5, fontweight="bold")
    ax.text(-0.15, 1.06, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    rows = load_refs()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.subplots_adjust(wspace=0.38)

    bar_panel(axes[0], rows, "c_index",    "Harrell C-index",         "C-index",                "A")
    bar_panel(axes[1], rows, "ibs",        "Integrated Brier Score",  "IBS",                    "B", lower_better=True)
    bar_panel(axes[2], rows, "mean_tdauc", "Mean TD-AUC",             "Mean TD-AUC",            "C")

    fig.suptitle(
        "Reference-configuration results — 5 temporal-encoding settings\n"
        "(UK Biobank respiratory cohort, test n = 18,623)",
        fontsize=10, y=1.02,
    )

    for ext in ("png", "pdf"):
        out = f"figures/figure_2a.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
