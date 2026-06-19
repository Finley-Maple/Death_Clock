"""Figure 2b: Grouped barplot of TD-AUC by prediction horizon.

X-axis: 3 horizon groups (9.5 yr, 15.4 yr, 19.7 yr)
Within each group: 5 bars (one per reference configuration), colour-coded.

Run from evaluation/ directory:
    python plot_horizon_bars.py

Outputs: figures/figure_2b.png, figures/figure_2b.pdf
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})

os.makedirs("figures", exist_ok=True)

COLORS = {
    "S1": "#7f7f7f",
    "S2": "#5B8DB8",
    "S3": "#E07B54",
    "S4": "#4C9B8E",
    "S3F": "#7B5EA7",
}

HORIZONS_DAYS = [3471, 5620, 7213]
HORIZON_LABELS = ["9.5 yr", "15.4 yr", "19.7 yr"]

REF_CONFIGS = [
    {
        "label": "S1 Baseline CoxPH",
        "color": COLORS["S1"],
        "file":  "benchmarking_results/benchmarking_results.json",
    },
    {
        "label": "S2 Delphi",
        "color": COLORS["S2"],
        "file":  "delphi_test_results.json",
    },
    {
        "label": "S3 Prose+Qwen",
        "color": COLORS["S3"],
        "file":  "embedding_results/s3_qwen_concat_results.json",
    },
    {
        "label": "S3-Full Qwen",
        "color": COLORS["S3F"],
        "file":  "embedding_results/s5_qwen_results.json",
    },
    {
        "label": "S4 Decoupled+Qwen",
        "color": COLORS["S4"],
        "file":  "embedding_results/s4_qwen_concat_results.json",
    },
]


def load_refs():
    rows = []
    for cfg in REF_CONFIGS:
        with open(cfg["file"]) as f:
            d = json.load(f)
        td = d["splits"]["test"]["td_auc"]
        vals = [td[str(h)] for h in HORIZONS_DAYS]
        rows.append({
            "label":  cfg["label"],
            "color":  cfg["color"],
            "values": vals,
        })
    return rows


def main():
    rows = load_refs()
    n_methods  = len(rows)
    n_horizons = len(HORIZONS_DAYS)

    bar_width  = 0.14
    group_gap  = 0.9
    x_centers  = np.arange(n_horizons) * group_gap

    fig, ax = plt.subplots(figsize=(9, 4.5))

    offsets = np.linspace(
        -(n_methods - 1) / 2 * bar_width,
         (n_methods - 1) / 2 * bar_width,
        n_methods,
    )

    for i, (row, offset) in enumerate(zip(rows, offsets)):
        xpos = x_centers + offset
        ax.bar(xpos, row["values"], width=bar_width,
               color=row["color"], edgecolor="white", linewidth=0.5,
               label=row["label"])

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(HORIZON_LABELS, fontsize=11)
    ax.set_xlabel("Prediction horizon", fontsize=10)
    ax.set_ylabel("TD-AUC  (↑)", fontsize=10)
    all_vals = [v for r in rows for v in r["values"]]
    lo = min(all_vals) - 0.02
    hi = max(all_vals) + 0.04
    ax.set_ylim(lo, hi)

    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.88,
              edgecolor="lightgray", ncol=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = f"figures/figure_2b.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
