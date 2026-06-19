"""Publication-quality comparison figure for all-cause mortality prediction.

All values are loaded from JSON result files — no hardcoded numbers.

Run from evaluation/ directory:
    python plot_comparison.py

Outputs: figures/figure_main.png, figures/figure_main.pdf
"""

import os
import json
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})

os.makedirs("figures", exist_ok=True)

# ── Result file mapping ───────────────────────────────────────────────────────
# Maps the "method" string in unified_comparison.json to:
#   label      : short display label (2-line for bar x-axis)
#   legend     : full label for line-chart legend
#   color      : hex color
#   horizon_file: path to per-method JSON with splits.test.td_auc  (optional;
#                 falls back to test_auc_by_horizon in unified_comparison.json)
#   horizon_fmt : "standard" (splits.test.td_auc) | "delphi" (same key, different root)
# ── Colour palette for the 5-setting framework ─────────────────────────────
# Desaturated Nature/Lancet palette — one hue per setting, lighter shade for BCB.
# S1 = grey (baseline, no encoder), S2 = steel blue (Delphi), S3–S3F = warm→cool.
_S3  = "#E07B54"   # burnt orange   — Setting 3 prose
_S4  = "#4C9B8E"   # teal           — Setting 4 trajectory
_S3F = "#7B5EA7"   # muted purple   — Setting 3-Full (full prose)
_BCB_ALPHA = "BB" # hex alpha for BCB variants (slightly lighter)

METHOD_CONFIG = {
    # ── Reference rows (Settings 1 & 2) ─────────────────────────────────────
    "Delphi (test)": {
        "label":        "S2 · Delphi\n(zero-shot)",
        "legend":       "Setting 2 — Delphi (zero-shot, sequential ordinal)",
        "color":        "#5B8DB8",   # steel blue
        "horizon_file": "delphi_test_results.json",
        "horizon_fmt":  "delphi",
    },
    "Benchmarking (CoxPH)": {
        "label":        "S1 · Baseline\nCoxPH",
        "legend":       "Setting 1 — Baseline CoxPH (39 features, cross-sectional)",
        "color":        "#7f7f7f",   # neutral grey
        "horizon_file": "benchmarking_results/benchmarking_results.json",
        "horizon_fmt":  "standard",
        "is_baseline":  True,        # draws dotted reference line in bar panels
    },

    # ── Setting 3 — Isolated temporal prose ──────────────────────────────────
    # (previously "Text Embedding + CoxPH" = Qwen + concat; kept for compat)
    "Text Embedding + CoxPH": {
        "label":        "S3 · Qwen\n+ concat",
        "legend":       "Setting 3 — Qwen3-8B prose + concat (original)",
        "color":        _S3,
        "horizon_file": "embedding_results/text_embedding_results.json",
        "horizon_fmt":  "standard",
    },
    "S3 · Qwen3-8B · concat": {
        "label":        "S3 · Qwen\n+ concat",
        "legend":       "Setting 3 — Qwen3-8B prose + concat",
        "color":        _S3,
        "horizon_file": "embedding_results/s3_qwen_concat_results.json",
        "horizon_fmt":  "standard",
    },
    "S3 · Qwen3-8B · sum": {
        "label":        "S3 · Qwen\n+ sum",
        "legend":       "Setting 3 — Qwen3-8B prose + summation",
        "color":        _S3 + "CC",
        "horizon_file": "embedding_results/s3_qwen_sum_results.json",
        "horizon_fmt":  "standard",
    },
    "S3 · BCB · concat": {
        "label":        "S3 · BCB\n+ concat",
        "legend":       "Setting 3 — Bio_ClinicalBERT prose + concat",
        "color":        _S3 + _BCB_ALPHA,
        "horizon_file": "embedding_results/s3_bcb_concat_results.json",
        "horizon_fmt":  "standard",
    },
    "S3 · BCB · sum": {
        "label":        "S3 · BCB\n+ sum",
        "legend":       "Setting 3 — Bio_ClinicalBERT prose + summation",
        "color":        _S3 + "88",
        "horizon_file": "embedding_results/s3_bcb_sum_results.json",
        "horizon_fmt":  "standard",
    },

    # ── Setting 4 — Decoupled time & event ───────────────────────────────────
    # (previously "Trajectory Embedding + CoxPH" = Qwen + concat; kept for compat)
    "Trajectory Embedding + CoxPH": {
        "label":        "S4 · Qwen\n+ concat",
        "legend":       "Setting 4 — Qwen3-8B trajectory + concat (original)",
        "color":        _S4,
        "horizon_file": "embedding_results/trajectory_embedding_results.json",
        "horizon_fmt":  "standard",
    },
    "S4 · Qwen3-8B · concat": {
        "label":        "S4 · Qwen\n+ concat",
        "legend":       "Setting 4 — Qwen3-8B decoupled + concat",
        "color":        _S4,
        "horizon_file": "embedding_results/s4_qwen_concat_results.json",
        "horizon_fmt":  "standard",
    },
    "S4 · Qwen3-8B · sum": {
        "label":        "S4 · Qwen\n+ sum",
        "legend":       "Setting 4 — Qwen3-8B decoupled + summation",
        "color":        _S4 + "CC",
        "horizon_file": "embedding_results/s4_qwen_sum_results.json",
        "horizon_fmt":  "standard",
    },
    "S4 · BCB · concat": {
        "label":        "S4 · BCB\n+ concat",
        "legend":       "Setting 4 — Bio_ClinicalBERT decoupled + concat",
        "color":        _S4 + _BCB_ALPHA,
        "horizon_file": "embedding_results/s4_bcb_concat_results.json",
        "horizon_fmt":  "standard",
    },
    "S4 · BCB · sum": {
        "label":        "S4 · BCB\n+ sum",
        "legend":       "Setting 4 — Bio_ClinicalBERT decoupled + summation",
        "color":        _S4 + "88",
        "horizon_file": "embedding_results/s4_bcb_sum_results.json",
        "horizon_fmt":  "standard",
    },

    # ── S3-Full — Full clinical prose (standalone, no raw features) ─────────
    # (previously labelled S5; kept compat keys for existing JSON data files)
    "Biomarker Text + CoxPH (emb only)": {
        "label":        "S3-Full · Qwen\n(no raw)",
        "legend":       "S3-Full — Qwen3-8B full prose, no raw features (original)",
        "color":        _S3F,
    },
    "S5 · Qwen3-8B": {
        "label":        "S3-Full · Qwen\n(no raw)",
        "legend":       "S3-Full — Qwen3-8B full prose, no raw features",
        "color":        _S3F,
        "horizon_file": "embedding_results/s5_qwen_results.json",
        "horizon_fmt":  "standard",
    },
    "S3-Full · Qwen3-8B": {
        "label":        "S3-Full · Qwen\n(no raw)",
        "legend":       "S3-Full — Qwen3-8B full prose, no raw features",
        "color":        _S3F,
        "horizon_file": "embedding_results/s5_qwen_results.json",
        "horizon_fmt":  "standard",
    },
    "S5 · BCB": {
        "label":        "S3-Full · BCB\n(no raw)",
        "legend":       "S3-Full — Bio_ClinicalBERT full prose, no raw features",
        "color":        _S3F + _BCB_ALPHA,
        "horizon_file": "embedding_results/s5_bcb_results.json",
        "horizon_fmt":  "standard",
    },
    "S3-Full · BCB": {
        "label":        "S3-Full · BCB\n(no raw)",
        "legend":       "S3-Full — Bio_ClinicalBERT full prose, no raw features",
        "color":        _S3F + _BCB_ALPHA,
        "horizon_file": "embedding_results/s5_bcb_results.json",
        "horizon_fmt":  "standard",
    },

    # ── Legacy fusion method (kept for backward compat) ──────────────────────
    "Fusion (text+traj emb + baseline)": {
        "label":        "Fusion\n(text+traj)",
        "legend":       "Fusion — text+traj PCA + baseline (original)",
        "color":        "#9467bd",
    },
}

# Fallback color cycle for methods not in METHOD_CONFIG
_COLOR_CYCLE = itertools.cycle([
    "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
])

MARKERS = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", "*"])

# ── Data loading ─────────────────────────────────────────────────────────────

def _load_horizon_tdauc(cfg, unified_entry):
    """Return {horizon_days_str: float} dict, or None if unavailable."""
    # 1. Try per-method file (most accurate, has full precision)
    fpath = cfg.get("horizon_file")
    if fpath and os.path.exists(fpath):
        with open(fpath) as f:
            src = json.load(f)
        fmt = cfg.get("horizon_fmt", "standard")
        if fmt == "delphi":
            td = src.get("splits", {}).get("test", {}).get("td_auc")
        else:
            td = src.get("splits", {}).get("test", {}).get("td_auc")
        if td:
            return td
    # 2. Fall back to unified_comparison.json field
    return unified_entry.get("test_auc_by_horizon")


def load_results(unified_path="unified_comparison.json"):
    with open(unified_path) as f:
        unified = json.load(f)

    methods = []
    all_horizons = None  # days, sorted

    for entry in unified:
        name = entry["method"]
        cfg = METHOD_CONFIG.get(name, {})

        color = cfg.get("color") or next(_COLOR_CYCLE)
        td_horizon = _load_horizon_tdauc(cfg, entry)

        if td_horizon and all_horizons is None:
            all_horizons = sorted(int(k) for k in td_horizon.keys())

        methods.append({
            "name":         name,
            "label":        cfg.get("label", name),
            "legend":       cfg.get("legend", name),
            "color":        color,
            "is_baseline":  cfg.get("is_baseline", False),
            "c_index":      entry.get("test_c_index"),
            "mean_tdauc":   entry.get("test_mean_td_auc"),
            "ibs":          entry.get("test_ibs"),
            "td_horizon":   td_horizon,  # {str(days): float} or None
        })

    return methods, all_horizons


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _bar_panel(ax, methods, field, ylabel, title, letter, ylim,
               lower_better=False, show_baseline_ref=True):
    values = [m[field] for m in methods]
    colors = [m["color"] for m in methods]
    labels = [m["label"] for m in methods]
    x = np.arange(len(methods))

    bars = ax.bar(x, values, 0.65, color=colors, edgecolor="white", linewidth=0.6)

    span = ylim[1] - ylim[0]
    for bar, val in zip(bars, values):
        if val is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + span * 0.012,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=7,
        )

    if show_baseline_ref:
        for m in methods:
            if m["is_baseline"] and m[field] is not None:
                ax.axhline(m[field], color=m["color"], linestyle=":",
                           linewidth=1.3, alpha=0.7, zorder=3)
                break

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.text(-0.14, 1.05, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _horizon_panel(ax, methods, horizons_days, letter):
    horizons_yr = [d / 365.25 for d in horizons_days]
    xtick_labels = [f"{y:.1f} yr" for y in horizons_yr]
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", "*"])

    for m in methods:
        td = m["td_horizon"]
        if not td:
            continue
        vals = [td.get(str(d)) or td.get(d) for d in horizons_days]
        if any(v is None for v in vals):
            continue
        ax.plot(horizons_yr, vals,
                marker=next(marker_cycle),
                color=m["color"],
                linewidth=1.9, markersize=6,
                label=m["legend"])

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
               alpha=0.4, zorder=1)

    ax.set_xlabel("Prediction horizon (years)", fontsize=10)
    ax.set_ylabel("TD-AUC", fontsize=10)
    ax.text(-0.04, 1.05, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.set_xticks(horizons_yr)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.legend(fontsize=8, loc="upper left", ncol=2,
              framealpha=0.85, edgecolor="lightgray")
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    methods, horizons_days = load_results()

    # Auto-compute ylims with a little padding
    def _ylim(values, pad=0.2, lower_better=False):
        vals = [v for v in values if v is not None]
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 1e-4)
        if lower_better:
            return (lo - span * pad, hi + span * pad * 3)
        return (lo - span * pad, hi + span * pad * 3)

    c_vals      = [m["c_index"] for m in methods]
    tdauc_vals  = [m["mean_tdauc"] for m in methods]
    ibs_vals    = [m["ibs"] for m in methods]

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.65, wspace=0.40)

    ax_c   = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[0, 1])
    ax_ibs = fig.add_subplot(gs[0, 2])
    ax_hor = fig.add_subplot(gs[1, :])

    _bar_panel(ax_c,   methods, "c_index",    "Harrell C-index",    "Harrell C-index",              "A", _ylim(c_vals))
    _bar_panel(ax_auc, methods, "mean_tdauc", "Mean TD-AUC",        "Mean TD-AUC",                  "B", _ylim(tdauc_vals))
    _bar_panel(ax_ibs, methods, "ibs",        "IBS",                "Integrated Brier Score",       "C", _ylim(ibs_vals, lower_better=True), lower_better=True)

    if horizons_days:
        _horizon_panel(ax_hor, methods, horizons_days, "D")
    else:
        ax_hor.text(0.5, 0.5, "No per-horizon TD-AUC data available",
                    ha="center", va="center", transform=ax_hor.transAxes, fontsize=11)
        ax_hor.axis("off")

    out_png = "figures/figure_main.png"
    out_pdf = "figures/figure_main.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}, {out_pdf}")


if __name__ == "__main__":
    main()
