"""Illustrate the all-cause mortality prediction task (Setting overview, Figure 0).

Layout: two panels side-by-side.
  Left  — Patient timeline schematic: ICD events before age 60 (feature window),
           outcomes after age 60 (follow-up).
  Right — Illustrative survival curves for high / medium / low risk patients,
           showing the model output S(t|x).

Run from evaluation/ directory:
    python plot_task_illustration.py
Outputs: figures/figure_0.png, figures/figure_0.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

os.makedirs("figures", exist_ok=True)

# ── Colour palette (matches paper palette) ──────────────────────────────────
C_FEATURE  = "#F0F4FA"   # light blue — feature window background
C_FOLLOWUP = "#FFF5F0"   # light orange — follow-up background
C_EVENT    = "#5B8DB8"   # blue dots  — ICD events
C_DEATH    = "#C0392B"   # red — death marker
C_CENSOR   = "#7B7B7B"   # grey — censoring marker
C_CUTOFF   = "#2C3E50"   # dark — age-60 line
C_S1       = "#C0392B"   # high-risk survival curve
C_S2       = "#E07B54"   # medium-risk
C_S3       = "#4C9B8E"   # low-risk
C_ARROW    = "#5B8DB8"

AGE_MIN, AGE_MAX = 38, 82
CUTOFF = 60

# Three illustrative patients (age of events before 60, outcome age, alive?)
PATIENTS = [
    {
        "label": "Patient A",
        "events": [41.2, 45.8, 50.3, 54.1, 57.9],
        "outcome_age": 67.5,
        "alive": False,
        "risk": "High risk",
    },
    {
        "label": "Patient B",
        "events": [43.5, 52.0, 58.6],
        "outcome_age": 76.2,
        "alive": True,
        "risk": "Medium risk",
    },
    {
        "label": "Patient C",
        "events": [46.0, 49.7],
        "outcome_age": 79.0,
        "alive": True,
        "risk": "Low risk",
    },
]

Y_POSITIONS = [3.0, 2.0, 1.0]   # y-coordinates for the three timelines
Y_LABEL = 0.12                   # y for x-axis label row


def survival_curve(t, lam):
    """Exponential survival for illustration."""
    return np.exp(-lam * t)


def main():
    fig, (ax_time, ax_surv) = plt.subplots(
        1, 2,
        figsize=(13, 4.2),
        gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.38},
    )

    # ── LEFT PANEL: timeline ──────────────────────────────────────────────────
    ax = ax_time

    # Background shading
    ax.axvspan(AGE_MIN, CUTOFF,  color=C_FEATURE,  alpha=1.0, zorder=0)
    ax.axvspan(CUTOFF,  AGE_MAX, color=C_FOLLOWUP, alpha=1.0, zorder=0)

    # Age-60 cutoff line
    ax.axvline(CUTOFF, color=C_CUTOFF, linewidth=1.8, linestyle="--", zorder=3)
    ax.text(CUTOFF, 3.75, "Age 60\ncutoff",
            ha="center", va="bottom", fontsize=8.5,
            color=C_CUTOFF, fontweight="bold")

    # Patient timelines
    for pat, y in zip(PATIENTS, Y_POSITIONS):
        # Horizontal timeline bar (grey backbone)
        ax.plot([AGE_MIN + 0.5, pat["outcome_age"]], [y, y],
                color="#CCCCCC", linewidth=2.0, solid_capstyle="round", zorder=1)

        # ICD events (blue dots)
        for ev in pat["events"]:
            ax.plot(ev, y, "o", color=C_EVENT, markersize=8, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)

        # Outcome marker
        if pat["alive"]:
            ax.plot(pat["outcome_age"], y, "o", color=C_CENSOR,
                    markersize=9, zorder=5,
                    markeredgecolor=C_CENSOR, markeredgewidth=2.0,
                    markerfacecolor="none")
        else:
            ax.plot(pat["outcome_age"], y, "x", color=C_DEATH,
                    markersize=11, zorder=5, mew=2.5)

        # Patient label (left)
        ax.text(AGE_MIN - 0.2, y, pat["label"],
                ha="right", va="center", fontsize=9.5, fontweight="bold",
                color="#2C3E50")

    # x-axis label ticks
    ticks = [40, 45, 50, 55, 60, 65, 70, 75, 80]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9)
    ax.set_xlabel("Age (years)", fontsize=10)

    ax.set_xlim(AGE_MIN - 1.5, AGE_MAX + 0.5)
    ax.set_ylim(0.45, 4.2)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)

    # Zone labels
    ax.text((AGE_MIN + CUTOFF) / 2, 4.0,
            "Feature extraction window\n(ICD events, biomarkers, demographics)",
            ha="center", va="top", fontsize=8.5, color="#34495E",
            style="italic")
    ax.text((CUTOFF + AGE_MAX) / 2, 4.0,
            "Prediction target\n(all-cause mortality after age 60)",
            ha="center", va="top", fontsize=8.5, color="#34495E",
            style="italic")

    # Legend items for dot types
    legend_elements = [
        mpatches.Patch(color=C_FEATURE,  label="Feature window"),
        mpatches.Patch(color=C_FOLLOWUP, label="Follow-up"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C_EVENT, markersize=8,
                   markeredgecolor="white", label="ICD event"),
        plt.Line2D([0], [0], marker="x", color=C_DEATH,
                   markersize=9, mew=2.5, linestyle="None", label="Death"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="none", markeredgewidth=2.0,
                   markeredgecolor=C_CENSOR, markersize=9,
                   linestyle="None", label="Censored"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right",
              framealpha=0.88, edgecolor="lightgray", ncol=2)

    ax.set_title("All-cause mortality prediction task\n"
                 "(UK Biobank respiratory cohort, $n$=124,158; event rate 26.2%)",
                 fontsize=10, fontweight="bold", pad=6)

    ax.text(-0.08, 1.04, "A", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

    # ── RIGHT PANEL: survival curves ──────────────────────────────────────────
    ax2 = ax_surv

    t = np.linspace(0, 20, 300)   # years after age 60
    lambdas = {"High risk":   0.18,
               "Medium risk": 0.06,
               "Low risk":    0.025}
    colors  = {"High risk":   C_S1,
               "Medium risk": C_S2,
               "Low risk":    C_S3}

    for label, lam in lambdas.items():
        s = survival_curve(t, lam)
        ax2.plot(t, s, color=colors[label], linewidth=2.2, label=label)

    ax2.axhline(0.5, color="#AAAAAA", linewidth=0.8, linestyle=":", alpha=0.7)
    ax2.set_xlabel("Years after age 60", fontsize=10)
    ax2.set_ylabel("Survival probability $S(t \mid x)$", fontsize=10)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.88, edgecolor="lightgray")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_title("Model output: predicted survival curve\n"
                  "(CoxPH head, evaluated with C-index / TD-AUC / IBS)",
                  fontsize=10, fontweight="bold", pad=6)

    ax2.text(-0.18, 1.04, "B", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="top")

    fig.suptitle(
        "Prediction task: temporal EHR encoding $\\rightarrow$ mortality risk stratification",
        fontsize=10.5, y=1.03,
    )

    for ext in ("png", "pdf"):
        out = f"figures/figure_0.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
