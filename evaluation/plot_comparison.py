import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("unified_comparison.csv", usecols=["method", "test_c_index", "test_mean_td_auc", "test_ibs"])

# Keep only rows with at least one metric
df = df.dropna(subset=["test_c_index", "test_mean_td_auc", "test_ibs"], how="all")

methods = df["method"].tolist()
x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

metrics = {
    "C-index": df["test_c_index"].tolist(),
    "Mean TD-AUC": df["test_mean_td_auc"].tolist(),
    "IBS (lower=better)": df["test_ibs"].tolist(),
}

offsets = [-width, 0, width]
colors = ["#4C72B0", "#DD8452", "#55A868"]

for i, (label, values) in enumerate(metrics.items()):
    bars = ax.bar(x + offsets[i], values, width, label=label, color=colors[i], alpha=0.85)
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("Score")
ax.set_title("Model Comparison (Test Set)")
ax.legend()
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("comparison_barplot.png", dpi=150)
print("Saved comparison_barplot.png")
plt.show()
