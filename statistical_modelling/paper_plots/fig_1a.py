import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

FONT_DIR = "fonts/"
for font in fm.findSystemFonts(fontpaths=FONT_DIR):
    fm.fontManager.addfont(font)

plt.rcParams["font.family"] = "CMU Serif"
plt.rcParams["axes.unicode_minus"] = False

results_folder = Path(__file__).parent.parent / "results"
metric_rankings = pd.read_parquet(results_folder / "all_metrics_rankings.parquet")
metric_values = pd.read_parquet(
    results_folder / "all_metrics_aggregated_values.parquet"
)

value_functions = metric_rankings.index.tolist()
ft = 18
label_ft = 15

value_function_names = {
    "baseline_zero": "Bs. Zero",
    "baseline_mean": "Bs. Mean",
    "marginal_bg100": "Marginal",
    "uniform_bg100": "Uniform",
    "jointmarginal_bg100": "Jt. Marg.",
    "filteredconditional_bg100": "Filt. Cond.",
    "conditional_bg100": "Cond.",
    "counterfactual_bg100": "Countf.",
}

x_values = []
y_values = []

for vf in value_functions:
    x = 8 - metric_rankings.loc[vf]["sparsity_mean"]
    x_values.append(x)

    y = metric_values.loc[vf]["insertion_auc_mean"]
    y_values.append(y)

group_colors = {
    "Referenced": "orange",
    "Empirical": "deepskyblue",
    "Counterfactual": "indigo",
}

color_group = {
    "uniform_bg100": "Referenced",
    "marginal_bg100": "Empirical",
    "conditional_bg100": "Empirical",
    "jointmarginal_bg100": "Empirical",
    "baseline_mean": "Referenced",
    "baseline_zero": "Referenced",
    "filteredconditional_bg100": "Counterfactual",
    "counterfactual_bg100": "Counterfactual",
}

text_position = {
    "jointmarginal_bg100": (0, -18),
    "conditional_bg100": (-25, 7),
    "marginal_bg100": (5, 5),
    "uniform_bg100": (-65, -15),
    "filteredconditional_bg100": (-55, -18),
    "counterfactual_bg100": (5, 5),
    "baseline_mean": (5, 5),
    "baseline_zero": (5, 5),
}

fig, ax = plt.subplots(figsize=(5, 5))

# Add text labels
ax.text(
    0.97,
    0.97,
    "Better",
    transform=ax.transAxes,
    fontsize=label_ft,
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="lightgreen",
        alpha=0.3,
        edgecolor="lightgreen",
        linewidth=0.1,
    ),
    zorder=10,
)
ax.text(
    0.03,
    0.03,
    "Worse",
    transform=ax.transAxes,
    fontsize=label_ft,
    ha="left",
    va="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="lightcoral",
        alpha=0.3,
        edgecolor="lightcoral",
        linewidth=0.1,
    ),
    zorder=10,
)

for i in range(len(x_values)):
    ax.scatter(
        x_values[i],
        y_values[i],
        s=30,
        edgecolors="black",
        linewidth=0.1,
        color=group_colors[color_group[value_functions[i]]],
        zorder=2,
    )
    ax.annotate(
        value_function_names[value_functions[i]],
        (x_values[i], y_values[i]),
        xytext=text_position[value_functions[i]],
        textcoords="offset points",
        fontsize=ft,
        zorder=3,
    )

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=group_colors[g],
        markersize=8,
        label=g,
    )
    for g in group_colors.keys()
]

plt.xlabel("Sparsity Ranking", fontsize=ft)
plt.xlim(1.5, 6.5)
plt.xticks([2, 3, 4, 5, 6], ["6", "5", "4", "3", "2"], fontsize=ft)
plt.ylabel("Insertion AUC", fontsize=ft)
plt.ylim(0.7, 1)
plt.yticks(fontsize=ft)

ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

ax.legend(
    handles=legend_elements,
    loc="best",
    fontsize=label_ft,
)

plt.title(
    "Insertion AUC-Sparsity Trade-off", fontsize=ft, fontweight="bold", pad=20, x=0.44
)
plt.tight_layout()

plots_folder = Path(__file__).parent.parent / "plots" / "paper"
os.makedirs(plots_folder, exist_ok=True)
plt.savefig(
    plots_folder / "fig_1a.pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)
plt.close(fig)
