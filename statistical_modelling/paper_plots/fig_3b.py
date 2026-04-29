import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedResultsWrapper

FONT_DIR = "fonts/"
for font in fm.findSystemFonts(fontpaths=FONT_DIR):
    fm.fontManager.addfont(font)

plt.rcParams["font.family"] = "CMU Serif"
plt.rcParams["axes.unicode_minus"] = False

# Load intersection models. Saved by analysis/intersection_models.py.
models_folder = Path(__file__).parent.parent / "models"
clarity_model = sm.load(models_folder / "intersection_clarity_logit.pkl")
confidence_model = OrderedResultsWrapper.load(
    models_folder / "intersection_confidence_orderedlogit.pkl"
)

METRIC_COLUMNS = [
    "Sparsity",
    "Contrastivity",
    "Attribution Sensitivity",
    "Deletion AUC",
    "Insertion AUC",
]

# Extract CIs from models
clarity_cis = clarity_model.conf_int(alpha=0.05)
confidence_cis = confidence_model.conf_int(alpha=0.05)

clarity_lower_bounds = np.array([np.exp(clarity_cis.loc[m, 0]) for m in METRIC_COLUMNS])
clarity_upper_bounds = np.array([np.exp(clarity_cis.loc[m, 1]) for m in METRIC_COLUMNS])
clarity_mid_point = np.exp(
    (np.log(clarity_lower_bounds) + np.log(clarity_upper_bounds)) / 2
)

confidence_lower_bounds = np.array(
    [np.exp(confidence_cis.loc[m, 0]) for m in METRIC_COLUMNS]
)
confidence_upper_bounds = np.array(
    [np.exp(confidence_cis.loc[m, 1]) for m in METRIC_COLUMNS]
)
confidence_mid_point = np.exp(
    (np.log(confidence_lower_bounds) + np.log(confidence_upper_bounds)) / 2
)

# Plot labels
order = [
    "Sparsity",
    "Contrastivity",
    "Sensitivity",
    "Deletion AUC",
    "Insertion AUC",
]

ft = 14
colors = ["#f7a399", "#8b5e34", "#ffe169", "#ef6351", "#208b3a"]

text_position = {
    "Sparsity": (5, 5),
    "Contrastivity": (-85, 6),
    "Sensitivity": (5, 5),
    "Deletion AUC": (-90, -15),
    "Insertion AUC": (5, 5),
}

fig, ax = plt.subplots(figsize=(5, 5 * 1.009689922))

ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)

for idx in range(len(order)):
    plt.plot(
        [clarity_lower_bounds[idx], clarity_upper_bounds[idx]],
        [confidence_mid_point[idx], confidence_mid_point[idx]],
        zorder=0,
        color=colors[idx],
        linewidth=2,
    )
    plt.plot(
        [clarity_mid_point[idx], clarity_mid_point[idx]],
        [confidence_lower_bounds[idx], confidence_upper_bounds[idx]],
        zorder=0,
        color=colors[idx],
        linewidth=2,
    )
    plt.scatter(
        clarity_mid_point[idx],
        confidence_mid_point[idx],
        s=30,
        edgecolors="black",
        linewidth=0.1,
        zorder=1,
        color=colors[idx],
    )
    ax.annotate(
        order[idx],
        (clarity_mid_point[idx], confidence_mid_point[idx]),
        xytext=text_position[order[idx]],
        textcoords="offset points",
        fontsize=ft,
    )

plt.yscale("log")
plt.xscale("log")

plt.xlabel("Clarity (Odds Ratio)", fontsize=ft)
plt.xticks([0.8, 1.0, 1.2], ["0.8", "1.0", "1.2"], fontsize=ft)
plt.ylabel("Confidence (Odds Ratio)", fontsize=ft)
plt.yticks([0.8, 1.0, 1.2], ["0.8", "1.0", "1.2"], fontsize=ft)

ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

plt.tick_params(
    axis="both",
    which="minor",
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False,
)

ax.text(
    0.98,
    0.97,
    "Better",
    transform=ax.transAxes,
    fontsize=ft - 1,
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
    0.02,
    0.03,
    "Worse",
    transform=ax.transAxes,
    fontsize=ft - 1,
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

plt.title("Quantitative vs Qualitative Metrics", fontsize=ft, fontweight="bold", pad=15)
plt.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.2, h_pad=0.2)

plots_folder = Path(__file__).parent.parent / "plots" / "paper"
os.makedirs(plots_folder, exist_ok=True)
plt.savefig(
    plots_folder / "fig_3b.pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)
