import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT_DIR = "fonts/"
for font in fm.findSystemFonts(fontpaths=FONT_DIR):
    fm.fontManager.addfont(font)

plt.rcParams["font.family"] = "CMU Serif"
plt.rcParams["axes.unicode_minus"] = False

data_folder = Path(__file__).parent.parent.parent / "data"
cross_agreement_df = pd.read_parquet(data_folder / "cross_agreement.parquet")

value_functions = cross_agreement_df.columns.tolist()
cross_agreement = cross_agreement_df.values

vf_order = [
    "marginal_bg100",
    "conditional_bg100",
    "jointmarginal_bg100",
    "baseline_mean",
    "counterfactual_bg100",
    "uniform_bg100",
    "filteredconditional_bg100",
    "baseline_zero",
]

ft = 16

reorder_indices = []
reordered_value_functions = []
for vf in vf_order:
    if vf in value_functions:
        idx = value_functions.index(vf)
        reorder_indices.append(idx)
        reordered_value_functions.append(vf)

cross_agreement = cross_agreement[np.ix_(reorder_indices, reorder_indices)]
value_functions = reordered_value_functions

# Get readable names for value functions
vf_names = {
    "marginal_bg100": "Marginal",
    "conditional_bg100": "Cond.",
    "jointmarginal_bg100": "Jt. Marg.",
    "baseline_mean": "Bs. Mean",
    "baseline_zero": "Bs. Zero",
    "filteredconditional_bg100": "Filt. Cond.",
    "counterfactual_bg100": "Countf.",
    "uniform_bg100": "Uniform",
}
vf_labels = [vf_names.get(vf, vf) for vf in value_functions]

fig, ax = plt.subplots(figsize=(5.5, 5))

# Mask upper triangle (symmetric matrix: show only lower half)
mask = np.triu(np.ones_like(cross_agreement, dtype=bool), k=1)
cross_agreement_masked = np.ma.masked_where(mask, cross_agreement)

# Create heatmap with better colormap
im = ax.imshow(cross_agreement_masked, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

# Set ticks and labels
ax.set_xticks(np.arange(len(value_functions)))
ax.set_yticks(np.arange(len(value_functions)))
ax.set_xticklabels(
    ["M.", "C.", "J.M.", "B.M.", "CF.", "U.", "F.C.", "B.Z."],
    ha="center",
    fontsize=ft,
    rotation=35,
)  # Changed ha to "left"
ax.set_yticklabels(vf_labels, fontsize=ft)

cbar = plt.colorbar(
    im,
    ax=ax,
    fraction=0.045,
    pad=0.08,
)
cbar.set_label("Average Spearman Correlation", fontsize=ft)
cbar.ax.yaxis.set_label_position("left")
cbar.ax.set_yticks([-1, 0, 1])
cbar.ax.tick_params(labelsize=ft)

# Improve appearance
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)

plt.title(
    "Cross Agreement Between Formulations", fontsize=ft, fontweight="bold", pad=20
)

plots_folder = Path(__file__).parent.parent / "plots" / "paper"
os.makedirs(plots_folder, exist_ok=True)
plt.savefig(
    plots_folder / "fig_1c.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
    dpi=300,
)
plt.close(fig)
