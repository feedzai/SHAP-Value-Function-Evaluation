import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedResultsWrapper

FONT_DIR = "fonts/"
for font in fm.findSystemFonts(fontpaths=FONT_DIR):
    fm.fontManager.addfont(font)

plt.rcParams["font.family"] = "CMU Serif"
plt.rcParams["axes.unicode_minus"] = False

models_folder = Path(__file__).parent.parent / "models"

# Load models. Note that these must be saved previously! All scripts from analysis.
accuracy_model = sm.load(models_folder / "accuracy_logit.pkl")
clarity_model = sm.load(models_folder / "clarity_logit.pkl")
confidence_model = OrderedResultsWrapper.load(
    models_folder / "confidence_orderedlogit.pkl"
)
time_model = sm.load(models_folder / "response_time_quantreg_q975.pkl")

# Getting the CIs at alpha=5% for every variable.
accuracy_cis = accuracy_model.conf_int(alpha=0.05)
clarity_cis = clarity_model.conf_int(alpha=0.05)
confidence_cis = confidence_model.conf_int(alpha=0.05)
time_cis = time_model.conf_int(alpha=0.05)

# Calculate the CI for clarity for the reference group (BS Zero)
exp_cols = clarity_model.params[
    clarity_model.params.index.str.startswith("explainer")
].index
mean = -clarity_model.params[exp_cols].sum()
std = np.sqrt(clarity_model.cov_params().loc[exp_cols, exp_cols].values.sum())
lower_ci = mean - (1 * std)
upper_ci = mean + (1 * std)

# Filter just the explainers from the CIs and create the lower and upper bounds vars
model_order = [
    "Countf.",
    "Cond.",
    "Jt. Marg.",
    "Uniform",
    "Bs. Mean",
    "Marginal",
    "Filt. Cond.",
    "Bs. Zero",
]
accuracy_cis = accuracy_cis[accuracy_cis.index.str.startswith("explainer")]
accuracy_cis.index = model_order
clarity_cis = clarity_cis[clarity_cis.index.str.startswith("explainer")]
clarity_cis = pd.concat(
    [
        clarity_cis,
        pd.DataFrame(
            {
                0: [
                    lower_ci,
                ],
                1: [
                    upper_ci,
                ],
            }
        ),
    ]
)
clarity_cis.index = model_order
confidence_cis = confidence_cis[confidence_cis.index.str.startswith("explainer")]
confidence_cis.index = model_order
time_cis = time_cis[time_cis.index.str.startswith("explainer")]
time_cis.index = model_order

# Create the lower and upper bounds vars
accuracy_lower_bounds = np.exp(accuracy_cis[0].values)
accuracy_upper_bounds = np.exp(accuracy_cis[1].values)
accuracy_mid_point = np.exp(
    (np.log(accuracy_upper_bounds) + np.log(accuracy_lower_bounds)) / 2
)
clarity_lower_bounds = np.exp(clarity_cis[0].values)
clarity_upper_bounds = np.exp(clarity_cis[1].values)
clarity_mid_point = np.exp(
    (np.log(clarity_upper_bounds) + np.log(clarity_lower_bounds)) / 2
)
confidence_lower_bounds = np.exp(confidence_cis[0].values)
confidence_upper_bounds = np.exp(confidence_cis[1].values)
confidence_mid_point = np.exp(
    (np.log(confidence_upper_bounds) + np.log(confidence_lower_bounds)) / 2
)
time_lower_bounds = np.exp(time_cis[0].values)
time_upper_bounds = np.exp(time_cis[1].values)
time_mid_point = np.exp((np.log(time_upper_bounds) + np.log(time_lower_bounds)) / 2)


# Set up color groups
group_colors = {
    "Referenced": "orange",
    "Empirical": "deepskyblue",
    "Counterfactual": "indigo",
}

explanation_group = {
    "Uniform": "Referenced",
    "Marginal": "Empirical",
    "Cond.": "Empirical",
    "Jt. Marg.": "Empirical",
    "Bs. Mean": "Referenced",
    "Bs. Zero": "Referenced",
    "Filt. Cond.": "Counterfactual",
    "Countf.": "Counterfactual",
}

# 2x2 panel with horizontal CIs per algorithm
plot_order = [
    "Bs. Zero",
    "Bs. Mean",
    "Uniform",
    "Marginal",
    "Jt. Marg.",
    "Cond.",
    "Countf.",
    "Filt. Cond.",
]

index_map = [model_order.index(name) for name in plot_order]

accuracy_lower = np.array(accuracy_lower_bounds)[index_map]
accuracy_upper = np.array(accuracy_upper_bounds)[index_map]
accuracy_mid = np.array(accuracy_mid_point)[index_map]

time_lower = np.array(time_lower_bounds)[index_map]
time_upper = np.array(time_upper_bounds)[index_map]
time_mid = np.array(time_mid_point)[index_map]

confidence_lower = np.array(confidence_lower_bounds)[index_map]
confidence_upper = np.array(confidence_upper_bounds)[index_map]
confidence_mid = np.array(confidence_mid_point)[index_map]

clarity_lower = np.array(clarity_lower_bounds)[index_map]
clarity_upper = np.array(clarity_upper_bounds)[index_map]
clarity_mid = np.array(clarity_mid_point)[index_map]

colors = [group_colors[explanation_group[name]] for name in plot_order]

y_positions = np.arange(len(plot_order))[::-1]


def plot_ci(ax, lower, upper, mid, title):
    for idx in range(len(plot_order)):
        ax.plot(
            [lower[idx], upper[idx]],
            [y_positions[idx], y_positions[idx]],
            color=colors[idx],
            linewidth=2,
        )
        ax.scatter(
            mid[idx],
            y_positions[idx],
            s=30,
            edgecolors="black",
            linewidth=0.1,
            color=colors[idx],
            zorder=3,
        )
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_order, fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1.0, 1.8])
    ax.set_xticklabels(["0.5", "1.0", "1.8"])
    ax.minorticks_off()
    ax.set_title(title, fontsize=12, fontweight="normal", pad=4)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5)


fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.6), sharex=True)

plot_ci(
    axes[0, 0],
    time_lower,
    time_upper,
    time_mid,
    "Decision Time",
)

plot_ci(
    axes[1, 0],
    accuracy_lower,
    accuracy_upper,
    accuracy_mid,
    "Accuracy",
)

plot_ci(
    axes[0, 1],
    clarity_lower,
    clarity_upper,
    clarity_mid,
    "Clarity",
)

plot_ci(
    axes[1, 1],
    confidence_lower,
    confidence_upper,
    confidence_mid,
    "Confidence",
)

for ax in axes[:, 1]:
    ax.tick_params(left=False, labelleft=False)

plt.tight_layout(rect=[0, 0, 1, 0.9], w_pad=0.2, h_pad=0.2)

left_center = (axes[0, 0].get_position().x0 + axes[0, 0].get_position().x1) / 2
right_center = (axes[0, 1].get_position().x0 + axes[0, 1].get_position().x1) / 2
separator_x = (left_center + right_center) / 2
bottom_y = axes[1, 0].get_position().y0
top_y = axes[0, 0].get_position().y1

fig.text(
    left_center,
    0.91,
    "Performance",
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
)
fig.text(
    right_center,
    0.91,
    "Feedback",
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
)

# Save as pdf
plots_folder = Path(__file__).parent.parent / "plots" / "paper"
os.makedirs(plots_folder, exist_ok=True)
plt.savefig(plots_folder / "fig_3.pdf", bbox_inches="tight", dpi=300)
