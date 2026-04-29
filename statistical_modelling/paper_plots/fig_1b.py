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

# Load models. Note that these must be saved previously! All scripts from analysis.
models_folder = Path(__file__).parent.parent / "models"

clarity_model = sm.load(models_folder / "clarity_logit.pkl")
confidence_model = OrderedResultsWrapper.load(
    models_folder / "confidence_orderedlogit.pkl"
)


# Calculate the mean and std for clarity for the reference group (BS Zero)
exp_cols = clarity_model.params[
    clarity_model.params.index.str.startswith("explainer")
].index
mean = np.exp(-clarity_model.params[exp_cols].sum())
std = np.sqrt(clarity_model.cov_params().loc[exp_cols, exp_cols].values.sum())

model_order = [
    "counterfactual_bg100",
    "conditional_bg100",
    "jointmarginal_bg100",
    "uniform_bg100",
    "baseline_mean",
    "marginal_bg100",
    "filteredconditional_bg100",
    "baseline_zero",
]

explainers = model_order

# Get the standard errors in Odds space (Delta Method)
clarity_means = np.exp(clarity_model.params[exp_cols])
clarity_means = pd.concat([clarity_means, pd.Series({"baseline_zero": mean})])
clarity_se = clarity_means * clarity_model.bse[exp_cols]
clarity_se.drop("baseline_zero", inplace=True)
clarity_se = pd.concat([clarity_se, pd.Series({"baseline_zero": std})])
clarity_means.index = model_order
clarity_se.index = model_order

exp_cols = confidence_model.params[
    confidence_model.params.index.str.startswith("explainer")
].index
confidence_means = np.exp(confidence_model.params[exp_cols])
confidence_se = confidence_means * confidence_model.bse[exp_cols]
confidence_means.index = model_order
confidence_se.index = model_order


y_values_1 = (clarity_means - clarity_se).values
y_values_2 = (clarity_means + clarity_se).values

y_values_mean = np.exp((np.log(y_values_1) + np.log(y_values_2)) / 2)

x_values_1 = (confidence_means - confidence_se).values
x_values_2 = (confidence_means + confidence_se).values
x_values_mean = np.exp((np.log(x_values_1) + np.log(x_values_2)) / 2)


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
    "jointmarginal_bg100": (-76, 5),
    "conditional_bg100": (5, -26),
    "marginal_bg100": (-75, 5),
    "uniform_bg100": (5, 5),
    "filteredconditional_bg100": (5, 5),
    "counterfactual_bg100": (-40, 23),
    "baseline_mean": (-77, -16),
    "baseline_zero": (5, -16),
}

fig, ax = plt.subplots(figsize=(5, 5))

ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5, zorder=1)
ax.axvline(x=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.5, zorder=1)

ft = 18
label_ft = 15

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

for idx in range(len(explainers)):
    plt.plot(
        [x_values_1[idx], x_values_2[idx]],
        [y_values_mean[idx], y_values_mean[idx]],
        color=group_colors[color_group[explainers[idx]]],
        zorder=2,
        linewidth=2,
    )
    plt.plot(
        [x_values_mean[idx], x_values_mean[idx]],
        [y_values_1[idx], y_values_2[idx]],
        color=group_colors[color_group[explainers[idx]]],
        zorder=2,
        linewidth=2,
    )
    plt.scatter(
        x_values_mean[idx],
        y_values_mean[idx],
        s=30,
        edgecolors="black",
        linewidth=0.1,
        color=group_colors[color_group[explainers[idx]]],
        zorder=3,
    )
    ax.annotate(
        value_function_names.get(explainers[idx], explainers[idx]),
        (x_values_mean[idx], y_values_mean[idx]),
        xytext=text_position[explainers[idx]],
        textcoords="offset points",
        fontsize=ft,
        zorder=4,
    )

plt.yscale("log")
plt.xscale("log")

plt.xlabel("Confidence (Odds Ratio)", fontsize=ft)
plt.xticks([1.0, 1.3, 1.8], ["1.0", "1.3", "1.8"], fontsize=ft)
plt.ylabel("Clarity (Odds Ratio)", fontsize=ft)
plt.yticks([0.6, 1.0, 1.8], ["0.6", "1.0", "1.8"], fontsize=ft)

ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Remove ticks from plt
plt.tick_params(
    axis="both",
    which="minor",
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False,
)

plt.title(
    "User Confidence-Clarity Trade-off", fontsize=ft, fontweight="bold", pad=20, x=0.45
)
plt.tight_layout()

plots_folder = Path(__file__).parent.parent / "plots" / "paper"
os.makedirs(plots_folder, exist_ok=True)
plt.savefig(
    plots_folder / "fig_1b.pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)
