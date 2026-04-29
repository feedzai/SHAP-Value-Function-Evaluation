import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# DATA PREPARATION
# =============================================================================

# Load data
root_folder = Path(__file__).parent.parent.parent
data = pd.read_parquet(root_folder / "data" / "responses.parquet").reset_index(
    drop=True
)

# Dependant Variables
data_dependant = data[["confidence"]].copy()
data_dependant["clarity"] = data["clarity"]
data_dependant["accuracy"] = data["label"] == data["prediction"]
data_dependant["log_decision_time"] = np.log(
    (data["timestamp_end"] - data["timestamp_start"]).dt.total_seconds()
)

data_dependant["confidence"] = data_dependant["confidence"].map(
    {
        "Strong": 1,
        "Moderate": 0.5,
        "Weak": 0.0,
    }
)
data_dependant["clarity"] = data_dependant["clarity"].map({"Yes": 1, "No": 0})
data_dependant["accuracy"] = data_dependant["accuracy"].astype(int)

# Control Variables
data_control = data[["explanation"]].copy().fillna("None")
data_control["model"] = data["model"]
data_control["domain"] = data["dataset"]
data_control["imbalance_factor"] = data_control["domain"].map(
    data.groupby("dataset")["label"]
    .value_counts()
    .unstack()
    .pipe(lambda x: x[0] / x[1])
)

data_control["score"] = data["model_score"]
data_control["score_percentile"] = data["score_percentile"]
data_control["score_error"] = np.abs(data["label"] - data["model_score"])
data_control["score_entropy"] = -(
    data["model_score"] * np.log(data["model_score"])
    + (1 - data["model_score"]) * np.log(1 - data["model_score"])
)

data_control["log_count_session"] = np.log(1 + data["case_counter"])
data_control["log_count_total"] = np.log(
    2 + data.sort_values(by="timestamp_start").groupby("user").cumcount()
)
data_control["log_count_domain"] = np.log(
    2 + data.sort_values(by="timestamp_start").groupby(["user", "dataset"]).cumcount()
)

data_control["knows_shapley"] = data["shapley_understanding"]
data_control["knows_ml"] = data["ml_understanding"]
data_control["knows_domain"] = data["domain_knowledge"]

# Mixed Effects
data_mixed = data[["user"]]


# BASIC SUMMARIES AND PLOTS
# =============================================================================
# Create plots directory if it doesn't exist
eda_plots_folder = root_folder / "statistical_modelling" / "plots" / "eda"
os.makedirs(eda_plots_folder, exist_ok=True)

# For categorical control variables
for category in [
    "explanation",
    "knows_shapley",
    "knows_ml",
    "knows_domain",
    "model",
    "domain",
]:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()

    # 1. KDE Plot
    sns.kdeplot(
        ax=axes[0],
        data=data_dependant,
        x="log_decision_time",
        hue=data_control[category],
        fill=True,
        common_norm=False,
        palette="deep",
        alpha=0.3,
        linewidth=2.5,
    )
    axes[0].set_title(f"Log Decision Time by {category.capitalize()}", fontsize=14)
    sns.move_legend(axes[0], "upper right")

    # 2-4. Bar Plots for Metrics
    for i, metric in enumerate(["accuracy", "confidence", "clarity"]):
        sns.barplot(
            ax=axes[i + 1],
            x=data_control[category],
            y=data_dependant[metric],
            hue=data_control[category],
            palette="deep",
            errorbar=("se", 1.96),
            capsize=0.1,
            legend=False,
        )

        labels = [label.get_text() for label in axes[i + 1].get_xticklabels()]
        short_labels = [
            f"{label[:8]}..." if len(label) > 10 else label for label in labels
        ]
        axes[i + 1].set_xticks(axes[i + 1].get_xticks())
        axes[i + 1].set_xticklabels(short_labels)

        axes[i + 1].set_title(
            f"{metric.capitalize()} by {category.capitalize()}", fontsize=14
        )
        axes[i + 1].set_xlabel(category.capitalize(), fontsize=12)
        axes[i + 1].set_ylabel(metric.capitalize(), fontsize=12)

    plt.suptitle(f"Analysis Overview: {category.upper()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(eda_plots_folder / f"eda_{category}.png")


# For numerical control variables
for category in [
    "score",
    "score_percentile",
    "score_error",
    "score_entropy",
    "log_count_session",
    "log_count_total",
    "log_count_domain",
]:
    raw_bins = pd.qcut(data_control[category], q=10)
    bin_means = data_control[category].groupby(raw_bins, observed=True).mean()

    map_dict = {interval: f"{mean:.2f}" for interval, mean in bin_means.items()}
    binned_series = raw_bins.map(map_dict)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()

    # 1. KDE Plot
    sns.kdeplot(
        ax=axes[0],
        data=data_dependant,
        x="log_decision_time",
        hue=binned_series,
        fill=True,
        common_norm=False,
        palette="viridis",
        alpha=0.3,
        linewidth=2.5,
    )
    axes[0].set_title(f"Log Decision Time by {category.capitalize()}", fontsize=14)
    sns.move_legend(axes[0], "upper right")

    # 2-4. Bar Plots for Metrics
    for i, metric in enumerate(["accuracy", "confidence", "clarity"]):
        sns.barplot(
            ax=axes[i + 1],
            x=binned_series,
            y=data_dependant[metric],
            hue=binned_series,
            palette="viridis",
            errorbar=("se", 1.96),
            capsize=0.1,
            legend=False,
        )

        axes[i + 1].set_title(
            f"{metric.capitalize()} by {category.capitalize()}", fontsize=14
        )
        axes[i + 1].set_xlabel(f"{category.capitalize()} (Bins)", fontsize=12)
        axes[i + 1].set_ylabel(metric.capitalize(), fontsize=12)

    plt.suptitle(
        f"Numerical Analysis: {category.upper()} (Binned into Deciles)", fontsize=16
    )
    plt.tight_layout()
    plt.savefig(eda_plots_folder / f"eda_num_{category}.png")
