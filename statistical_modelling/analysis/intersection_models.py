"""
Summary:
- Intersection models replace the explainer categorical variable with
  per-instance quantitative explanation metrics (Sparsity, Contrastivity,
  Attribution Sensitivity, Deletion AUC, Insertion AUC).
- Two models are fitted:
    1. Logistic regression on Clarity (binary).
    2. Ordered logistic regression on Confidence (ordinal: Weak/Moderate/Strong).
- Quantitative metrics are clipped at the 5th/95th percentile per dataset,
  then standardised per dataset before entering the design matrix.
- Rows without an explanation (and therefore without metrics) are dropped.

To note:
    - High multicollinearity needs correcting, dropping correlated covariates
    - The script only retains relatively significant and uncorrelated factors
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
results_folder = Path(__file__).parent.parent / "results"
data = pd.read_parquet(results_folder / "responses.parquet").reset_index(drop=True)

# Remove cases without an explainer (cannot compute metrics)
data = data[~data["explanation"].isnull()]

# Exclude Fraud Dataset and Specific user
data = data[data["dataset"] != "Fraud Dataset"]
data = data[data["user"] != "6bb6f838-322d-4a01-ac67-0dfc199e9350"]

# Identify users with at least 20 reviews
min_obs = 20
user_counts = data["user"].value_counts()
valid_users = user_counts[user_counts >= min_obs].index

# Filter the dataset
data = data[data["user"].isin(valid_users)].copy().reset_index()

# Load per-instance quantitative metrics
metrics = pd.read_parquet(results_folder / "per_instance_results.parquet")
metrics = metrics.rename(
    columns={"Unnamed: 0": "instance_id", "explanation_method": "explanation"}
)
metrics["instance_id"] = metrics["instance_id"].astype(str)

METRIC_COLUMNS = [
    "Sparsity",
    "Contrastivity",
    "Attribution Sensitivity",
    "Deletion AUC",
    "Insertion AUC",
]

# Merge quantitative metrics into responses
data = data.merge(
    metrics[["instance_id", "dataset", "model", "explanation"] + METRIC_COLUMNS],
    on=["instance_id", "dataset", "model", "explanation"],
    how="inner",
)
data.dropna(subset=METRIC_COLUMNS, inplace=True)
data.reset_index(drop=True, inplace=True)


# Shared helpers
# =============================================================================


def standardize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def clip_and_standardize_metrics(data_in, clip_percentile=0.05):
    """Clip metrics at given percentile per dataset, then standardise per dataset."""
    out = data_in.copy()
    for metric in METRIC_COLUMNS:
        for dataset in out["dataset"].unique():
            mask = out["dataset"] == dataset
            top = out.loc[mask, metric].quantile(1 - clip_percentile)
            bottom = out.loc[mask, metric].quantile(clip_percentile)
            out.loc[mask, metric] = out.loc[mask, metric].clip(lower=bottom, upper=top)

        mean_by_dataset = out.groupby("dataset")[metric].transform("mean")
        std_by_dataset = out.groupby("dataset")[metric].transform("std")
        out[metric] = (out[metric] - mean_by_dataset) / std_by_dataset
    return out


def build_design_matrix(data_in):
    """Build the common design matrix (without explainer dummies)."""
    X = pd.DataFrame(index=data_in.index)

    # === Categorical controls ===
    X["model"] = data_in["model"]
    X["domain"] = data_in["dataset"]
    X["knows_domain"] = data_in["domain_knowledge"]
    X["analyst"] = data_in[["user"]]

    X = pd.get_dummies(X, drop_first=True, dtype=float)

    # === Numerical controls ===
    X["score_error"] = standardize(np.abs(data_in["label"] - data_in["model_score"]))
    X["score_entropy"] = standardize(
        -data_in["model_score"] * np.log(data_in["model_score"])
        - (1 - data_in["model_score"]) * np.log(1 - data_in["model_score"])
    )
    X["log_count_session"] = standardize(np.log(1 + data_in["case_counter"]))
    X["log_count_total"] = standardize(
        np.log(2 + data_in.sort_values(by="timestamp_start").groupby("user").cumcount())
    )

    # === Quantitative metrics ===
    for metric in METRIC_COLUMNS:
        X[metric] = data_in[metric]

    return X.astype(float)


# Logistic model on Explanation Clarity — Intersection
# =============================================================================

print("\n" + "=" * 80)
print("INTERSECTION MODEL — CLARITY (Logit)")
print("=" * 80)

# === Response var and Design matrix ===

data_clarity = clip_and_standardize_metrics(data, clip_percentile=0.05)
y_clarity = data_clarity["clarity"].map({"Yes": 1, "No": 0})
X_clarity = build_design_matrix(data_clarity)
X_clarity = sm.add_constant(X_clarity)

# === Variance Inflation Factors ===

vif_clarity = pd.DataFrame()
vif_clarity["feature"] = X_clarity.columns
vif_clarity["VIF"] = [
    variance_inflation_factor(X_clarity.values, i) for i in range(X_clarity.shape[1])
]
print(vif_clarity.sort_values("VIF", ascending=False))

# === Model ===

logit_clarity = sm.Logit(y_clarity, X_clarity).fit(maxiter=5000, method="newton")
print(logit_clarity.summary())

# === Plotting Clarity (Odds Ratios for Quantitative Metrics) ===


def extract_metric_effects(mod, columns):
    params = mod.params
    conf = mod.conf_int()
    results = []
    for col in columns:
        if col in params.index:
            results.append(
                {
                    "Metric": col,
                    "OR": np.exp(params[col]),
                    "Lower": np.exp(conf.loc[col, 0]),
                    "Upper": np.exp(conf.loc[col, 1]),
                }
            )
    return pd.DataFrame(results)


df_plot_clarity = extract_metric_effects(logit_clarity, METRIC_COLUMNS)
df_plot_clarity = df_plot_clarity.sort_values("OR")
sort_order = df_plot_clarity["Metric"].tolist()

plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
sns.set_context("talk")

palette_colors = dict(zip(sort_order, sns.color_palette("RdYlGn", len(sort_order))))
ax = sns.barplot(
    data=df_plot_clarity,
    x="Metric",
    y="OR",
    order=sort_order,
    hue="Metric",
    palette=palette_colors,
    edgecolor="0.1",
    alpha=0.7,
    legend=False,
)

for i, metric in enumerate(sort_order):
    row = df_plot_clarity[df_plot_clarity["Metric"] == metric].iloc[0]
    plt.errorbar(
        x=i,
        y=row["OR"],
        yerr=[[row["OR"] - row["Lower"]], [row["Upper"] - row["OR"]]],
        fmt="none",
        color="black",
        capsize=8,
        elinewidth=2,
    )

plt.yscale("log")
yticks = [0.75, 0.85, 1.0, 1.15, 1.33]
ax.set_yticks(yticks)
ax.set_yticklabels([str(y) for y in yticks])
ax.yaxis.set_minor_locator(plt.NullLocator())

plt.title(
    "Impact of Quantitative Metrics on Explanation Clarity (Odds Ratios)",
    fontsize=18,
    pad=25,
)
plt.ylabel("Odds Ratio (Log Scale)", fontsize=16)
plt.xlabel("", fontsize=14)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

modelling_plots_folder = Path(__file__).parent.parent / "plots" / "modelling"
os.makedirs(modelling_plots_folder, exist_ok=True)
plt.savefig(modelling_plots_folder / "intersection_clarity_odds_ratios.png")


# Ordered model on Decision Confidence — Intersection
# =============================================================================

print("\n" + "=" * 80)
print("INTERSECTION MODEL — CONFIDENCE (Ordered Logit)")
print("=" * 80)

# === Response var and Design matrix ===

data_confidence = clip_and_standardize_metrics(data, clip_percentile=0.25)
y_confidence = data_confidence["confidence"].map(
    {"Weak": 0, "Moderate": 1, "Strong": 2}
)
X_confidence = build_design_matrix(data_confidence)
# No constant for OrderedModel — handled internally

# === Variance Inflation Factors ===

vif_confidence = pd.DataFrame()
vif_confidence["feature"] = X_confidence.columns
vif_confidence["VIF"] = [
    variance_inflation_factor(X_confidence.values, i)
    for i in range(X_confidence.shape[1])
]
print(vif_confidence.sort_values("VIF", ascending=False))

# === Model ===

mod_confidence = OrderedModel(y_confidence.values, X_confidence, distr="logit").fit(
    method="bfgs", maxiter=500
)
print(mod_confidence.summary())

# === Plotting Confidence (Odds Ratios for Quantitative Metrics) ===

df_plot_conf = extract_metric_effects(mod_confidence, METRIC_COLUMNS)
df_plot_conf = df_plot_conf.sort_values("OR")
sort_order = df_plot_conf["Metric"].tolist()

plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
sns.set_context("talk")

palette_colors = dict(zip(sort_order, sns.color_palette("viridis", len(sort_order))))
ax = sns.barplot(
    data=df_plot_conf,
    x="Metric",
    y="OR",
    order=sort_order,
    hue="Metric",
    palette=palette_colors,
    edgecolor="0.1",
    alpha=0.8,
    legend=False,
)

for i, metric in enumerate(sort_order):
    row = df_plot_conf[df_plot_conf["Metric"] == metric].iloc[0]
    plt.errorbar(
        x=i,
        y=row["OR"],
        yerr=[[row["OR"] - row["Lower"]], [row["Upper"] - row["OR"]]],
        fmt="none",
        color="black",
        capsize=8,
        elinewidth=2,
    )

plt.yscale("log")
yticks = [0.75, 0.85, 1.0, 1.15, 1.33]
ax.set_yticks(yticks)
ax.set_yticklabels([str(y) for y in yticks])
ax.yaxis.set_minor_locator(plt.NullLocator())

plt.title(
    "Impact of Quantitative Metrics on Analyst Confidence (Odds Ratios)",
    fontsize=18,
    pad=25,
)
plt.ylabel("Odds Ratio for Higher Confidence (Log Scale)", fontsize=16)
plt.xlabel("", fontsize=14)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(modelling_plots_folder / "intersection_confidence_odds_ratios.png")

# === Save models ===

models_folder = Path(__file__).parent.parent / "models"
os.makedirs(models_folder, exist_ok=True)
logit_clarity.save(models_folder / "intersection_clarity_logit.pkl")
mod_confidence.save(models_folder / "intersection_confidence_orderedlogit.pkl")
