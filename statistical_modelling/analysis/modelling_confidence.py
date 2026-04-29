"""
Summary:
- This dataset contains 3,715 records and 19 columns.
- Captures cases of user interactions with a risk analysis tool.
- The tool is used for A/B testing with different Shapley explainers.
- Participants are evaluated in machine learning model predictions.
- Risk cases span 5 datasets, 2 model types, and 8 explainers.

### Dependent Variables ###

- Temporal Metrics:

    'timestamp_start':          Risk case start timestamp
    'timestamp_end':            Risk case end timestamp

- Analyst Feedback:

    'confidence':               Subjective evaluation of Analyst 'prediction' quality
    'clarity':                  Subjective evaluation of 'explanation' intelligibility

### Independent Variables ###

- Shapley Explainers:

    'explanation':              Shapley explainer configuration, e.g. marginal

### Confounding Variables ###

- Identification:

    'user':                     Unique Analyst ID or Name
    'dataset':                  Dataset analyzed for a Risk case, e.g. GermanCredit
    'model':                    Model analyzed in a Risk case, e.g. lightgbm

- Predictions & Labels:

    'prediction':               Analyst Prediction (binary: risk or no risk)
    'model_score':              Score of 'model' for a Risk case
    'score_percentile':         Percentile of 'model_score for Risk case across dataset
    'label':                    Ground Truth label (risk or no risk), unobserved to user

- Analyst Expertise:

    'domain_knowledge':         Self-reported domain knowledge for dataset in Risk case
    'ml_understanding':         Self-reported understanding of Machine Learning
    'shapley_understanding':    Self-reported understanding of Shapley values

- Temporal Metrics:

    'case_counter':             Counter or Risk cases seen by Analyst in single session

### Other Variables ###

- Identification:

    'instance_id':              Unique case Identifier

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
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
root_folder = Path(__file__).parent.parent.parent
data = pd.read_parquet(root_folder / "data" / "responses.parquet").reset_index(
    drop=True
)

# Identify users with at least 20 reviews
min_obs = 20
user_counts = data["user"].value_counts()
valid_users = user_counts[user_counts >= min_obs].index

# Filter the dataset
data = data[data["user"].isin(valid_users)].copy().reset_index()


# Logistic model on Decision Confidence - GLM (Ordered Response) -> Odds Ratios
# =============================================================================

# ===  Response var and Design matrix ===

X = pd.DataFrame(index=data.index)

# Encode Ordered levels for a Proportional Odds model
y = data["confidence"].map({"Weak": 0, "Moderate": 1, "Strong": 2})

# === Categorical controls ===

X["explainer"] = data[["explanation"]].fillna("None")
X["model"] = data["model"]
X["domain"] = data["dataset"]
X["knows_domain"] = data["domain_knowledge"]
X["analyst"] = data[["user"]]

# === Categorical controls to dummies -> Drops Explainer 'None' ===

X["explainer"] = pd.Categorical(
    X["explainer"],
    categories=[
        "None",  # Must be first so that it becomes the dropped category
        "counterfactual_bg100",
        "conditional_bg100",
        "jointmarginal_bg100",
        "uniform_bg100",
        "baseline_mean",
        "marginal_bg100",
        "filteredconditional_bg100",
        "baseline_zero",
    ],
    ordered=True,
)

X = pd.get_dummies(X, drop_first=True, dtype=float)

# === Manual standardization for numerical controls ===


def standardize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


X["score"] = standardize(data["model_score"])
X["score_error"] = standardize(np.abs(data["label"] - data["model_score"]))
X["score_entropy"] = standardize(
    -data["model_score"] * np.log(data["model_score"])
    - (1 - data["model_score"]) * np.log(1 - data["model_score"])
)

X["log_count_session"] = standardize(np.log(1 + data["case_counter"]))
X["log_count_total"] = standardize(
    np.log(2 + data.sort_values(by="timestamp_start").groupby("user").cumcount())
)

# We add NO constant in an ordered modelled -> Handled internally

# === Variance Inflation Factors ===

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data.sort_values("VIF", ascending=False))

# === Models ===

# Ordered Model
mod_ord = OrderedModel(y, X, distr="logit").fit(method="bfgs", maxiter=500)
print(mod_ord.summary())

# === Plotting Analyst Confidence (Odds Ratios) ===


def extract_confidence_effects(mod_res):
    params = mod_res.params
    conf = mod_res.conf_int()

    explainer_cols = [c for c in params.index if c.startswith("explainer_")]

    results = []
    for col in explainer_cols:
        name = col.replace("explainer_", "")

        results.append(
            {
                "Explainer": name,
                "OR": np.exp(params[col]),
                "Lower": np.exp(conf.loc[col, 0]),
                "Upper": np.exp(conf.loc[col, 1]),
            }
        )

    # Add the 'None' baseline (Reference category OR is always 1.0)
    results.append({"Explainer": "None", "OR": 1.0, "Lower": 1.0, "Upper": 1.0})
    return pd.DataFrame(results)


# Prepare Data
df_conf_plot = extract_confidence_effects(mod_ord)

# Sort by Odds Ratio (ascending)
sort_order = df_conf_plot.sort_values("OR")["Explainer"].tolist()

# Plotting
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
sns.set_context("talk")

# Using a palette that highlights the difference from 1.0
palette_colors = dict(zip(sort_order, sns.color_palette("viridis", len(sort_order))))
ax = sns.barplot(
    data=df_conf_plot,
    x="Explainer",
    y="OR",
    order=sort_order,
    hue="Explainer",
    palette=palette_colors,
    edgecolor="0.1",
    alpha=0.8,
    legend=False,
)

# Add Error Bars
for i, explainer in enumerate(sort_order):
    subset = df_conf_plot[df_conf_plot["Explainer"] == explainer].iloc[0]

    if explainer != "None":
        plt.errorbar(
            x=i,
            y=subset["OR"],
            yerr=[[subset["OR"] - subset["Lower"]], [subset["Upper"] - subset["OR"]]],
            fmt="none",
            color="black",
            capsize=8,
            elinewidth=2,
        )

# Log scale is statistically correct for Odds Ratios
plt.yscale("log")

# Customizing Y-ticks to be readable linear values
yticks = [0.25, 0.5, 1.0, 2.0]
ax.set_yticks(yticks)
ax.set_yticklabels([str(y) for y in yticks])
ax.yaxis.set_minor_locator(plt.NullLocator())

# Labels and Styling
plt.title(
    "Impact of Explainers on Analyst Confidence (Odds Ratios vs. No Explainer)",
    fontsize=18,
    pad=25,
)
plt.ylabel("Odds Ratio for Higher Confidence (Log Scale)", fontsize=16)
plt.xlabel("", fontsize=14)
plt.xticks(rotation=35, ha="right")

plt.tight_layout()

# Create directory if it doesn't exist
modelling_plots_folder = root_folder / "statistical_modelling" / "plots" / "modelling"
os.makedirs(modelling_plots_folder, exist_ok=True)
plt.savefig(modelling_plots_folder / "confidence_odds_ratios.png")

# Save model
models_folder = root_folder / "statistical_modelling" / "models"
os.makedirs(models_folder, exist_ok=True)
mod_ord.save(models_folder / "confidence_orderedlogit.pkl")
