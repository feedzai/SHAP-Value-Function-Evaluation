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
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
root_folder = Path(__file__).parent.parent.parent
data = pd.read_parquet(root_folder / "data" / "responses.parquet").reset_index(
    drop=True
)

# Remove cases without an explainer being present (cannot be used for Clarity)
data = data[~data["explanation"].isnull()]

# Identify users with at least 20 reviews
min_obs = 20
user_counts = data["user"].value_counts()
valid_users = user_counts[user_counts >= min_obs].index

# Filter the dataset
data = data[data["user"].isin(valid_users)].copy().reset_index()


# Logistic model on Explanation Clarity - GLM -> Odds Ratios
# =============================================================================

# ===  Response var and Design matrix ===

X = pd.DataFrame(index=data.index)
y = data["clarity"].map({"Yes": 1, "No": 0})

# === Categorical controls ===

X["model"] = data["model"]
X["domain"] = data["dataset"]
X["knows_domain"] = data["domain_knowledge"]
X["analyst"] = data[["user"]]

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

# === SUM CODING for Explainers ===

# To analyze the explainers relative to each other without a baseline
explainer_categories = data["explanation"].fillna("None").unique()
reference_explainer = explainer_categories[-1]
other_explainers = [e for e in explainer_categories if e != reference_explainer]

# Create standard dummies (0/1)
for exp in other_explainers:
    X[f"explainer_{exp}"] = (data["explanation"] == exp).astype(float)

# Apply a -1 shift for the reference category
is_reference = data["explanation"] == reference_explainer
for exp in other_explainers:
    X.loc[is_reference, f"explainer_{exp}"] = -1.0

# Design matrix
X = sm.add_constant(X.astype(float))

# === Variance Inflation Factors ===

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data.sort_values("VIF", ascending=False))

# === Models ===

# Logistic Model
logit_model = sm.Logit(y, X).fit(maxiter=100, method="newton")
print(logit_model.summary())

# === Explainers Relative to Average Clarity ===


def get_full_explainer_results(mod, prefix="explainer_", ref_name=reference_explainer):
    params = mod.params
    conf = mod.conf_int()
    pvalues = mod.pvalues

    explainer_cols = [c for c in params.index if c.startswith(prefix)]

    results = []
    for col in explainer_cols:
        name = col.replace(prefix, "")
        results.append(
            {
                "Explainer": name,
                "coef": round(params[col], 4),
                "std err": round(mod.bse[col], 3),
                "P>|z|": round(pvalues[col], 3),
                "Lower": round(conf.loc[col, 0], 3),
                "Upper": round(conf.loc[col, 1], 3),
            }
        )

    # Calculate Implicit Reference Category Stats
    # Beta_ref = -(Sum of all other Betas)
    implicit_coef = -params[explainer_cols].sum()

    # Var(Beta_ref) = Sum of all elements in the covariance sub-matrix
    cov_matrix = mod.cov_params().loc[explainer_cols, explainer_cols]
    implicit_se = np.sqrt(cov_matrix.values.sum())

    # CI and P-value for the implicit category
    z_score = 1.96
    lower_ci = implicit_coef - (z_score * implicit_se)
    upper_ci = implicit_coef + (z_score * implicit_se)
    p_val = 2 * (1 - stats.norm.cdf(np.abs(implicit_coef / implicit_se)))

    results.append(
        {
            "Explainer": ref_name,
            "coef": round(implicit_coef, 4),
            "std err": round(implicit_se, 3),
            "P>|z|": round(p_val, 3),
            "Lower": round(lower_ci, 3),
            "Upper": round(upper_ci, 3),
        }
    )

    df_res = pd.DataFrame(results)
    # Calculate Odds Ratios for easier interpretation
    df_res["OR"] = np.exp(df_res["coef"])
    df_res["OR_Lower"] = np.exp(df_res["Lower"])
    df_res["OR_Upper"] = np.exp(df_res["Upper"])

    return df_res


# Execute extraction
df_clarity = get_full_explainer_results(logit_model)
print(df_clarity)

# === Plotting Explanation Clarity (Odds Ratios) Relative to Average ===

df_plot = df_clarity.sort_values("OR").copy()
sort_order = df_plot["Explainer"].tolist()

plt.figure(figsize=(16, 9))
sns.set_style("whitegrid")
sns.set_context("talk")

colors = sns.diverging_palette(10, 130, sep=10, n=len(df_plot))

palette_colors = dict(zip(sort_order, colors))
ax = sns.barplot(
    data=df_plot,
    x="Explainer",
    y="OR",
    order=sort_order,
    hue="Explainer",
    palette=palette_colors,
    edgecolor="0.2",
    alpha=0.85,
    legend=False,
)

# Add Error Bars
for i, explainer in enumerate(sort_order):
    row = df_plot[df_plot["Explainer"] == explainer].iloc[0]

    # In Sum Coding, all categories (including implicit) have valid CIs
    plt.errorbar(
        x=i,
        y=row["OR"],
        yerr=[[max(0, row["OR"] - row["OR_Lower"])], [row["OR_Upper"] - row["OR"]]],
        fmt="none",
        color="black",
        capsize=6,
        elinewidth=1.5,
        alpha=0.8,
    )

# Log scale is statistically correct for Odds Ratios
plt.yscale("log")

# Customizing Y-ticks to be readable
yticks = [0.25, 0.5, 1.0, 2.0]
ax.set_yticks(yticks)
ax.set_yticklabels([str(y) for y in yticks])
ax.yaxis.set_minor_locator(plt.NullLocator())

plt.title(
    "Explainer Intelligibility (Odds Ratios Relative to Average)", fontsize=20, pad=30
)
plt.ylabel("Odds Ratio vs. Average Explainer (Log Scale)", fontsize=16)
plt.xlabel(" ", fontsize=18)
plt.xticks(rotation=40, ha="right", fontsize=14)

plt.tight_layout()

# Create directory if it doesn't exist
modelling_plots_folder = root_folder / "statistical_modelling" / "plots" / "modelling"
os.makedirs(modelling_plots_folder, exist_ok=True)
plt.savefig(modelling_plots_folder / "clarity_odds_ratios.png")

# Save model
models_folder = root_folder / "statistical_modelling" / "models"
os.makedirs(models_folder, exist_ok=True)
logit_model.save(models_folder / "clarity_logit.pkl")
