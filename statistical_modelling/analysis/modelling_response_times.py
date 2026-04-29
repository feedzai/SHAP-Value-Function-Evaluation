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


# Fit linear model on Logarithm of Review Times - OLS
# =============================================================================

# ===  Response var and Design matrix ===

X = pd.DataFrame(index=data.index)
y = np.log((data["timestamp_end"] - data["timestamp_start"]).dt.total_seconds())

# === Categorical controls ===

X["explainer"] = data[["explanation"]].fillna("None")
X["model"] = data["model"]
X["domain"] = data["dataset"]
# X['knows_shapley'] = data['shapley_understanding']    # Only add if no Analyst
# X['knows_ml'] = data['ml_understanding']              # Only add if no Analyst
X["knows_domain"] = data["domain_knowledge"]
X["analyst"] = data[["user"]]

# === Categorical controls to dummies -> Remove Explainer 'None' ===

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

# Design matrix
X = sm.add_constant(X.astype(float))

# === Variance Inflation Factors ===

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data.sort_values("VIF", ascending=False))

# === Models ===

# Mean Absolute Error
model_ols = sm.QuantReg(y, X).fit(q=0.5)
print(model_ols.summary())

# Tail Regressions
model_q025 = sm.QuantReg(y, X).fit(q=0.025)
print(model_q025.summary())
model_q975 = sm.QuantReg(y, X).fit(q=0.975)
print(model_q975.summary())

# === Plotting ===


# Convert Explainer log-coefficients to Multiplicative Effects (%)
def extract_explainer_effects(mod, label: str) -> list:
    params = mod.params
    conf = mod.conf_int()

    # Filter only columns related to explainers
    explainer_cols = [c for c in X.columns if c.startswith("explainer_")]

    results = []
    for col in explainer_cols:
        name = col.replace("explainer_", "")

        results.append(
            {
                "Explainer": name,
                "Metric": label,
                "Effect": np.exp(params[col]) * 100,
                "Lower": np.exp(conf.loc[col, 0]) * 100,
                "Upper": np.exp(conf.loc[col, 1]) * 100,
            }
        )

    # Add the 'None' baseline (Reference category is always 100% with no error)
    results.append(
        {
            "Explainer": "None",
            "Metric": label,
            "Effect": 100.0,
            "Lower": 100.0,
            "Upper": 100.0,
        }
    )
    return results


# Combine Results
all_results = []
all_results.extend(extract_explainer_effects(model_q025, "Low (q=0.025)"))
all_results.extend(extract_explainer_effects(model_ols, "Mean (OLS)"))
all_results.extend(extract_explainer_effects(model_q975, "High (q=0.975)"))

df_plot = pd.DataFrame(all_results)

# Plotting
sort_order = (
    df_plot[df_plot["Metric"] == "Mean (OLS)"]
    .sort_values("Effect")["Explainer"]
    .tolist()
)

plt.figure(figsize=(20, 9))
sns.set_style("whitegrid")
sns.set_context("talk")

palette = {
    "Low (q=0.025)": "#66c2a5",
    "Mean (OLS)": "#8da0cb",
    "High (q=0.975)": "#fc8d62",
}

ax = sns.barplot(
    data=df_plot,
    x="Explainer",
    y="Effect",
    hue="Metric",
    order=sort_order,
    palette=palette,
    edgecolor="0.1",
    alpha=0.85,
)

# Add Error Bars (since we have exact CI bounds from the model)
n_metrics = 3
bar_width = 0.8 / n_metrics  # Standard grouped bar width logic

for i, explainer in enumerate(sort_order[1:]):
    for j, metric in enumerate(["Low (q=0.025)", "Mean (OLS)", "High (q=0.975)"]):
        subset = df_plot[
            (df_plot["Explainer"] == explainer) & (df_plot["Metric"] == metric)
        ].iloc[0]

        # Calculate x-coordinate shift for each grouped bar
        x_coord = 1 + i + (j - (n_metrics - 1) / 2) * bar_width

        plt.errorbar(
            x=x_coord,
            y=subset["Effect"],
            yerr=[
                [subset["Effect"] - subset["Lower"]],
                [subset["Upper"] - subset["Effect"]],
            ],
            fmt="none",
            color="black",
            capsize=5,
            elinewidth=1.5,
        )

# Labels and Styling
plt.title("Multiplicative Impact on Decision Time", fontsize=18, pad=20)
plt.ylabel('Decision Time relative to "None" (%)', fontsize=16)
plt.xlabel(" ", fontsize=14)
plt.legend(title="Quantile / Model", bbox_to_anchor=(0, 1), loc="upper left")

plt.ylim(0, df_plot["Upper"].max() * 1.1)  # Ensure space for error bars
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

# Create directory if it doesn't exist
modelling_plots_folder = root_folder / "statistical_modelling" / "plots" / "modelling"
os.makedirs(modelling_plots_folder, exist_ok=True)
plt.savefig(modelling_plots_folder / "decision_time_effects.png")

# Save models
models_folder = root_folder / "statistical_modelling" / "models"
os.makedirs(models_folder, exist_ok=True)
model_ols.save(models_folder / "response_time_quantreg_q50.pkl")
model_q025.save(models_folder / "response_time_quantreg_q025.pkl")
model_q975.save(models_folder / "response_time_quantreg_q975.pkl")
