# A/B Testing Statistical Modelling

Statistical analysis of a user study evaluating how different Shapley value explainers affect human decision-making in a risk analysis tool.

## Overview

The dataset contains 3,715 records from an A/B test where analysts interact with a risk analysis tool. Each analyst reviews risk cases across 5 datasets and 2 model types, with one of 8 Shapley explainer configurations (or no explainer) randomly assigned per case.

The analysis fits regression models to isolate the effect of explainer type on four dependent variables, controlling for confounding factors like analyst identity, domain knowledge, model score characteristics, and learning effects.

## Data

The response data is retrieved from ... and has the following structure:

| Column | Description |
|--------|-------------|
| `timestamp_start` | Risk case start timestamp |
| `timestamp_end` | Risk case end timestamp |
| `confidence` | Analyst self-reported prediction confidence (Weak / Moderate / Strong) |
| `clarity` | Analyst evaluation of explanation intelligibility (Yes / No) |
| `prediction` | Analyst prediction (binary: risk or no risk) |
| `label` | Ground truth label (unobserved by analyst) |
| `explanation` | Shapley explainer configuration (null if no explainer) |
| `user` | Unique analyst identifier |
| `dataset` | Dataset analyzed (e.g. GermanCredit) |
| `model` | Model type (e.g. lightgbm) |
| `model_score` | Model score for the risk case |
| `score_percentile` | Percentile of model score within dataset |
| `domain_knowledge` | Self-reported domain knowledge |
| `ml_understanding` | Self-reported ML understanding |
| `shapley_understanding` | Self-reported Shapley value understanding |
| `case_counter` | Number of cases seen by analyst in current session |
| `instance_id` | Unique case identifier |

## Explainer Conditions

The study compares 8 Shapley explainer configurations against a no-explainer baseline:

| Condition | Description |
|-----------|-------------|
| None | No explainer shown (reference category) |
| `baseline_zero` | Zero baseline value function |
| `baseline_mean` | Mean baseline value function |
| `marginal_bg100` | Marginal sampling (background size 100) |
| `jointmarginal_bg100` | Per-feature marginal sampling (background size 100) |
| `uniform_bg100` | Uniform distribution sampling (background size 100) |
| `conditional_bg100` | Kernel-weighted nearest neighbors (background size 100) |
| `filteredconditional_bg100` | Lowest-prediction filtering (background size 100) |
| `counterfactual_bg100` | DiCE counterfactual generation (background size 100) |

## Project Structure

```
statistical_modelling/
├── analysis/
│   ├── eda.py                        # Exploratory data analysis and plots
│   ├── intersection_models.py        # Interaction/intersection regression models
│   ├── modelling_accuracy.py         # Logistic regression on analyst accuracy
│   ├── modelling_clarity.py          # Logistic regression on explanation clarity
│   ├── modelling_confidence.py       # Ordered logit on decision confidence
│   └── modelling_response_times.py   # Quantile regression on decision time
├── models/                           # Serialised fitted model objects (created by the analysis scripts)
├── paper_plots/
│   ├── fig_1a.py                     # Paper figure 1a generation
│   ├── fig_1b.py                     # Paper figure 1b generation
│   ├── fig_1c.py                     # Paper figure 1c generation
│   ├── fig_3.py                      # Paper figure 3 generation
│   └── fig_4.py                      # Paper figure 4 generation
├── plots/
│   ├── eda/                          # EDA output plots
│   ├── modelling/                    # Regression result plots
│   └── paper/                        # Publication-ready figures (PDF)
└── results/                          # Aggregated results and metrics from the benchmark experiments, as well as a parquet version of the human-AI interaction dataset
```

