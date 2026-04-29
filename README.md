# SHAP Value Function Evaluation

A research project studying how different Shapley value function formulations affect both machine learning explanation quality and human decision-making. It consists of three components:

1. **Benchmark** — Train and evaluate amortized SHAP value explainers using neural networks across multiple value function formulations.
2. **Dashboard** — A Streamlit front-end for an A/B testing user study where analysts review risk cases with different Shapley explanations.
3. **Statistical Modelling** — Regression analysis of user study results, measuring the effect of explainer type on accuracy, confidence, clarity, and response time.

We further include the release of a dataset consisting of 3,735 granular human-AI interaction measurements to support the development of behaviorally-grounded XAI benchmarks. The dataset, alongside its croissant metadata file, can be found in the `data` folder.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd SHAP-Value-Function-Evaluation
uv sync
```

## Quick Start

### 1. Benchmark

All benchmark scripts run from the project root.

**Prepare a dataset:**

```bash
uv run python benchmark/preprocess_dataset.py GermanCredit
uv run python benchmark/train_classifier.py GermanCredit lightgbm
uv run python benchmark/calculate_reference_shap.py GermanCredit lightgbm
```

**Train an amortized explainer:**

```bash
uv run python benchmark/train.py GermanCredit lightgbm marginal --save
```

**Evaluate against reference SHAP values:**

```bash
uv run python benchmark/evaluate.py GermanCredit_lightgbm_marginal_bg50_epoch100_repeats4
```

See [`benchmark/README.md`](benchmark/README.md) for the full list of datasets, models, value functions, training parameters, and custom data instructions.

### 2. Dashboard

The dashboard requires a running backend API at `http://localhost:8000`. Once the API is available:

```bash
uv run streamlit run ui/dashboard/app.py
```

See [`ui/README.md`](ui/README.md) for the expected API endpoints, page descriptions, and widget details.

### 3. Statistical Modelling

Analysis scripts read from `data/responses.parquet` and output plots to `statistical_modelling/plots/`.

```bash
uv run python statistical_modelling/analysis/eda.py
uv run python statistical_modelling/analysis/modelling_accuracy.py
uv run python statistical_modelling/analysis/modelling_confidence.py
uv run python statistical_modelling/analysis/modelling_clarity.py
uv run python statistical_modelling/analysis/modelling_response_times.py
```

See [`statistical_modelling/README.md`](statistical_modelling/README.md) for the data schema and explainer conditions.

## Project Structure

```
├── pyproject.toml                  # Project metadata and dependencies
├── data/
│   └── responses.jsonl             # A/B test response data
│   └── data-croissant.jsonld       # Croissant metadata file
├── benchmark/                      # Amortized SHAP explainer training and evaluation
├── ui/dashboard/                   # Streamlit A/B testing dashboard
└── statistical_modelling/          # Regression analysis of user study results
```

## Supported Datasets

| Dataset | Task | Source | License |
|---------|------|--------|---------|
| GermanCredit | Credit default risk | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data | CC-BY 4.0 |
| MaternalRisk | Pregnancy health risk | https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data | CC-BY 4.0 | 
| HELOC | Home equity line of credit risk | https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc | CC-BY 4.0 |
| Adult | Income prediction (poverty risk) | https://github.com/dssg/aequitas/raw/master/datasets/FolkTables/ACSIncome.train.parquet | MIT |

## Value Functions

| Formulation | CLI Name |
|-------------|----------|
| Baseline (zero/mean) | `baseline` |
| Marginal | `marginal` |
| Joint Marginal | `jointmarginal` |
| Uniform | `uniform` |
| Conditional | `conditional` |
| Filtered Conditional | `filteredconditional` |
| Counterfactual | `counterfactual` |
