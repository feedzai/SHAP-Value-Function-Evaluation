# SHAP Value Function Benchmark

A benchmark framework for training amortized SHAP value explainers using neural networks. Implements FastSHAP-style amortized explainers that learn to predict SHAP values efficiently, avoiding the computational overhead of traditional SHAP computation methods.

## Overview

This project provides a framework for:
- Training neural networks to predict SHAP values (amortized explainers)
- Benchmarking different SHAP value function formulations
- Evaluating attribution quality using quantitative and alignment metrics

## Quick Start

All scripts should be run from the `benchmark/` directory.

### 1. Prepare a Dataset

Before training an amortiser, these steps should be taken to make sure all necessary artifacts are available.

1. Preprocess and cache the data. The prepocessed data will be stored in a `benchmark/cache/{dataset name}` folder.

```bash
uv run python preprocess_dataset.py GermanCredit
```

2. Train a classifier on the processed data.

```bash
uv run python train_classifier.py GermanCredit lightgbm
```

The traning configurations can be edited on `benchmark/data/configs.py` under `model_config`. 
The trained classifier is stored in the `benchmark/cache/{dataset name}` folder.

3. Calculate and cache reference TreeSHAP and KernelSHAP values for the validation and test splits:

```bash
uv run python calculate_reference_shap.py GermanCredit lightgbm
```

### 2. Train an Amortized Explainer

Train a neural network to predict SHAP values that explain a classifier. The results will be stored in a `benchmark/results/{experiment name}` folder. The experiment name is automatically generated.

```bash
uv run python train.py GermanCredit lightgbm marginal --save
```

All training parameters can be customized:

```bash
uv run python train.py GermanCredit lightgbm marginal \
  --background_size 50 \
  --num_epochs 100 \
  --lr_adam 1e-3 \
  --lr_sgd 1e-3 \
  --adam_epochs 30 \
  --warmup_epochs 10 \
  --n_repeats 4 \
  --reg 0.1 \
  --batch_size 64 \
  --save
```

To train using the baseline value functions, the baseline type must be selected (it will be `zero`by default):

```bash
uv run python train.py GermanCredit lightgbm baseline --baseline_type zero --save
```

### 3. Evaluate Attributions

Compare the amortized explainer against reference SHAP values using bootstrap sampling:

```bash
uv run python evaluate.py GermanCredit_lightgbm_baseline_zero_epoch100_repeats4
```

The experiment name is auto-generated during training as `{dataset}_{model}_{value_function}_epoch{N}_repeats{R}`.

## Supported Datasets

We support 4 binary classification datasets related, where the prediction task is related to risk.

| Dataset | Task | Source | License |
|---------|------|--------|---------|
| GermanCredit | Credit default risk | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data | CC-BY 4.0 |
| MaternalRisk | Pregnancy health risk | https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data | CC-BY 4.0 | 
| HELOC | Home equity line of credit risk | https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc | CC-BY 4.0 |
| Adult | Income prediction (poverty risk) | https://github.com/dssg/aequitas/raw/master/datasets/FolkTables/ACSIncome.train.parquet | MIT |

Datasets are automatically downloaded and preprocessed by `preprocess_dataset.py`. Sources include UCI ML Repository and Kaggle.

## Supported Models

- `lightgbm` — LightGBM gradient boosting (supports TreeSHAP for reference values)
- `logisticregression` — Logistic Regression

## Value Functions

Value functions define how absent features are handled when computing SHAP values. Each formulation produces a different background distribution for masked features:

| Value Function | CLI Name |
|----------------|----------|
| Baseline | `baseline` |
| Marginal | `marginal` | 
| Joint Marginal | `jointmarginal` | 
| Uniform | `uniform` | 
| Conditional | `conditional` | 
| Filtered Conditional | `filteredconditional` | 
| Counterfactual | `counterfactual` | 

## Using Custom Data

To use your own data and model, create a folder inside `cache/` with your dataset name:

```
cache/
└── YourDatasetName/
    ├── X_train.parquet          # Training features
    ├── X_val.parquet            # Validation features
    ├── X_test.parquet           # Test features
    ├── y_train.parquet          # Training targets
    ├── y_val.parquet            # Validation targets
    ├── y_test.parquet           # Test targets
    ├── features.yaml            # Feature configuration
    └── classifier.pkl           # Trained classifier model (joblib format)
```

**Feature configuration (`features.yaml`):**

```yaml
cardinalities:
  education: 4        # 0=high_school, 1=bachelor, 2=master, 3=phd
  marital_status: 3   # 0=single, 1=married, 2=divorced
categorical_features:
- education
- marital_status
numerical_features:
- age
- income
- credit_score
target_column: default_risk
```

Notes:
- Categorical features must be integer-encoded starting from 0
- `cardinalities` specifies the number of unique values per categorical feature
- If there are no categorical features, set `categorical_features: []` and `cardinalities: null`
- Feature names must match column names in the parquet files
- The target column name must match the column name in `y_*.parquet` files

## Project Structure

```
benchmark/
├── preprocess_dataset.py          # Download and preprocess datasets
├── train_classifier.py            # Train a classifier model
├── calculate_reference_shap.py    # Compute reference TreeSHAP/KernelSHAP values
├── train.py                       # Train an amortized explainer
├── evaluate.py                    # Evaluate against reference SHAP values
├── data/
│   ├── benchmarks.py              # Dataset download and preprocessing pipeline
│   ├── configs.py                 # Per-dataset configuration (targets, transforms, model params)
│   ├── dataset.py                 # Dataset class with data loaders
│   └── utils.py                   # Data loading and utility functions
├── src/
│   ├── amortiser.py               # Amortiser: neural network that learns SHAP values
│   ├── nn_embeddings.py           # Neural network with categorical embeddings
│   ├── optimizer.py               # Scheduled optimizer (Adam → SGD with warmup)
│   ├── sampler.py                 # SHAP kernel mask sampling
│   ├── scaler.py                  # Robust feature scaling
│   ├── models/
│   │   ├── model.py               # Abstract model interface
│   │   ├── lightgbm.py            # LightGBM wrapper
│   │   └── logisticregression.py  # Logistic Regression wrapper
│   └── value_functions/
│       ├── base_value_function.py  # Abstract base class
│       ├── baseline.py             # Zero/mean baseline
│       ├── marginal.py             # Marginal value function
│       ├── jointmarginal.py        # Joint-marginal value function
│       ├── uniform.py              # Uniform value function
│       ├── conditional.py          # Conditional value function
│       ├── filtered_conditional.py # Filtered Conditional value function
│       ├── counterfactual.py       # Coutnerfactual value function
│       └── utils.py                # Value function factory
├── evaluation/
│   ├── benchmark.py                   # Bootstrap metric computation
│   ├── quantitative_metrics.py        # Quantitative evaluation metrics
│   ├── amortiser_alignment_metrics.py # Alignment metrics vs. reference SHAP
│   └── utils.py                       # Evaluation utilities
└── experiment/
    ├── arg_parsers.py             # CLI argument parsers for all scripts
    ├── constants.py               # Default parameters, dataset/model/value function lists
    ├── counterfactuals.py         # Counterfactual generation utilities
    ├── models.py                  # Model training and evaluation helpers
    ├── paths.py                   # Cache, results, and benchmark directory paths
    ├── shap_values.py             # Reference SHAP computation (TreeSHAP, KernelSHAP)
    └── utils.py                   # Experiment naming and config saving
```

## Results

Trained models, amortized attributions, experiment configs, and evaluation results are saved in the `results/` directory, organized by experiment name.
