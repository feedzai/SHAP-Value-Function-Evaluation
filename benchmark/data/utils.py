import yaml
import numpy as np
import pandas as pd
from experiment.paths import CACHE_DIR
from src.models.lightgbm import LightGBMModel
from src.models.logisticregression import LogisticRegressionModel
from src.models.model import Model


def load_data(
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load benchmark dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
            X_train (DataFrame): Training features.
            X_val (DataFrame): Validation features.
            X_test (DataFrame): Test features.
            y_train (Series): Training labels.
            y_val (Series): Validation labels.
            y_test (Series): Test labels.
    """
    try:
        with open(CACHE_DIR / dataset_name / "features.yaml", "r") as f:
            feature_config = yaml.load(f, Loader=yaml.SafeLoader)
        target_column = feature_config["target_column"]

        X_train = pd.read_parquet(CACHE_DIR / dataset_name / "X_train.parquet")
        X_val = pd.read_parquet(CACHE_DIR / dataset_name / "X_val.parquet")
        X_test = pd.read_parquet(CACHE_DIR / dataset_name / "X_test.parquet")
        y_train = pd.read_parquet(CACHE_DIR / dataset_name / "y_train.parquet")[
            target_column
        ]
        y_val = pd.read_parquet(CACHE_DIR / dataset_name / "y_val.parquet")[
            target_column
        ]
        y_test = pd.read_parquet(CACHE_DIR / dataset_name / "y_test.parquet")[
            target_column
        ]
        return X_train, X_val, X_test, y_train, y_val, y_test

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data not found for {dataset_name}. "
            f"You can preprocess the implemented benchmarks using "
            f"`python preprocess_dataset.py {dataset_name}`. "
            "Check the README for more information."
        )


def load_model(
    dataset_name: str,
    model_name: str,
) -> Model:
    """
    Load model from cache.
    """
    if model_name == "lightgbm":
        model = LightGBMModel(dataset_name)
    elif model_name == "logisticregression":
        model = LogisticRegressionModel(dataset_name)
    else:
        raise ValueError(f"Model {model_name} not implemented")

    model.load()
    return model


def load_ref_shap_values(
    dataset_name: str, model_name: str, value_function_name: str, split: str
) -> np.ndarray:
    try:
        if value_function_name.startswith("conditional") and model_name == "lightgbm":
            return np.load(
                CACHE_DIR / dataset_name / f"{model_name}_tree_shap_{split}.npy"
            )
        else:
            return np.load(
                CACHE_DIR
                / dataset_name
                / f"{model_name}_kernel_shap_{value_function_name}_{split}.npy"
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference SHAP values for {model_name} trained on "
            f"{dataset_name} not found. "
            f"Run `python calculate_ref_shap.py {dataset_name} {model_name}`. "
            f"More information on the README."
        )


def stratified_sampling(
    X: np.ndarray, preds: np.ndarray, size: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.quantile(preds, np.linspace(0, 1, size, endpoint=False))
    quantile_bins = np.digitize(preds, bins=quantiles)
    selected_indices = []

    unique_bins = np.unique(quantile_bins)
    for q in unique_bins:
        q_indices = np.where(quantile_bins == q)[0]
        if len(q_indices) > 0:
            chosen = np.random.choice(q_indices, size=1, replace=False)
            selected_indices.extend(chosen)

    if len(selected_indices) < size:
        remaining_indices = np.setdiff1d(np.arange(len(preds)), selected_indices)
        if len(remaining_indices) > 0:
            additional_needed = size - len(selected_indices)
            additional = np.random.choice(
                remaining_indices,
                size=min(additional_needed, len(remaining_indices)),
                replace=False,
            )
            selected_indices.extend(additional)

    return X[selected_indices], np.array(selected_indices)
