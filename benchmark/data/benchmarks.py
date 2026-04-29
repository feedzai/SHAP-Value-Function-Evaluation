import os
import warnings
from typing import Callable, Optional

import kagglehub
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from data.configs import DATASET_CONFIGS
from experiment.paths import CACHE_DIR

warnings.filterwarnings("ignore")


def preprocess_and_cache_dataset(
    dataset_name: str,
) -> None:
    """
    Preprocess implemented datasets.

    Args:
        dataset_name (str): Name of the dataset to preprocess.
                            Currently implemented: "GermanCredit", "MaternalRisk",
                            "HELOC".
    """
    custom_preprocessing = None

    if dataset_name == "GermanCredit":
        download_german_credit()
        custom_preprocessing = custom_preprocessing_german_credit

    elif dataset_name == "MaternalRisk":
        download_maternal_risk()

    elif dataset_name == "HELOC":
        download_heloc()
        custom_preprocessing = custom_preprocessing_heloc

    elif dataset_name == "Adult":
        download_adult()
        custom_preprocessing = custom_preprocessing_adult

    else:
        raise ValueError(f"Dataset '{dataset_name}' not implemented")

    _preprocess_and_cache(dataset_name, custom_preprocessing)


def _preprocess_and_cache(
    dataset_name: str,
    custom_preprocessing: Optional[Callable] = None,
) -> None:
    """
    General preprocessing function for any dataset.

    Args:
        dataset_name (str): Name of the dataset to preprocess.
        custom_preprocessing (Callable): Custom preprocessing function.
    """
    # Get configuration (custom takes precedence over default)
    config = DATASET_CONFIGS.get(dataset_name, {}).copy()

    if not config:
        raise ValueError(f"No configuration found for dataset '{dataset_name}'")

    target_column = config["target_column"]
    categorical_features = config.get("categorical_features", [])

    # Load data
    data = pd.read_csv(CACHE_DIR / dataset_name / "data.csv")
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Apply custom preprocessing if provided
    if custom_preprocessing:
        X, y = custom_preprocessing(X, y, config)

    # Apply target transformation
    target_transform = config.get("target_transform", None)
    if target_transform:
        y = target_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=min(int(0.2 * len(X)), 400), random_state=0, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=0, stratify=y_test
    )

    # Save processed data
    _save_processed_data(dataset_name, X_train, X_val, X_test, y_train, y_val, y_test)

    # Save feature configuration
    _save_feature_config(dataset_name, X, target_column, categorical_features)


def _save_processed_data(
    dataset_name: str, X_train, X_val, X_test, y_train, y_val, y_test
) -> None:
    """
    Save processed data splits to parquet files.

    Args:
        dataset_name (str): Name of the dataset to save the data splits for.
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        y_test (pd.Series): Test labels.
    """
    pd.DataFrame(X_train).to_parquet(CACHE_DIR / dataset_name / "X_train.parquet")
    pd.DataFrame(y_train).to_parquet(CACHE_DIR / dataset_name / "y_train.parquet")
    pd.DataFrame(X_val).to_parquet(CACHE_DIR / dataset_name / "X_val.parquet")
    pd.DataFrame(y_val).to_parquet(CACHE_DIR / dataset_name / "y_val.parquet")
    pd.DataFrame(X_test).to_parquet(CACHE_DIR / dataset_name / "X_test.parquet")
    pd.DataFrame(y_test).to_parquet(CACHE_DIR / dataset_name / "y_test.parquet")


def _save_feature_config(
    dataset_name: str,
    X: pd.DataFrame,
    target_column: str,
    categorical_features: list[str],
) -> None:
    """
    Save feature configuration to YAML file.

    Args:
        dataset_name (str): Name of the dataset to save the feature configuration for.
        X (pd.DataFrame): Features.
        target_column (str): Target column.
        categorical_features (list[str]): Categorical features.
    """
    numerical_features = [f for f in X.columns if f not in categorical_features]

    if categorical_features:
        cardinalities = {
            cat_feature: len(X[cat_feature].unique())
            for cat_feature in categorical_features
        }
    else:
        cardinalities = None

    features = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "cardinalities": cardinalities,
        "target_column": target_column,
    }

    with open(CACHE_DIR / dataset_name / "features.yaml", "w") as f:
        yaml.dump(features, f)


def custom_preprocessing_german_credit(X, y, config):
    """
    Custom preprocessing for GermanCredit dataset.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        config (dict): Configuration.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Preprocessed data splits
    """
    categorical_features = config.get("categorical_features", [])

    for feature in categorical_features:
        if feature in X.columns:
            X[feature] = pd.Categorical(X[feature]).codes

    X["Attribute19"] = X["Attribute19"].apply(lambda x: 1 if x == "A192" else 0)
    X["Attribute20"] = X["Attribute20"].apply(lambda x: 1 if x == "A201" else 0)
    return X, y


def custom_preprocessing_adult(X, y, config):
    """
    Custom preprocessing for Adult dataset.
    """
    categorical_features = config.get("categorical_features", [])

    for feature in categorical_features:
        if feature in X.columns:
            X[feature] = pd.Categorical(X[feature]).codes

    X["SEX"] = X["SEX"].replace(2, 0)
    return X, y


def custom_preprocessing_heloc(
    X: pd.DataFrame, y: pd.Series, config: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Custom preprocessing for HELOC dataset.
    """
    # set -7, -8 and -9 to NaN
    X = X.replace([-7, -8, -9], np.nan)
    X = X.drop(columns=["MSinceMostRecentDelq"])
    X = X.dropna(how="all")
    X = X.dropna(subset=["MSinceMostRecentInqexcl7days", "MSinceOldestTradeOpen"])
    y = y.loc[X.index]

    for col in [
        "NumInstallTradesWBalance",
        "NumBank2NatlTradesWHighUtilization",
        "NumRevolvingTradesWBalance",
    ]:
        X[col] = X[col].fillna(0)

    # fill external risk estimate with median
    X["ExternalRiskEstimate"] = X["ExternalRiskEstimate"].fillna(
        X["ExternalRiskEstimate"].median()
    )

    for feature, supporting in [
        ("NetFractionRevolvingBurden", "NumRevolvingTradesWBalance"),
        ("NetFractionInstallBurden", "NumInstallTradesWBalance"),
        ("PercentTradesWBalance", "NumTotalTrades"),
    ]:
        X.loc[X[supporting] == 0, feature] = 0
        X.loc[X[feature].isna() & (X[supporting] > 0), feature] = X[feature].median()

    return X, y


def download_german_credit() -> None:
    """
    Download GermanCredit dataset from UCI repository.
    """
    statlog_german_credit_data = fetch_ucirepo(id=144)
    features = statlog_german_credit_data.data.features
    target_column = statlog_german_credit_data.data.targets

    data = pd.concat([features, target_column], axis=1)

    if not os.path.exists(CACHE_DIR / "GermanCredit"):
        os.makedirs(CACHE_DIR / "GermanCredit")
    data.to_csv(CACHE_DIR / "GermanCredit" / "data.csv", index=False)


def download_maternal_risk() -> None:
    """
    Download MaternalRisk dataset from UCI repository.
    """

    path = kagglehub.dataset_download(
        "csafrit2/maternal-health-risk-data", path="Maternal Health Risk Data Set.csv"
    )

    data = pd.read_csv(path)

    if not os.path.exists(CACHE_DIR / "MaternalRisk"):
        os.makedirs(CACHE_DIR / "MaternalRisk")
    data.to_csv(CACHE_DIR / "MaternalRisk" / "data.csv", index=False)


def download_heloc() -> None:
    """
    Download HELOC dataset from UCI repository.
    """

    path = kagglehub.dataset_download(
        "averkiyoliabev/home-equity-line-of-creditheloc",
        path="heloc_dataset_v1 (1).csv",
    )

    data = pd.read_csv(path)

    if not os.path.exists(CACHE_DIR / "HELOC"):
        os.makedirs(CACHE_DIR / "HELOC")
    data.to_csv(CACHE_DIR / "HELOC" / "data.csv", index=False)


def download_adult() -> None:
    data = pd.read_parquet(
        "https://github.com/dssg/aequitas/raw/master/datasets/"
        "FolkTables/ACSIncome.train.parquet"
    )

    if not os.path.exists(CACHE_DIR / "Adult"):
        os.makedirs(CACHE_DIR / "Adult")
    data.to_csv(CACHE_DIR / "Adult" / "data.csv", index=False)
