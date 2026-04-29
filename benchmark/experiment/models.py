from typing import Any, Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.lightgbm import LightGBMModel
from src.models.logisticregression import LogisticRegressionModel
from src.models.model import Model


def get_model(dataset_name: str, model_name: str) -> Model:
    if model_name == "lightgbm":
        return LightGBMModel(dataset_name)
    elif model_name == "logisticregression":
        return LogisticRegressionModel(dataset_name)
    else:
        raise ValueError(f"Model {model_name} not implemented")


def evaluate_model(
    model: Model, objective: str, X_train, y_train, X_val, y_val, X_test, y_test
) -> None:
    """
    Evaluate the model performance.

    Args:
        model (Model): Model.
        objective (str): Objective function.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
    """
    is_classification = objective in ["binary", "multiclass"]

    y_train_hat = model(X_train)
    y_val_hat = model(X_val)
    y_test_hat = model(X_test)

    print("\n===== Model Performance =====")

    if is_classification:
        auc_train = roc_auc_score(y_train, y_train_hat)
        auc_val = roc_auc_score(y_val, y_val_hat)
        auc_test = roc_auc_score(y_test, y_test_hat)

        print(f"\nTrain set AUC: {auc_train}")
        print(f"Validation set AUC: {auc_val}")
        print(f"Test set AUC: {auc_test}")

    mse_train = np.mean((y_train_hat - y_train) ** 2)
    mse_val = np.mean((y_val_hat - y_val) ** 2)
    mse_test = np.mean((y_test_hat - y_test) ** 2)

    print(f"\nTrain set MSE: {mse_train}")
    print(f"Validation set MSE: {mse_val}")
    print(f"Test set MSE: {mse_test}\n")


def train_and_save_model(
    dataset_name: str,
    model_name: str,
    model_config: Dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
) -> Model:
    """
    Train and save LightGBM model for the given dataset.

    Args:
        dataset_name (str): Name of the dataset.
        model_config (Dict[str, Any]): Model configuration.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.

    Returns:
        Model: Trained model.
    """
    model = get_model(dataset_name, model_name)

    model.train(X_train, y_train, X_val, y_val, model_config, save=True)

    return model
