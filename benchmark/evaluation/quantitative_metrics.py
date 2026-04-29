from typing import Optional

import numpy as np
import pandas as pd
import torch

from experiment.counterfactuals import (
    compute_counterfactuals,
    initialize_counterfactual_generator,
)
from src.amortiser import InferenceAmortiser
from src.models.model import Model
from src.value_functions.base_value_function import BaseValueFunction


def reconstruction_mse(
    model: Model, amortiser: InferenceAmortiser, x: np.ndarray, attr: np.ndarray
) -> float:
    """
    Compute the reconstruction MSE between the output of the model and the sum of the
    attributions and the average constant.

    Args:
        model (Model): The model to be explained
        amortiser (InferenceAmortiser): The amortiser used to compute the attributions
        x (np.ndarray): The input data
        attr (np.ndarray): The attributions

    Returns:
        float: The reconstruction MSE
    """
    y_hat = model(x, raw_score=True)
    average_constant = amortiser.get_average_constant()

    return np.mean(
        [(y_hat[i] - average_constant - np.sum(attr[i])) ** 2 for i in range(len(x))],
        dtype=float,
    )


def sparsity(attr: np.ndarray) -> float:
    """
    Compute the ratio between the L1 and L2 norms of the attributions.

    Args:
        attr (np.ndarray): The attributions

    Returns:
        float: The ratio between the L1 and L2 norms
    """
    l1_norm = [np.sum(np.abs(attr[i])) for i in range(len(attr))]
    l2_norm = [np.sqrt(np.sum(attr[i] ** 2)) for i in range(len(attr))]

    return np.mean(
        [l1_norm[i] / max(l2_norm[i], 1e-5) for i in range(len(l1_norm))], dtype=float
    )


def add_noise(
    X: pd.DataFrame, n_perturbations: int, noise_magnitude: float = 0.1
) -> pd.DataFrame:
    """
    Add noise to numerical input data

    Args:
        X (pd.DataFrame): Input data
        n_perturbations (int): Number of perturbed samples to create
        noise_magnitude (float): Magnitude of noise to add

    Returns:
        pd.DataFrame: Noisy input data
    """
    X_repeated = X.loc[X.index.repeat(n_perturbations)].reset_index(drop=True)
    std = X.std(axis=0).to_numpy().reshape(1, -1).repeat(len(X_repeated), axis=0)

    X_noisy = X_repeated + (
        np.random.normal(0, noise_magnitude, (X_repeated.shape[0], X_repeated.shape[1]))
        * std
    )

    return X_noisy


def add_noise_categorical(
    X: pd.DataFrame,
    category_probs: dict[str, dict[int, float]],
    n_perturbations: int,
    noise_magnitude: float = 0.5,
) -> pd.DataFrame:
    """
    Add noise to categorical input data by perturbing categories

    Args:
        X (pd.DataFrame): Input data
        category_probs (dict): Dictionary mapping column names to category probability
                               distributions
        n_perturbations (int): Number of perturbed samples to create
        noise_magnitude (float): Probability of perturbing each category

    Returns:
        pd.DataFrame: Noisy data
    """
    X_repeated = X.loc[X.index.repeat(n_perturbations)].reset_index(drop=True)
    X_noisy = X_repeated.copy()
    X_noisy = X_noisy.astype(int)

    for col in X.columns:
        col_category_probs = category_probs[col]
        unique_categories = list(col_category_probs.keys())

        should_perturb = np.random.binomial(
            1, noise_magnitude, size=len(X_repeated)
        ).astype(bool)

        probs = list(col_category_probs.values())
        sampled_indices = np.random.multinomial(1, probs, size=should_perturb.sum())
        sampled_indices = np.argmax(sampled_indices, axis=1)

        new_categories = [unique_categories[idx] for idx in sampled_indices]

        X_noisy.loc[should_perturb, col] = new_categories

    return X_noisy


def attribution_sensitivity(
    X: pd.DataFrame,
    classifier: Model,
    amortiser: InferenceAmortiser,
    numerical_features: list[str],
    categorical_features: list[str],
    category_probs: dict[str, dict[int, float]],
    n_perturbations: int = 100,
    noise_magnitude_numerical: float = 0.1,
    noise_magnitude_categorical: float = 0.5,
) -> float:
    """
    Compute the sensitivity of attributions to random perturbations in the input data.

    Args:
        X (pd.DataFrame): Input data
        classifier (Model): Classifier model
        amortiser (InferenceAmortiser): Amortiser for computing attributions
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        category_probs (dict): Dictionary mapping column names to category probability
                               distributions
        n_perturbations (int): Number of perturbed samples to create
        noise_magnitude_numerical (float): Magnitude of noise to add to numerical data
        noise_magnitude_categorical (float): Probability of perturbing each category

    Returns:
        float: Sensitivity of attributions to random perturbations
    """
    # Add noise to input data
    X_num = X[numerical_features]
    X_cat = X[categorical_features]

    X_num_noisy = add_noise(X_num, n_perturbations, noise_magnitude_numerical)
    X_cat_noisy = add_noise_categorical(
        X_cat, category_probs, n_perturbations, noise_magnitude_categorical
    )

    X_noisy = pd.concat([X_num_noisy, X_cat_noisy], axis=1)
    X_noisy = X_noisy[X.columns]

    # Compute amortised attributions and predictions for noisy data
    attr_noisy = amortiser(
        torch.tensor(X_num_noisy.values, dtype=torch.float32),
        torch.tensor(X_cat_noisy.values, dtype=torch.long),
    )
    y_hat_noisy = classifier(X_noisy.values, raw_score=True)

    # Compute amortised attributions and predictions for reference data
    attr = amortiser(
        torch.tensor(X_num.values, dtype=torch.float32),
        torch.tensor(X_cat.values, dtype=torch.long),
    )
    attr = attr.repeat_interleave(n_perturbations, dim=0)
    y_hat = classifier(X.values, raw_score=True).repeat(n_perturbations)

    # Compute variance of attributions and predictions
    attr_var = (
        torch.sqrt(torch.sum((attr_noisy - attr) ** 2, dim=1)).detach().cpu().numpy()
    )
    y_hat_var = np.abs(y_hat_noisy - y_hat) + 1e-5
    var = attr_var / y_hat_var
    return np.mean(var, dtype=float)


def prediction_entropy(y_pred: torch.Tensor) -> float:
    """
    Compute the entropy of predictions.

    Args:
        y_pred (torch.Tensor): Predictions

    Returns:
        float: Entropy of the predictions
    """
    y_pred_np = y_pred.detach().cpu().numpy()
    y_pred_np = np.clip(y_pred_np, 1e-7, 1 - 1e-7)
    entropy = -(
        y_pred_np * np.log2(y_pred_np) + (1 - y_pred_np) * np.log2(1 - y_pred_np)
    )
    return float(np.mean(entropy))


def deletion_auc(
    X: pd.DataFrame,
    attr: np.ndarray,
    value_function: BaseValueFunction,
    top_k: Optional[int] = 5,
) -> list[float]:
    """
    Compute the AUC of the deletion curve.


    Args:
        X (pd.DataFrame): Input data
        attr (np.ndarray): Attributions
        value_function (BaseValueFunction): Value function to use
        top_k (int): Number of top features to consider

    Returns:
        list[float]: AUC of the deletion curve
    """
    if top_k is None:
        top_k = X.shape[1]

    ordered_k = np.argsort(np.abs(attr))[:, ::-1]
    ordered_k = ordered_k.copy()

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_hat_original = torch.tensor(
        value_function.score_classifier(X.values, raw_score=False), dtype=torch.float32
    )

    best_entropy = prediction_entropy(y_hat_original)

    scores = [best_entropy]
    for i in range(min(top_k, X.shape[1])):
        features_to_mask = ordered_k[:, : i + 1]
        mask = torch.ones(X.shape, dtype=torch.float32)
        for row in range(len(mask)):
            mask[row, features_to_mask[row]] = 0

        y_hat, _ = value_function(
            X_tensor,
            mask,
            raw_score=False,
        )
        scores.append(prediction_entropy(y_hat))

    worst_entropy = scores[-1]

    original_difference = worst_entropy - best_entropy
    if original_difference == 0:
        original_difference = 1e-7

    deletion_scores = [
        1 - ((entropy - best_entropy) / original_difference) for entropy in scores
    ]

    return deletion_scores


def insertion_auc(
    X: pd.DataFrame,
    attr: np.ndarray,
    value_function: BaseValueFunction,
    top_k: Optional[int] = None,
) -> list[float]:
    """
    Compute the AUC of the insertion curve.

    Args:
        X (pd.DataFrame): Input data
        attr (np.ndarray): Attributions
        value_function (BaseValueFunction): Value function to use
        top_k (int): Number of top features to consider

    Returns:
        list[float]: AUC of the insertion curve
    """
    if top_k is None:
        top_k = X.shape[1]

    ordered_k = np.argsort(np.abs(attr))[:, ::-1]
    ordered_k = ordered_k.copy()

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    scores = []
    for i in range(min(top_k, X.shape[1]) + 1):
        features_to_mask = ordered_k[:, :i]
        mask = torch.zeros(X.shape, dtype=torch.float32)
        for row in range(len(mask)):
            mask[row, features_to_mask[row]] = 1

        y_hat, _ = value_function(
            X_tensor,
            mask,
            raw_score=False,
        )
        scores.append(prediction_entropy(y_hat))

    worst_entropy = scores[0]
    best_entropy = scores[-1]

    original_difference = worst_entropy - best_entropy
    if original_difference == 0:
        original_difference = 1e-7

    insertion_scores = [
        ((worst_entropy - entropy) / original_difference) for entropy in scores
    ]

    return insertion_scores


def compute_counterfactual_attributions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
    numerical_features: list[str],
    X_test: pd.DataFrame,
    classifier: Model,
    amortiser: InferenceAmortiser,
):
    """
    Compute counterfactual attributions for a set of test instances.

    Args:
        X_train (pd.DataFrame): Training data features
        y_train (pd.Series): Training data labels
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        X_test (pd.DataFrame): Test data features
        classifier (Model): Classifier model
        amortiser (InferenceAmortiser): Amortiser for computing attributions

    Returns:
        tuple[np.ndarray, np.ndarray]: Counterfactual attributions and predictions
    """
    # Prepare data for DiCE
    data = pd.concat([X_train, y_train], axis=1)

    cf_generator = initialize_counterfactual_generator(
        data, str(y_train.name), numerical_features, classifier
    )
    cf = compute_counterfactuals(cf_generator, X_test, 1, desired_class="opposite")
    X_cf = pd.DataFrame(cf.reshape(-1, X_test.shape[1]), columns=X_test.columns)

    # Compute attributions for counterfactual instances
    X_num_cf = X_cf[numerical_features]
    X_cat_cf = X_cf[categorical_features]
    attr_cf: torch.Tensor = amortiser(
        torch.tensor(X_num_cf.values, dtype=torch.float32),
        torch.tensor(X_cat_cf.values, dtype=torch.long),
    )
    pred_cf = classifier(X_cf.values, raw_score=True)

    return attr_cf.detach().cpu().numpy(), pred_cf


def contrastivity(
    attr: np.ndarray,
    pred: np.ndarray,
    attr_cf: np.ndarray,
    pred_cf: np.ndarray,
) -> float:
    """
    Calculate contrastivity metric using counterfactual explanations.

    Args:
        attr (np.ndarray): Attributions
        pred (np.ndarray): Predictions
        attr_cf (np.ndarray): Counterfactual attributions
        pred_cf (np.ndarray): Counterfactual predictions

    Returns:
        Mean contrastivity score across all instances
    """
    # Compute contrastivity per instance
    attr_diff = np.sqrt(np.sum((attr - attr_cf) ** 2, axis=1))
    pred_diff = np.abs(pred - pred_cf) + 1e-5

    return np.mean(attr_diff / pred_diff, dtype=float)
