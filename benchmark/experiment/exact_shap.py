import math

import numpy as np
import pandas as pd
import torch
from experiment.paths import CACHE_DIR
from src.value_functions.base_value_function import BaseValueFunction


def _enumerate_masks(num_features: int) -> np.ndarray:
    """
    Enumerate all 2**num_features coalition masks.

    Args:
        num_features (int): Number of features (d).

    Returns:
        np.ndarray: Array of shape (2**d, d) where mask[k, j] == 1 iff feature j
            is present in coalition k. Matches the value-function convention where
            S == 1 keeps the instance value and S == 0 uses the background.
    """
    subsets = np.arange(2**num_features, dtype=np.int64)
    bits = np.arange(num_features, dtype=np.int64)
    return ((subsets[:, None] >> bits[None, :]) & 1).astype(np.float32)


def _shapley_size_weights(num_features: int) -> np.ndarray:
    """
    Precompute the Shapley weight for a coalition S (not containing feature j) as a
    function of its size |S|: |S|! (d - |S| - 1)! / d!.

    Args:
        num_features (int): Number of features (d).

    Returns:
        np.ndarray: Array of shape (d,) indexed by coalition size 0..d-1.
    """
    d = num_features
    return np.array(
        [
            math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
            for s in range(d)
        ],
        dtype=np.float64,
    )


def compute_exact_shap_values(
    value_function: BaseValueFunction,
    X: np.ndarray,
    device: torch.device,
    raw_score: bool = True,
    coalition_batch_size: int = 8192,
) -> tuple[np.ndarray, float]:
    """
    Compute exact (brute-force) Shapley values for every instance in X against the
    game defined by ``value_function``.

    For each instance the value function is evaluated on all 2**d coalitions, then
    combined with the exact Shapley weighting:

        phi_j = sum_{S subset N\\{j}} |S|!(d-|S|-1)!/d! * (v(S u {j}) - v(S))

    This is only tractable for a small number of features (the caller is responsible
    for enforcing feasibility).

    Args:
        value_function (BaseValueFunction): Cooperative game v(x, S) -> (value, const).
        X (np.ndarray): Instances to explain, shape (n, d).
        device (torch.device): Torch device for value-function evaluation.
        raw_score (bool, optional): Pass-through to the value function. Defaults to True.
        coalition_batch_size (int, optional): Max coalitions evaluated per forward call
            to bound memory. Defaults to 8192.

    Returns:
        tuple[np.ndarray, float]: Exact Shapley values of shape (n, d) and the value
            function's average constant v(empty).
    """
    n, d = X.shape
    masks = _enumerate_masks(d)  # (2**d, d)
    num_coalitions = masks.shape[0]
    popcount = masks.sum(axis=1).astype(np.int64)  # |S| for each coalition
    size_weights = _shapley_size_weights(d)  # weight indexed by |S|

    masks_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
    subset_ids = np.arange(num_coalitions, dtype=np.int64)

    # Precompute, per feature, the coalitions that exclude it and their +j counterparts.
    without_j = [subset_ids[((subset_ids >> j) & 1) == 0] for j in range(d)]
    with_j = [without_j[j] | (1 << j) for j in range(d)]
    weights_j = [size_weights[popcount[without_j[j]]] for j in range(d)]

    attributions = np.zeros((n, d), dtype=np.float64)
    avg_constant: float | None = None

    for i in range(n):
        x_i = torch.as_tensor(X[i], dtype=torch.float32, device=device).reshape(1, -1)
        x_rep = x_i.expand(num_coalitions, -1)

        v = np.empty(num_coalitions, dtype=np.float64)
        for start in range(0, num_coalitions, coalition_batch_size):
            end = min(start + coalition_batch_size, num_coalitions)
            preds, const = value_function(
                x_rep[start:end], masks_t[start:end], raw_score
            )
            v[start:end] = preds.detach().cpu().numpy().reshape(-1)

        if avg_constant is None:
            avg_constant = float(np.asarray(const.detach().cpu()).reshape(-1)[0])

        for j in range(d):
            attributions[i, j] = np.sum(weights_j[j] * (v[with_j[j]] - v[without_j[j]]))

    return attributions.astype(np.float32), float(avg_constant)


def save_exact_shap_values(
    dataset_name: str,
    model_name: str,
    attr: np.ndarray,
    avg_constant: float,
    split: str,
    value_function_name: str,
) -> None:
    """Save exact Shapley values (mirrors the KernelSHAP reference naming)."""
    np.save(
        CACHE_DIR
        / dataset_name
        / f"{model_name}_exact_shap_{value_function_name}_{split}.npy",
        attr,
    )
    np.save(
        CACHE_DIR
        / dataset_name
        / f"{model_name}_exact_shap_avg_constant_{value_function_name}_{split}.npy",
        np.array([avg_constant]),
    )


def compute_and_save_exact_shap_values(
    dataset_name: str,
    model_name: str,
    value_function: BaseValueFunction,
    X: pd.DataFrame,
    split: str,
    device: torch.device,
) -> np.ndarray:
    """
    Compute exact Shapley values for a split and cache them alongside the reference
    SHAP values.

    Args:
        dataset_name (str): Dataset name.
        model_name (str): Model name.
        value_function (BaseValueFunction): Value function defining the game.
        X (pd.DataFrame): Instances to explain.
        split (str): Split label used in the cache filename (e.g. "test").
        device (torch.device): Torch device for value-function evaluation.

    Returns:
        np.ndarray: Exact Shapley values of shape (len(X), num_features).
    """
    attr, avg_constant = compute_exact_shap_values(value_function, X.values, device)
    save_exact_shap_values(
        dataset_name, model_name, attr, avg_constant, split, value_function.name
    )
    return attr
