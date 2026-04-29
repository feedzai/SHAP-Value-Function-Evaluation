import numpy as np
from scipy.stats import spearmanr


def reconstruction_quality(attr: np.ndarray, pred: np.ndarray) -> float:
    """
    Reconstruction quality of the amortiser using mean squared error.

    Args:
        attr (np.ndarray): Reference attributions.
        pred (np.ndarray): Estimated attributions.

    Returns:
        (float) Mean squared error between reference and estimated attributions.

    """
    return np.mean((attr - pred) ** 2, dtype=float)


def top_k_recall(attr: np.ndarray, pred: np.ndarray, k: int) -> float:
    """
    Top-k recall of the amortiser.

    Args:
        attr (np.ndarray): Reference attributions.
        pred (np.ndarray): Estimated attributions.
        k (int): Number of top attributes to consider.

    Returns
        (float): Top-k recall.

    """
    if k >= attr.shape[1]:
        return 1.0

    reference_ranking = np.argsort(-np.array(abs(attr), dtype=float))
    estimated_ranking = np.argsort(-np.array(abs(pred), dtype=float))

    reference_ranking = [set(row) for row in reference_ranking[:, :k]]
    estimated_ranking = [set(row) for row in estimated_ranking[:, :k]]

    intersection = [
        reference_ranking[i] & estimated_ranking[i]
        for i in range(len(reference_ranking))
    ]
    intersection = [len(row) / k for row in intersection]

    return np.mean(intersection, dtype=float)


def correlation(attr: np.ndarray, pred: np.ndarray) -> float:
    """
    Cross agreement between the reference and estimated attributions.

    Args:
        attr (np.ndarray): Reference attributions.
        pred (np.ndarray): Estimated attributions.

    Returns
        (float): Mean Spearman correlation.

    """
    correlations = np.array([spearmanr(f, k)[0] for f, k in zip(attr, pred)])
    return np.mean(correlations, dtype=float)
