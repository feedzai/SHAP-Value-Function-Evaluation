import numpy as np
import torch

from experiment.paths import RESULTS_DIR


def load_amortised_attributions(experiment_name: str) -> np.ndarray:
    """
    Calculatethe amortised attributions using a trained Amortiser.

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        np.ndarray: Amortised attributions.
    """
    return np.load(RESULTS_DIR / experiment_name / "amortised_attr_test.npy")


def get_average_constant(experiment_name: str) -> float:
    """
    Get the average constant from a trained Amortiser.

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        float: Average constant.
    """
    return float(
        torch.load(
            RESULTS_DIR / experiment_name / "amortiser.pth",
            map_location=torch.device("cpu"),
        )["average_constant"]
    )
