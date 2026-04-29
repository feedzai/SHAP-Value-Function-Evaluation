import numpy as np
import torch

from src.models.model import Model
from src.value_functions.marginal import MarginalValueFunction


class FilteredConditionalValueFunction(MarginalValueFunction):
    def __init__(
        self,
        classifier: Model,
        device: torch.device,
        data: np.ndarray,
        background_size: int = 100,
    ):
        """
        Filtered Conditional value function.

        Args:
            classifier (Model): Classifier model.
            device (torch.device): Device to use for calculation.
            data (np.ndarray): Data for background calculation.
            background_size (int, optional): Background size. Defaults to 100.
        """
        super().__init__(classifier, device, data, background_size)
        self.name = "filteredconditional_bg" + str(background_size)

    def _generate_background(self, X: np.ndarray, background_size: int) -> np.ndarray:
        """
        Generate background data based on classifier scores.

        Args:
            X (np.ndarray): Input data.
            background_size (int): Background size.

        Returns:
            np.ndarray: Background data.

        """
        y_pred = self.score_classifier(X, raw_score=False)
        idx = np.argsort(y_pred)[:background_size]

        return X[idx]
