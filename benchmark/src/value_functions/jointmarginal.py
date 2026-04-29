import numpy as np
import torch

from src.models.model import Model
from src.value_functions.marginal import MarginalValueFunction
from data.utils import stratified_sampling


class JointMarginalValueFunction(MarginalValueFunction):
    def __init__(
        self,
        classifier: Model,
        device: torch.device,
        data: np.ndarray,
        background_size: int = 100,
    ):
        """
        Joint-Marginal value function.

        Args:
            classifier (Model): Classifier model.
            device (torch.device): Device to use for calculation.
            data (np.ndarray): Data for background calculation.
            background_size (int, optional): Background size. Defaults to 100.
        """
        super().__init__(classifier, device, data, background_size)
        self.name = "jointmarginal_bg" + str(background_size)

    def _generate_background(self, X: np.ndarray, background_size: int) -> np.ndarray:
        """
        Generate background data.

        Args:
            X (np.ndarray): Data.
            background_size (int): Background size.

        Returns:
            np.ndarray: Background data.
        """
        background = np.zeros((background_size, X.shape[1]))
        y_pred = self.score_classifier(X, raw_score=False)
        for feature in range(X.shape[1]):
            bg, _ = stratified_sampling(X, y_pred, background_size)
            background[:, feature] = bg[:, feature]

        return background
