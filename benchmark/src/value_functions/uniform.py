import numpy as np
import torch

from src.models.model import Model
from src.value_functions.marginal import MarginalValueFunction


class UniformValueFunction(MarginalValueFunction):
    def __init__(
        self,
        classifier: Model,
        device: torch.device,
        data: np.ndarray,
        background_size: int = 100,
    ):
        """
        Uniform value function.

        Args:
            classifier (Model): Classifier model.
            device (torch.device): Device to use for calculation.
            data (np.ndarray): Data for background calculation.
            background_size (int, optional): Background size. Defaults to 100.
        """
        super().__init__(classifier, device, data, background_size)
        self.name = "uniform_bg" + str(background_size)

    def _generate_background(self, X: np.ndarray, background_size: int) -> np.ndarray:
        """
        Generate background data from uniform distribution.

        Args:
            X (np.ndarray): Data to generate background from.
            background_size (int): Size of background data.

        Returns:
            np.ndarray: Background data.
        """
        X_tensor = torch.from_numpy(X).to(self.device)
        min_values = X_tensor.min(dim=0)[0].float()
        max_values = X_tensor.max(dim=0)[0].float()

        uniform_distribution = torch.distributions.Uniform(min_values, max_values)
        background: torch.Tensor = uniform_distribution.rsample(
            sample_shape=torch.Size([background_size])
        )

        return background.detach().cpu().numpy()
