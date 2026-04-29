import numpy as np
import torch

from data.utils import stratified_sampling
from src.models.model import Model
from src.value_functions.base_value_function import BaseValueFunction


class MarginalValueFunction(BaseValueFunction):
    def __init__(
        self,
        classifier: Model,
        device: torch.device,
        data: np.ndarray,
        background_size: int = 100,
    ):
        """
        Marginal value function.

        Args:
            classifier (Model): Classifier model.
            device (torch.device): Device to use for calculation.
            data (np.ndarray): Data for background calculation.
            background_size (int, optional): Background size. Defaults to 100.
        """
        super().__init__(classifier, device)
        self.name = "marginal_bg" + str(background_size)
        self.background_size = background_size

        background = self._generate_background(data, background_size)
        self.average_constant = torch.tensor(
            self.score_classifier(background).mean(),
            dtype=torch.float32,
            device=device,
        )

        self.background = torch.from_numpy(background).to(self.device)

    def _generate_background(self, X: np.ndarray, background_size: int) -> np.ndarray:
        """
        Generate background data.

        Args:
            X (np.ndarray): Data.
            background_size (int): Background size.

        Returns:
            np.ndarray: Background data.
        """
        y_pred = self.score_classifier(X, raw_score=False)
        background, _ = stratified_sampling(X, y_pred, background_size)
        return background

    def __call__(
        self, x: torch.Tensor, S: torch.Tensor, raw_score: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get predicitons for masked inputs.

        Args:
            x (torch.Tensor): Input tensor.
            S (torch.Tensor): Masks.
            raw_score (bool, optional): Whether to return raw scores. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Masked predictions and constant baseline.
        """
        batch_size, num_features = x.shape

        # Apply mask S on x and (1 - S) on background
        masked_inputs = (S * x)[:, None, :] + (1 - S)[:, None, :] * self.background[
            None, :, :
        ]
        flat_masked = masked_inputs.reshape(
            batch_size * self.background_size, num_features
        )

        # Predict and reshape to (batch_size, background_size)
        preds = torch.tensor(
            self.score_classifier(flat_masked.detach().cpu().numpy(), raw_score),
            dtype=torch.float32,
            device=self.device,
        ).reshape(batch_size, self.background_size)

        return preds.mean(dim=1), self.average_constant
