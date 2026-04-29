import numpy as np
import torch

from data.utils import stratified_sampling
from src.models.model import Model
from src.value_functions.base_value_function import BaseValueFunction


class ConditionalValueFunction(BaseValueFunction):
    def __init__(
        self,
        classifier: Model,
        device: torch.device,
        data: np.ndarray,
        background_size: int = 100,
        temperature: float = 0.5,
    ):
        """
        Conditional value function.

        Args:
            classifier (Model): Classifier model.
            device (torch.device): Device to use for calculation.
            data (np.ndarray): Data for background calculation.
            background_size (int, optional): Background size. Defaults to 100.
            temperature (float, optional): Temperature for kernel. Defaults to 0.5.
        """
        super().__init__(classifier, device)
        self.name = "conditional_bg" + str(background_size)
        self.background_size = background_size
        self.temperature = temperature

        self.stds = torch.tensor(
            np.std(data, axis=0), dtype=torch.float32, device=device
        )

        y_pred = self.score_classifier(data, raw_score=False)
        bg, _ = stratified_sampling(data, y_pred, self.background_size)

        self.background = torch.tensor(bg, dtype=torch.float32, device=device)

        self.average_constant = torch.tensor(
            self.score_classifier(bg).mean(),
            dtype=torch.float32,
            device=device,
        )

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

        weights = x[:, None, :] - self.background[None, :, :]
        weights = (weights / (self.stds + 1e-8)) * S[:, None, :]
        dist = torch.abs(weights).mean(dim=2)
        kernel = torch.exp(-dist / self.temperature)

        preds = (preds * kernel).sum(dim=1) / kernel.sum(dim=1)
        return preds, self.average_constant
