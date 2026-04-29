from typing import Literal

import numpy as np
import torch

from src.models.model import Model
from src.value_functions.base_value_function import BaseValueFunction


class BaselineValueFunction(BaseValueFunction):
    def __init__(
        self,
        classifier: Model,
        baseline_type: Literal["zero", "mean"],
        data: np.ndarray,
        device: torch.device,
    ):
        """
        Baseline value function.

        Args:
            classifier (Model): Classifier model.
            baseline_type (Literal): Type of baseline. Options: zero, mean.
            data (np.ndarray): Data for background calculation.
            device (torch.device): Device to use for calculation.
        """
        super().__init__(classifier, device)
        self.name = "baseline_" + baseline_type

        if baseline_type == "zero":
            self.baseline = torch.zeros(data.shape[1], device=device)
        elif baseline_type == "mean":
            self.baseline = torch.tensor(
                np.mean(data, axis=0), dtype=torch.float32, device=device
            )

        self.average_constant = torch.tensor(
            self.score_classifier(self.baseline.reshape(1, -1).detach().cpu().numpy()),
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
        masked_inputs = (S * x) + (1 - S) * self.baseline

        # Predict and reshape to (batch_size, background_size)
        preds = torch.tensor(
            self.score_classifier(masked_inputs.detach().cpu().numpy(), raw_score),
            dtype=torch.float32,
            device=self.device,
        )

        return preds, self.average_constant
