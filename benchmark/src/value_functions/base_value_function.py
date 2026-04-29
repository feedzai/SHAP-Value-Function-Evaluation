from abc import ABC, abstractmethod

import numpy as np
import torch

from src.models.model import Model


class BaseValueFunction(ABC):
    average_constant: torch.Tensor
    name: str

    def __init__(
        self,
        classifier: Model,
        device: torch.device,
    ):
        """
        Base class for SHAP value functions.


        Args:
            classifier (Model): Classifier model.
            device (torch.device): Device to use.
        """
        self.classifier = classifier
        self.device = device

    @abstractmethod
    def __call__(
        self, x: torch.Tensor, S: torch.Tensor, raw_score: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def score_classifier(self, x: np.ndarray, raw_score: bool = True) -> np.ndarray:
        """
        Get model score for given input features.

        Args:
            x (np.ndarray): Input features.
            raw_score (bool, optional): Whether to return raw score or probability.

        Returns:
            np.ndarray: Model score.
        """
        return self.classifier(x, raw_score=raw_score)
