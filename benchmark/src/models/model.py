from abc import ABC, abstractmethod
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from experiment.paths import CACHE_DIR


class Model(ABC):
    @abstractmethod
    def __init__(self, dataset_name: str):
        """
        Model base class

        Args:
            dataset_name (str): name of the dataset
        """
        self.dataset_name = dataset_name
        self.model_name = None
        self.model = None

    def load(self) -> None:
        """Load the model from disk."""
        try:
            self.model = joblib.load(
                CACHE_DIR / self.dataset_name / f"{self.model_name}.pkl"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model {self.model_name} not found. Train and save the model first."
            )

    @abstractmethod
    def __call__(self, x: np.ndarray, raw_score: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        model_config: Dict[str, Any],
        save: bool = True,
    ):
        pass

    @abstractmethod
    def save(self) -> None:
        pass
