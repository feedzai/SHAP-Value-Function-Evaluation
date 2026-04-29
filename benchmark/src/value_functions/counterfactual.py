import warnings

import pandas as pd
import torch

from experiment.counterfactuals import (
    compute_counterfactuals,
    initialize_counterfactual_generator,
)
from data.utils import stratified_sampling
from src.models.model import Model
from src.value_functions.base_value_function import BaseValueFunction

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="dice_ml.*",
)


class CounterfactualValueFunction(BaseValueFunction):
    def __init__(
        self,
        classifier: Model,
        device: torch.device,
        data: pd.DataFrame,
        target_column: str,
        numerical_features: list[str],
        background_size: int = 100,
    ):
        """
        Counterfactual value function.

        Args:
            classifier (Model): Classifier model.
            device (torch.device): Device to use for calculation.
            data (np.ndarray): Data for background calculation.
            target_column (str): Target column name.
            numerical_features (list[str]): Numerical features.
            background_size (int, optional): Background size. Defaults to 100.
        """
        super().__init__(classifier, device)
        self.name = "counterfactual_bg" + str(background_size)

        self.background_size = background_size
        self.features = data.columns.tolist()
        self.features.remove(target_column)

        y_pred = self.score_classifier(
            data.drop(columns=[target_column]).values, raw_score=False
        )
        sample, _ = stratified_sampling(data.values, y_pred, self.background_size)
        data_sample = pd.DataFrame(
            sample,
            columns=data.columns,
        )

        self.cf_generator = initialize_counterfactual_generator(
            data_sample, target_column, numerical_features, classifier
        )

        background = data_sample.drop(columns=[target_column])
        self.background_tensor = torch.tensor(
            background.values,
            dtype=torch.float32,
            device=self.device,
        )

        precomputed_cf = compute_counterfactuals(
            self.cf_generator,
            background,
            self.background_size,
            desired_class=0,
        )

        self.precomputed_cf_tensor = torch.tensor(
            precomputed_cf, dtype=torch.float32, device=self.device
        )

        self.average_constant = torch.tensor(
            self.score_classifier(
                precomputed_cf.reshape(-1, precomputed_cf.shape[-1])
            ).mean(),
            dtype=torch.float32,
            device=self.device,
        )

        print("Counterfactuals pre-calculated and cached.")

    def _find_closest_indices(
        self, x: torch.Tensor, train_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Find the indices of the closest training instances for each instance in x.

        Args:
            x: Batch of instances (batch_size, n_features)
            train_data: Training data (n_train, n_features)

        Returns:
            Indices of closest training instances (batch_size,)
        """
        distances = torch.cdist(x, train_data, p=2)
        return torch.argmin(distances, dim=1)

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

        closest_indices = self._find_closest_indices(x, self.background_tensor)
        cf_array = self.precomputed_cf_tensor[closest_indices]

        masked_inputs = (S * x)[:, None, :] + (1 - S)[:, None, :] * cf_array
        flat_masked = masked_inputs.reshape(
            batch_size * self.background_size, num_features
        )

        preds = torch.tensor(
            self.score_classifier(flat_masked.detach().cpu().numpy(), raw_score),
            dtype=torch.float32,
            device=self.device,
        ).reshape(batch_size, self.background_size)

        return preds.mean(dim=1), self.average_constant
