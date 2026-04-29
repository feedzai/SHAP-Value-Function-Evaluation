import os
from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from experiment.paths import RESULTS_DIR
from src.nn_embeddings import NNWithEmbeddings
from src.optimizer import ScheduledOptimizer
from src.sampler import shap_kernel_mask
from src.scaler import Scaler
from src.value_functions.base_value_function import BaseValueFunction


class Amortiser(torch.nn.Module):
    """
    Neural network amortiser that learns to predict SHAP values.
    """

    def __init__(
        self,
        device: torch.device,
        input_dim_num: int,
        cardinalities: Optional[list[int]],
        value_function: BaseValueFunction,
        scaler: Scaler,
    ):
        """
        Initialize the Amortiser.

        Args:
            device (torch.device): Device to use.
            input_dim_num (int): Number of numerical features.
            cardinalities (Optional[list[int]]): Cardinalities of the categorical
                                                 features.
            value_function (BaseValueFunction): SHAP value function.
            scaler (Optional[Scaler]): Scaler to scale the input features.
        """
        super().__init__()

        self.amortiser = NNWithEmbeddings(input_dim_num, cardinalities, device)

        self.input_dim_num = input_dim_num
        self.cardinalities = cardinalities
        self.value_function = value_function
        self.scaler = scaler
        self.device = device

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.


        Args:
            x_num (torch.Tensor): Numerical features.
            x_cat (torch.Tensor): Categorical features.

        Returns:
            torch.Tensor: Predicted SHAP values.
        """
        self.amortiser.eval()
        self.scaler.eval()
        return self.amortiser(self.scaler(x_num), x_cat)

    def train(
        self,
        train_loader: DataLoader[TensorDataset],
        val_loader: DataLoader[TensorDataset],
        num_epochs: int,
        lr_adam: float,
        lr_sgd: float,
        adam_epochs: int,
        warmup_epochs: int,
        n_repeats: int,
        reg: float,
    ):
        """
        Train the Amortiser.

        Args:
            train_loader (DataLoader[TensorDataset]): DataLoader for the training set.
            val_loader (DataLoader[TensorDataset]): DataLoader for the validation set.
            num_epochs (int): Number of training epochs.
            lr_adam (float): Adam learning rate.
            lr_sgd (float): SGD learning rate.
            adam_epochs (int): Number of Adam epochs.
            warmup_epochs (int): Number of warmup epochs.
            n_repeats (int): Number of repeats.
            reg (float): Regularization factor.
        """
        optimizer = ScheduledOptimizer(
            model=self.amortiser,
            adam_lr=lr_adam,
            sgd_lr=lr_sgd,
            total_epochs=num_epochs,
            adam_epochs=adam_epochs,
            warmup_epochs=warmup_epochs,
        )

        for epoch in range(num_epochs):
            epoch_loss, lr = self.train_epoch(
                epoch,
                train_loader,
                optimizer,
                n_repeats,
                reg,
            )
            val_loss = self.validate(val_loader)
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"LR: {lr:.5f} | Optimizer: {type(optimizer.optimizer).__name__} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Validation Loss: {val_loss:.4f} | "
            )

    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader[TensorDataset],
        optimizer: ScheduledOptimizer,
        n_repeats: int,
        reg: float,
    ) -> tuple[float, float]:
        """
        Train the Amortiser for one epoch.

        Args:
            epoch (int): Current epoch.
            train_loader (DataLoader[TensorDataset]): DataLoader for the training set.
            optimizer (ScheduledOptimizer): Optimizer.
            n_repeats (int): Number of repeats.
            reg (float): Regularization factor.

        Returns:
            tuple[float, float]: Epoch average loss and learning rate.
        """
        _ = self.amortiser.train()

        lr = optimizer.update(epoch)

        total_loss = 0

        for batch_idx, (x, x_num, x_cat, y_hat) in enumerate(train_loader):
            x: torch.Tensor = x.to(self.device)
            x_num: torch.Tensor = x_num.to(self.device)
            x_cat: torch.Tensor = x_cat.to(self.device)
            y_hat: torch.Tensor = y_hat.to(self.device)

            x_repeated = x.repeat_interleave(n_repeats, dim=0)

            with torch.no_grad():
                mask = shap_kernel_mask(x_repeated)
                masked_preds, avg = self.value_function(x_repeated, mask)

            attr = self.amortiser(self.scaler(x_num), x_cat)
            attr_repeated = attr.repeat_interleave(n_repeats, dim=0)
            masked_attr = mask * attr_repeated

            shap_loss = (masked_preds - avg - torch.sum(masked_attr, dim=1)) ** 2
            eff_loss = (y_hat - avg - torch.sum(attr, dim=1)) ** 2

            loss = shap_loss.mean() + reg * eff_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.amortiser.parameters(), max_norm=1.0)
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d} | "
                    f"LR: {lr:.5f} | "
                    f"Optimizer: {type(optimizer.optimizer).__name__} | "
                    f"Loss: {loss:.4f} | "
                    f"Batch {batch_idx + 1:4d}/{len(train_loader):4d}",
                    end="\r",
                    flush=True,
                )

            total_loss += loss.item()

        return total_loss / len(train_loader), lr

    @torch.no_grad()
    def validate(self, val_loader: DataLoader[TensorDataset]):
        """
        Validate the Amortiser.

        Args:
            val_loader (DataLoader[TensorDataset]): DataLoader for the validation set.

        Returns:
            float: Validation loss.
        """
        self.amortiser.eval()  # Deactivate Dropout layers
        total_mse = 0
        for x_num, x_cat, ref_attr in val_loader:
            x_num: torch.Tensor = x_num.to(self.device)
            x_cat: torch.Tensor = x_cat.to(self.device)
            ref_attr: torch.Tensor = ref_attr.to(self.device)

            amortised_attr = self.amortiser(self.scaler(x_num), x_cat)
            total_mse += torch.mean((amortised_attr - ref_attr) ** 2)

        return total_mse / len(val_loader)

    def save_for_inference(self, experiment_name: str):
        """
        Save everything needed for instanciating a lightweight amortiser.

        Args:
            experiment_name (str): Name of the experiment.
        """
        if not os.path.exists(RESULTS_DIR / experiment_name):
            os.makedirs(RESULTS_DIR / experiment_name)

        save_dict = {
            "amortiser_state_dict": self.amortiser.state_dict(),
            "input_dim_num": self.input_dim_num,
            "cardinalities": self.cardinalities,
            "scaler_state_dict": self.scaler.state_dict(),
            "scaler_n_features": self.scaler.n_features,
            "scaler_method": self.scaler.method,
            "scaler_saturation": self.scaler.saturation,
            "average_constant": self.value_function.average_constant,
        }
        torch.save(save_dict, RESULTS_DIR / experiment_name / "amortiser.pth")


class InferenceAmortiser(torch.nn.Module):
    """
    Lightweight amortiser for inference only.
    Contains only the neural network and scaler, no training dependencies.
    """

    def __init__(
        self,
        amortiser_state_dict: dict,
        input_dim_num: int,
        cardinalities: Optional[list[int]],
        scaler_state_dict: dict,
        scaler_n_features: int,
        scaler_method: str,
        scaler_saturation: float,
        average_constant: float,
    ):
        """
        Initialize the InferenceAmortiser.

        Args:
            amortiser_state_dict (dict): State dict of the neural network.
            input_dim_num (int): Number of numerical features.
            cardinalities (Optional[list[int]]): Cardinalities of the categorical
                                                 features.
            scaler_state_dict (dict): State dict of the scaler.
            scaler_n_features (int): Number of features to scale.
            scaler_method (str): Scaling method.
            scaler_saturation (float): Saturation level for the scaler.
            average_constant (float): Average constant for the value function.

        """
        super().__init__()

        self.amortiser = NNWithEmbeddings(
            input_dim_num, cardinalities, device=torch.device("cpu")
        )
        self.amortiser.load_state_dict(amortiser_state_dict)

        self.scaler = Scaler(
            n_features=scaler_n_features,
            method=scaler_method,
            saturation=scaler_saturation,
        )
        self.scaler.load_state_dict(scaler_state_dict, strict=False)

        self.average_constant = average_constant

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Predict SHAP values for given input features.

        Args:
            x_num (torch.Tensor): Numerical features.
            x_cat (torch.Tensor): Categorical features.

        Returns:
            torch.Tensor: Predicted SHAP values.
        """
        self.amortiser.eval()
        self.scaler.eval()
        return self.amortiser(self.scaler(x_num), x_cat)

    def get_average_constant(self) -> float:
        return self.average_constant

    @classmethod
    def load(cls, experiment_name: str) -> "InferenceAmortiser":
        """Load inference-only amortiser from file."""
        save_dict = torch.load(
            RESULTS_DIR / experiment_name / "amortiser.pth",
            map_location=torch.device("cpu"),
        )

        return cls(**save_dict)
