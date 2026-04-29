from typing import Any

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class ScheduledOptimizer:
    """
    Scheduled optimizer that switches from Adam to SGD after a certain number of epochs.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        adam_lr: float,
        sgd_lr: float,
        total_epochs: int,
        adam_epochs: int,
        warmup_epochs: int,
    ):
        """
        Initialize the Scheduled optimizer.

        Args:
            model (torch.nn.Module): Model to optimize.
            adam_lr (float): Adam learning rate.
            sgd_lr (float): SGD learning rate.
            total_epochs (int): Total number of epochs.
            adam_epochs (int): Number of Adam epochs.
            warmup_epochs (int): Number of warmup epochs.
        """
        assert warmup_epochs < adam_epochs, (
            "warmup_epochs must be less than adam_epochs"
        )
        assert warmup_epochs < (total_epochs - adam_epochs), (
            "warmup_epochs must be less than (total_epochs - adam_epochs)"
        )

        self.model = model
        self.adam_lr = adam_lr
        self.sgd_lr = sgd_lr
        self.total_epochs = total_epochs
        self.adam_epochs = adam_epochs
        self.warmup_epochs = warmup_epochs

        self.optimizer: Optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        self.is_sgd: bool = False

    def update(self, epoch: int) -> float:
        """
        Update the optimizer's learning rate.

        Args:
            epoch (int): Current epoch.

        Returns:
            float: Learning rate.
        """
        if not self.is_sgd:
            # Warmup for Adam
            if epoch < self.warmup_epochs:
                lr = 1e-5 + (self.adam_lr - 1e-5) * (epoch / self.warmup_epochs)
            else:
                lr = self.adam_lr
        else:
            # Warmup for SGD after switching
            if epoch < self.adam_epochs + self.warmup_epochs:
                sgd_epoch = epoch - self.adam_epochs
                lr = 1e-5 + (self.sgd_lr - 1e-5) * (sgd_epoch / self.warmup_epochs)
            else:
                # Cosine decay
                decay_start = self.adam_epochs + self.warmup_epochs
                schedule_numerator = epoch - decay_start
                schedule_denominator = self.total_epochs - decay_start
                progress = schedule_numerator / schedule_denominator
                lr = 0.5 * self.sgd_lr * (1 + np.cos(np.pi * progress))

        if epoch == self.adam_epochs and not self.is_sgd:
            print(f"Switching to SGD with momentum at epoch {epoch}")
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=1e-5, momentum=0.9
            )
            lr = 1e-5
            self.is_sgd = True
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        return lr

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)
