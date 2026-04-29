from typing import Optional

import torch


class Scaler(torch.nn.Module):
    """
    Scaler that applies robust scaling followed by a saturation function.
    """

    def __init__(
        self,
        X: Optional[torch.Tensor] = None,
        n_features: Optional[int] = None,
        method: str = "tanh",
        saturation: float = 3.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Scaler.

        Args:
            X (Optional[torch.Tensor]): Data to fit the scaler on.
            n_features (Optional[int]): Number of features.
            method (str): Saturation method.
            saturation (float): Saturation scale.
            device (Optional[torch.device]): Device to use for computation.
        """
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        if n_features is None:
            if X is None:
                raise ValueError("Either X or n_features must be provided.")
            else:
                n_features = X.shape[1]
        self.n_features = n_features

        self.method = method.lower()
        if self.method not in {"tanh", "log"}:
            raise ValueError(f"Unknown saturation method: {self.method}")

        self.register_buffer("center_", torch.zeros(n_features, device=device))
        self.register_buffer("scale_", torch.ones(n_features, device=device))
        self.saturation = saturation
        self.device = device

        if X is not None:
            self.fit(X)

    def fit(self, X: torch.Tensor):
        """
        Fit the scaler by computing median and IQR.

        Args:
            X (torch.Tensor): Data to fit the scaler on.
        """
        X = X.to(self.device)

        median = torch.median(X, dim=0)[0]
        q1 = torch.quantile(X, 0.25, dim=0)
        q3 = torch.quantile(X, 0.75, dim=0)
        iqr = q3 - q1

        iqr = torch.where(iqr == 0, torch.ones_like(iqr), iqr)

        self.center_ = median.to(self.device)
        self.scale_ = iqr.to(self.device)

    def _apply_saturation(self, X_scaled: torch.Tensor) -> torch.Tensor:
        """
        Apply the saturation function.

        Args:
            X_scaled (torch.Tensor): Scaled data.

        Returns:
            torch.Tensor: Saturated data.
        """
        if self.method == "tanh":
            return self.saturation * torch.tanh(X_scaled / self.saturation)
        elif self.method == "log":
            return torch.log(1.0 + torch.abs(X_scaled)) * torch.sign(X_scaled)
        else:
            raise ValueError(f"Unknown saturation method: {self.method}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: applies robust scaling followed by saturation.

        Args:
            X (torch.Tensor): Input tensor to scale.

        Returns:
            torch.Tensor: Scaled and saturated tensor.
        """
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        X = X.to(self.device)
        X_scaled = (X - self.center_) / self.scale_
        X_transformed = self._apply_saturation(X_scaled)

        return X_transformed
