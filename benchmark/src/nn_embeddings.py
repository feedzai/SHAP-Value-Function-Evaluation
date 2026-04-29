from typing import Optional

import torch


class NNWithEmbeddings(torch.nn.Module):
    """
    Neural network with embeddings.
    """

    def __init__(
        self,
        input_dim_num: int,
        cardinalities: Optional[list[int]],
        device: torch.device,
        emb_dim: Optional[list[int]] = None,
        force_positive: bool = False,
    ):
        """
        Initialize the Neural network with embeddings.

        Args:
            input_dim_num (int): Number of numerical features.
            cardinalities (Optional[list[int]]): Cardinalities of the categorical
                                                 features.
            device (torch.device): Device to use.
            emb_dim (Optional[list[int]]): Embedding dimensions.
            force_positive (bool): Whether to force the output to be positive.
        """
        super().__init__()
        if cardinalities is not None:
            if emb_dim is None:
                emb_dim = [1 for _ in cardinalities]

            self.embeddings = torch.nn.ModuleList(
                [
                    torch.nn.Embedding(card, emb_dim[i])
                    for i, card in enumerate(cardinalities)
                ]
            ).to(device)

            total_emb_dim = sum(emb_dim)
            full_input_dim = input_dim_num + total_emb_dim
            output_dim = input_dim_num + len(cardinalities)
        else:
            self.embeddings = None
            full_input_dim = input_dim_num
            output_dim = input_dim_num

        self.net = torch.nn.Sequential(
            torch.nn.Linear(full_input_dim, 5 * full_input_dim),
            torch.nn.LayerNorm(5 * full_input_dim),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(5 * full_input_dim, 3 * output_dim),
            torch.nn.LayerNorm(3 * output_dim),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(3 * output_dim, output_dim),
            torch.nn.Softplus() if force_positive else torch.nn.Identity(),
        ).to(device)

        self.device = device

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_num: Numerical features.
            x_cat: Categorical features.

        Returns:
            torch.Tensor: Output.
        """
        x_num = x_num.to(self.device)
        x_cat = x_cat.to(self.device)

        if self.embeddings is not None:
            emb = [
                emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)
            ]
            x_cat_emb = torch.cat(emb, dim=1)
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num

        return self.net(x)

    def get_categorical_embeddings(self):
        """
        Get the categorical embeddings.

        Returns:
            list[torch.Tensor]: Categorical embeddings.
        """
        if self.embeddings is not None:
            embeddings = [emb.weight.detach().cpu().numpy() for emb in self.embeddings]
            return embeddings
        else:
            return None
