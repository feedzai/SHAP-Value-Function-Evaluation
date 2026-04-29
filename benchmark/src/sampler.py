import torch


def shap_kernel_mask(batch: torch.Tensor) -> torch.Tensor:
    """
    Sample a mask for the SHAP kernel.

    Args:
        batch (torch.Tensor): Batch of data.

    Returns:
        torch.Tensor: Mask.
    """
    batch_size, num_features = batch.shape
    device = batch.device

    # SHAP kernel weights
    arange = torch.arange(1, num_features, device=device)
    w = 1.0 / (arange * (num_features - arange))
    w /= w.sum()

    # Sample how many features to include for each item in the batch
    included = torch.distributions.Categorical(probs=w).sample(
        sample_shape=torch.Size([batch_size])
    )

    # Create base masks
    base_masks = torch.tril(torch.ones((num_features - 1, num_features), device=device))
    S = base_masks[included]

    # Shuffle each row independently
    rand = torch.rand((batch_size, num_features), device=device)
    indices = rand.argsort(dim=1)

    return torch.gather(S, 1, indices)
