import warnings
from typing import Literal

import numpy as np
import torch

from data.dataset import Dataset
from experiment.arg_parsers import parse_exact_shap_args
from experiment.exact_shap import compute_and_save_exact_shap_values
from src.value_functions.utils import get_value_function

warnings.filterwarnings("ignore")

# Exact Shapley enumerates 2**d coalitions per instance. This guard prevents an
# accidental multi-hour (or intractable) run on high-dimensional datasets. Only
# small feature counts (e.g. MaternalRisk, d=6) are feasible; override with --force.
MAX_FEATURES = 16


def main(
    dataset_name: str,
    model_name: str,
    value_function_name: str,
    background_size: int,
    baseline_type: Literal["zero", "mean"],
    split: str,
    force: bool,
):
    # Seed before building the dataset/value function so the sampled background
    # matches the one used to train the amortiser (train.py also seeds with 0).
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(dataset_name, model_name)

    num_features = dataset.X_train.shape[1]
    if num_features > MAX_FEATURES and not force:
        raise SystemExit(
            f"{dataset_name} has {num_features} features -> 2**{num_features} "
            f"coalitions per instance, which is infeasible for exact Shapley. "
            f"Exact computation is intended for small feature counts (e.g. "
            f"MaternalRisk). Re-run with --force to override."
        )

    value_function = get_value_function(
        value_function_name,
        dataset,
        background_size,
        baseline_type,
        device,
    )

    X = dataset.X_test if split == "test" else dataset.X_val

    print(
        f"Computing exact Shapley values for {dataset_name}/{model_name} "
        f"with value function '{value_function.name}' on the {split} split "
        f"({len(X)} instances x 2**{num_features} coalitions)."
    )

    compute_and_save_exact_shap_values(
        dataset_name, model_name, value_function, X, split, device
    )

    print("Done.")


if __name__ == "__main__":
    args = parse_exact_shap_args()
    main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        value_function_name=args.value_function,
        background_size=args.background_size,
        baseline_type=args.baseline_type,
        split=args.split,
        force=args.force,
    )
