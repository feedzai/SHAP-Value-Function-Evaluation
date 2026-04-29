from typing import Literal

import pandas as pd
import torch

from data.dataset import Dataset
from src.value_functions.base_value_function import BaseValueFunction
from src.value_functions.baseline import BaselineValueFunction
from src.value_functions.conditional import ConditionalValueFunction
from src.value_functions.counterfactual import CounterfactualValueFunction
from src.value_functions.filtered_conditional import FilteredConditionalValueFunction
from src.value_functions.jointmarginal import JointMarginalValueFunction
from src.value_functions.marginal import MarginalValueFunction
from src.value_functions.uniform import UniformValueFunction


def get_value_function(
    value_function_name: str,
    dataset: Dataset,
    background_size: int,
    baseline_type: Literal["zero", "mean"],
    device: torch.device,
) -> BaseValueFunction:
    """
    Create a value function with the corresponding parameters given its name.

    Args:
        value_function_name (str): Name of the value function.
        dataset (Dataset): Dataset.
        background_size (int): Background set size.
        baseline_type (Literal["zero", "mean"]): Baseline type.
        device (torch.device): Device to use.

    Returns:
        BaseValueFunction: Value function.
    """
    data = dataset.X_train.values
    classifier = dataset.model

    if value_function_name == "baseline":
        return BaselineValueFunction(classifier, baseline_type, data, device)

    elif value_function_name == "marginal":
        return MarginalValueFunction(
            classifier,
            device,
            data,
            background_size,
        )

    elif value_function_name == "uniform":
        return UniformValueFunction(
            classifier,
            device,
            data,
            background_size,
        )

    elif value_function_name == "jointmarginal":
        return JointMarginalValueFunction(
            classifier,
            device,
            data,
            background_size,
        )

    elif value_function_name == "conditional":
        return ConditionalValueFunction(
            classifier,
            device,
            data,
            background_size,
        )

    elif value_function_name == "counterfactual":
        return CounterfactualValueFunction(
            classifier,
            device,
            pd.concat([dataset.X_train, dataset.y_train], axis=1),
            str(dataset.y_train.name),
            dataset.numerical_features,
            background_size,
        )

    elif value_function_name == "filteredconditional":
        return FilteredConditionalValueFunction(
            classifier,
            device,
            data,
            background_size,
        )

    raise ValueError(f"Value function {value_function_name} not found")
