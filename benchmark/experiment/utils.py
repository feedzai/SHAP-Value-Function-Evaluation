from typing import Optional

import yaml

from experiment.paths import RESULTS_DIR


def save_experiment_config(
    experiment_name: str,
    dataset_name: str,
    value_function_name: str,
    num_epochs: int,
    input_dim_num: int,
    cardinalities: Optional[list[int]],
    batch_size: int,
    reg: float,
    lr_adam: float,
    lr_sgd: float,
    adam_epochs: int,
    warmup_epochs: int,
    n_repeats: int,
):
    """
    Save the experiment configuration to a YAML file.
    """

    experiment_config = {
        "dataset_name": dataset_name,
        "value_function_name": value_function_name,
        "num_epochs": num_epochs,
        "input_dim_num": input_dim_num,
        "cardinalities": cardinalities,
        "batch_size": batch_size,
        "reg": reg,
        "lr_adam": lr_adam,
        "lr_sgd": lr_sgd,
        "adam_epochs": adam_epochs,
        "warmup_epochs": warmup_epochs,
        "n_repeats": n_repeats,
    }

    with open(RESULTS_DIR / experiment_name / "experiment_config.yaml", "w") as f:
        yaml.dump(experiment_config, f)


def generate_experiment_name(
    dataset_name: str,
    model_name: str,
    value_function_name: str,
    num_epochs: int,
    n_repeats: int,
) -> str:
    """
    Generate a unique experiment name.

    Args:
        dataset_name (str): Name of the dataset.
        value_function_name (str): Name of the value function.
        num_epochs (int): Number of training epochs.
        n_repeats (int): Number of repeats.

    Returns:
        str: Unique experiment name.
    """
    experiment_name = (
        f"{dataset_name}_{model_name}_{value_function_name}"
        f"_epoch{num_epochs}_repeats{n_repeats}"
    )
    return experiment_name
