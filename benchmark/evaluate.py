import json

import numpy as np
import torch
import os

from data.dataset import Dataset
from data.utils import load_ref_shap_values
from evaluation.benchmark import bootstrap_metrics
from evaluation.utils import load_amortised_attributions
from experiment.paths import RESULTS_DIR
from experiment.arg_parsers import parse_evaluation_args
from src.amortiser import InferenceAmortiser
from src.value_functions.utils import get_value_function


def main(experiment_name: str, n_bootstrap: int):
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(RESULTS_DIR / experiment_name):
        print(f"Experiment {experiment_name} doesn't exist.")
        return -1

    args = experiment_name.split("_")
    dataset_name = args[0]
    model_name = args[1]
    value_function_type = args[2]
    value_function_name = args[2] + "_" + args[3]
    if value_function_name == "baseline_zero":
        baseline_type = "zero"
        background_size = 1
    elif value_function_name == "baseline_mean":
        baseline_type = "mean"
        background_size = 1
    else:
        baseline_type = "zero"
        background_size = int(args[3].replace("bg", ""))

    dataset = Dataset(dataset_name, model_name)

    X_train = dataset.X_train
    y_train = dataset.y_train
    X_test = dataset.X_test
    y_test = dataset.y_test

    numerical_features = dataset.numerical_features
    categorical_features = dataset.categorical_features

    classifier = dataset.model
    amortiser = InferenceAmortiser.load(experiment_name)

    value_function = get_value_function(
        value_function_type,
        dataset,
        background_size,
        baseline_type,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # ===== Reference Attributions =============================================

    ref_shap = load_ref_shap_values(
        dataset_name, model_name, value_function_name, "test"
    )

    # ===== Amortised Attributions =============================================

    amortised_attr = load_amortised_attributions(experiment_name)

    # ===== Calculate Metrics =================================================

    results = bootstrap_metrics(
        n_bootstrap,
        X_test,
        y_test,
        ref_shap,
        amortised_attr,
        X_train,
        y_train,
        classifier,
        amortiser,
        numerical_features,
        categorical_features,
        value_function,
        seed=seed,
    )

    # ===== Save Results ======================================================

    with open(RESULTS_DIR / experiment_name / "results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = parse_evaluation_args()
    main(args.experiment_name, args.n_bootstrap)
