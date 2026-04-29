import argparse

from experiment.constants import DATASETS, MODELS, VALUE_FUNCTIONS, DEFAULTS


def parse_preprocess_args() -> argparse.Namespace:
    """
    Parse command line arguments for the preprocessing script.
    """
    parser = argparse.ArgumentParser(description="Preprocess and cache dataset")

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to process",
        choices=DATASETS,
    )

    return parser.parse_args()


def parse_train_classifier_args() -> argparse.Namespace:
    """
    Parse command line arguments for the classifier training script.
    """
    parser = argparse.ArgumentParser(description="Train a classifier for the dataset")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to train a classifier model on",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model to use",
        choices=MODELS,
    )

    return parser.parse_args()


def parse_reference_shap_args() -> argparse.Namespace:
    """
    Parse command line arguments for the reference SHAP calculation script.
    """
    parser = argparse.ArgumentParser(description="Calculate reference SHAP values")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to calculate reference SHAP values for",
        choices=DATASETS,
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model to use",
        choices=MODELS,
    )
    parser.add_argument(
        "--background_size",
        type=int,
        default=DEFAULTS["background_size"],
        help="Background size",
    )
    return parser.parse_args()


def parse_experiment_args() -> argparse.Namespace:
    """
    Parse command line arguments for the amortiser training script.
    """
    parser = argparse.ArgumentParser(
        description="Train an amortised explainer to predict SHAP values"
    )

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name to use",
        choices=DATASETS,
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model to use",
        choices=MODELS,
    )
    parser.add_argument(
        "value_function",
        type=str,
        help="Name of value function to use",
        choices=VALUE_FUNCTIONS,
    )
    parser.add_argument(
        "--background_size",
        type=int,
        default=DEFAULTS["background_size"],
        help="Background size",
    )
    parser.add_argument(
        "--baseline_type",
        type=str,
        choices=["zero", "mean"],
        default=DEFAULTS["baseline_type"],
        help="Baseline type for baseline value function (e.g. zero, mean)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULTS["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr_adam", type=float, default=DEFAULTS["lr_adam"], help="Adam learning rate"
    )
    parser.add_argument(
        "--lr_sgd", type=float, default=DEFAULTS["lr_sgd"], help="SGD learning rate"
    )
    parser.add_argument(
        "--adam_epochs",
        type=int,
        default=DEFAULTS["adam_epochs"],
        help="Number of Adam epochs",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=DEFAULTS["warmup_epochs"],
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--n_repeats", type=int, default=DEFAULTS["n_repeats"], help="Number of repeats"
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=DEFAULTS["reg"],
        help="Regularization parameter",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULTS["batch_size"], help="Batch size"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save the amortised explainer"
    )

    return parser.parse_args()


def parse_evaluation_args() -> argparse.Namespace:
    """
    Parse command line arguments for the evaluation script.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an amortised explainer against reference SHAP values"
    )
    parser.add_argument(
        "experiment_name", type=str, help="Name of the experiment to evaluate"
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=50, help="Number of bootstrap samples"
    )
    return parser.parse_args()


def parse_benchmark_args() -> argparse.Namespace:
    """
    Parse command line arguments for the benchmark script.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark value function formulations"
    )
    parser.add_argument(
        "dataset_name", type=str, help="Name of the dataset to benchmark"
    )
    parser.add_argument("model_name", type=str, help="Name of the model to use")
    parser.add_argument(
        "--background_size",
        type=int,
        default=DEFAULTS["background_size"],
        help="Background size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULTS["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--n_repeats", type=int, default=DEFAULTS["n_repeats"], help="Number of repeats"
    )
    return parser.parse_args()
