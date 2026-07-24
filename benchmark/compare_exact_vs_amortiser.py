import argparse
import json

import numpy as np

from evaluation.amortiser_alignment_metrics import (
    correlation,
    reconstruction_quality,
    top_k_recall,
)
from evaluation.utils import load_amortised_attributions
from experiment.paths import CACHE_DIR, RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact Shapley values against amortised attributions"
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Name of the trained experiment (as produced by train.py --save)",
    )
    return parser.parse_args()


def load_exact_shap_values(
    dataset_name: str, model_name: str, value_function_name: str
) -> np.ndarray:
    """
    Load cached exact Shapley values for the test split.

    Args:
        dataset_name (str): Dataset name.
        model_name (str): Model name.
        value_function_name (str): Value function name (e.g. "marginal_bg100").

    Returns:
        np.ndarray: Exact Shapley values of shape (n_test, num_features).
    """
    path = (
        CACHE_DIR
        / dataset_name
        / f"{model_name}_exact_shap_{value_function_name}_test.npy"
    )
    try:
        return np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Exact Shapley values not found at {path}. Run "
            f"`uv run python calculate_exact_shap.py {dataset_name} {model_name} "
            f"<value_function> --background_size <N>` with a background size "
            f"matching '{value_function_name}' first."
        )


def main(experiment_name: str) -> dict:
    # Experiment names are "{dataset}_{model}_{value_function}_epoch{N}_repeats{R}"
    # where value_function is "{type}_bg{size}" or "baseline_{zero,mean}".
    args = experiment_name.split("_")
    dataset_name = args[0]
    model_name = args[1]
    value_function_name = args[2] + "_" + args[3]

    exact_shap = load_exact_shap_values(dataset_name, model_name, value_function_name)
    amortised_attr = load_amortised_attributions(experiment_name)

    if exact_shap.shape != amortised_attr.shape:
        raise ValueError(
            f"Shape mismatch between exact Shapley values {exact_shap.shape} and "
            f"amortised attributions {amortised_attr.shape}. Make sure the exact "
            f"values were computed with the same background_size as training."
        )

    num_features = exact_shap.shape[1]
    results = {
        "Reconstruction Quality (MSE)": reconstruction_quality(
            amortised_attr, exact_shap
        ),
        "Recall@1": top_k_recall(amortised_attr, exact_shap, 1),
        "Recall@3": top_k_recall(amortised_attr, exact_shap, min(3, num_features)),
        "Recall@5": top_k_recall(amortised_attr, exact_shap, min(5, num_features)),
        "Spearman Correlation": correlation(amortised_attr, exact_shap),
    }

    print(f"Comparison: exact Shapley vs amortiser for '{experiment_name}'")
    print(f"Value function: {value_function_name}  |  n_test={exact_shap.shape[0]}")
    for name, value in results.items():
        print(f"  {name}: {value:.6f}")

    out_path = RESULTS_DIR / experiment_name / "exact_shap_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    return results


if __name__ == "__main__":
    args = parse_args()
    main(args.experiment_name)
