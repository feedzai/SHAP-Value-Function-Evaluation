import warnings

from experiment.shap_values import (
    compute_and_save_kernel_shap_values,
    compute_and_save_treeshap_values,
)
from data.configs import DATASET_CONFIGS
from data.utils import load_data, load_model
from experiment.arg_parsers import parse_reference_shap_args

warnings.filterwarnings("ignore")


def main(dataset_name: str, model_name: str, background_size: int):
    model = load_model(dataset_name, model_name)
    X_train, X_val, X_test, y_train, _, _ = load_data(dataset_name)
    categorical_features = DATASET_CONFIGS.get(dataset_name, {}).get(
        "categorical_features", []
    )
    numerical_features = [x for x in X_train.columns if x not in categorical_features]

    # === Calculate Reference Attributions ====================================

    if model_name == "lightgbm":
        compute_and_save_treeshap_values(dataset_name, model, X_val, "val")
        compute_and_save_treeshap_values(dataset_name, model, X_test, "test")

    compute_and_save_kernel_shap_values(
        dataset_name,
        model,
        X_train,
        y_train,
        numerical_features,
        X_val,
        "val",
        background_size,
    )

    compute_and_save_kernel_shap_values(
        dataset_name,
        model,
        X_train,
        y_train,
        numerical_features,
        X_test,
        "test",
        background_size,
    )


if __name__ == "__main__":
    args = parse_reference_shap_args()
    main(args.dataset_name, args.model_name, args.background_size)
