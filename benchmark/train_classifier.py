from experiment.models import evaluate_model, train_and_save_model
from data.utils import load_data
from data.configs import DATASET_CONFIGS
from experiment.arg_parsers import parse_train_classifier_args


def main(dataset_name: str, model_name: str):
    # === Load data ===========================================================

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset_name)

    # === Model training ======================================================

    model_config = DATASET_CONFIGS.get(dataset_name, {}).get("model_config", {})

    model = train_and_save_model(
        dataset_name,
        model_name,
        model_config,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # === Print Model Performance =============================================

    objective = model_config.get("objective", "")
    evaluate_model(
        model,
        objective,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    args = parse_train_classifier_args()
    main(args.dataset_name, args.model_name)
