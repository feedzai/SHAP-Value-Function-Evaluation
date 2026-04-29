import numpy as np
import pandas as pd
import shap

from experiment.counterfactuals import (
    compute_counterfactuals,
    initialize_counterfactual_generator,
)
from data.utils import stratified_sampling
from experiment.paths import CACHE_DIR
from src.models.model import Model


def compute_and_save_treeshap_values(
    dataset_name: str, model: Model, X: pd.DataFrame, split: str
) -> np.ndarray:
    """
    Compute TreeSHAP values for a given dataset and model.

    Args:
        dataset_name (str): Name of the dataset
        model (lgb.Booster): LightGBM model
        X (pd.DataFrame): Features
        split (str): Train, val, or test split (for storing the values in cache)

    Returns:
        TreeSHAP values (np.ndarray)
    """
    tree_explainer = shap.TreeExplainer(
        model.model,
        data=None,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    attr = tree_explainer.shap_values(X)

    model_name = model.model_name
    np.save(CACHE_DIR / dataset_name / f"{model_name}_tree_shap_{split}.npy", attr)

    return attr


def compute_and_save_kernel_shap_values(
    dataset_name: str,
    model: Model,
    data: pd.DataFrame,
    labels: pd.Series,
    numerical_features: list[str],
    X: pd.DataFrame,
    split: str,
    kernel_shap_bg_size: int,
):
    """
    Compute KernelSHAP values for a given dataset and model.

    Args:
        dataset_name (str): Name of the dataset
        model (Model): Model instance
        data (pd.DataFrame): Data used to calculate kernelSHAP background sets
        labels (pd.Series): Labels for counterfactual generation
        numerical_features (list[str]): List of numerical feature names
        X (pd.DataFrame): Instances to calculate kernelSHAP values for
        split (str): Train, val, or test split (for storing the values in cache)
        kernel_shap_bg_size (int): Size of the background set for KernelSHAP
    """
    model_name = str(model.model_name)
    y_pred = model(data.values, raw_score=False)

    # Define background generators as a list of (name, generator_function) tuples
    background_generators = [
        ("baseline_zero", lambda: _generate_baseline_zero_background(data)),
        ("baseline_mean", lambda: _generate_baseline_mean_background(data)),
        (
            f"marginal_bg{kernel_shap_bg_size}",
            lambda: _generate_marginal_background(data, y_pred, kernel_shap_bg_size),
        ),
        (
            f"uniform_bg{kernel_shap_bg_size}",
            lambda: _generate_uniform_background(data, kernel_shap_bg_size),
        ),
        (
            f"jointmarginal_bg{kernel_shap_bg_size}",
            lambda: _generate_joint_marginal_background(
                data, y_pred, kernel_shap_bg_size
            ),
        ),
        (
            f"conditional_bg{kernel_shap_bg_size}",
            lambda: _generate_conditional_background(
                data.values, X.values, kernel_shap_bg_size
            ),
        ),
        (
            f"filteredconditional_bg{kernel_shap_bg_size}",
            lambda: _generate_filtered_conditional_background(
                data, y_pred, kernel_shap_bg_size
            ),
        ),
        (
            f"counterfactual_bg{kernel_shap_bg_size}",
            lambda: _generate_counterfactual_background(
                model, data, labels, numerical_features, X, kernel_shap_bg_size
            ),
        ),
    ]

    # Process each background type
    for background_name, generator_func in background_generators:
        print(f"Computing {background_name} background")
        background = generator_func()
        _compute_and_save_shap(
            model=model,
            background=background,
            X=X,
            dataset_name=dataset_name,
            model_name=model_name,
            split=split,
            background_name=background_name,
        )


def _compute_and_save_shap(
    model: Model,
    background: np.ndarray | list[np.ndarray],
    X: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    split: str,
    background_name: str,
):
    """Helper function to compute SHAP values and save them."""
    if isinstance(background, list):
        # Handle per-instance backgrounds (e.g., counterfactual)
        attr = []
        avg_constant = []
        for i in range(len(X)):
            x = X.iloc[[i]]
            bg = background[i]
            instance_attr, instance_avg_constant = calculate_kernel_shap(model, bg, x)
            attr.append(instance_attr[0])
            avg_constant.append(instance_avg_constant)
        attr = np.array(attr)
        avg_constant = np.array(avg_constant)
    else:
        # Handle single background set
        attr, avg_constant = calculate_kernel_shap(model, background, X)
        avg_constant = np.array([avg_constant])

    save_kernel_shap_values(
        dataset_name, model_name, attr, avg_constant, split, background_name
    )


def calculate_kernel_shap(
    model: Model, background: np.ndarray, X: pd.DataFrame
) -> tuple[np.ndarray, float]:
    kernel_explainer = shap.KernelExplainer(
        lambda x: model(x, raw_score=True),
        data=background,
    )

    return kernel_explainer.shap_values(X), kernel_explainer.expected_value


def save_kernel_shap_values(
    dataset_name: str,
    model_name: str,
    attr: np.ndarray,
    avg_constant: np.ndarray,
    split: str,
    background_name: str,
):
    np.save(
        CACHE_DIR
        / dataset_name
        / f"{model_name}_kernel_shap_{background_name}_{split}.npy",
        attr,
    )

    np.save(
        CACHE_DIR
        / dataset_name
        / f"{model_name}_kernel_shap_avg_constant_{background_name}_{split}.npy",
        avg_constant,
    )


def _generate_baseline_zero_background(data: pd.DataFrame) -> np.ndarray:
    """Generate baseline background with zeros."""
    return np.zeros(data.shape[1]).reshape(1, -1)


def _generate_baseline_mean_background(data: pd.DataFrame) -> np.ndarray:
    """Generate baseline background with mean values."""
    return np.array([np.mean(data, axis=0)]).reshape(1, -1)


def _generate_marginal_background(
    data: pd.DataFrame, y_pred: np.ndarray, bg_size: int
) -> np.ndarray:
    """Generate marginal background using stratified sampling."""
    background, _ = stratified_sampling(data.values, y_pred, bg_size)
    return background


def _generate_uniform_background(data: pd.DataFrame, bg_size: int) -> np.ndarray:
    """Generate uniform background from feature ranges."""
    return np.random.uniform(
        data.min(axis=0), data.max(axis=0), size=(bg_size, data.shape[1])
    )


def _generate_joint_marginal_background(
    data: pd.DataFrame, y_pred: np.ndarray, bg_size: int
) -> np.ndarray:
    """Generate joint marginal background."""
    joint_marginal_background = np.zeros((bg_size, data.shape[1]))
    for i in range(len(data.columns)):
        bg, _ = stratified_sampling(data.values, y_pred, bg_size)
        joint_marginal_background[:, i] = bg[:, i]
    return joint_marginal_background


def _generate_filtered_conditional_background(
    data: pd.DataFrame, y_pred: np.ndarray, bg_size: int
) -> np.ndarray:
    """Generate filtered conditional background from lowest predictions."""
    idx = np.argsort(y_pred)[:bg_size]
    return data.values[idx]


def _generate_counterfactual_background(
    model: Model,
    data: pd.DataFrame,
    labels: pd.Series,
    numerical_features: list[str],
    X: pd.DataFrame,
    bg_size: int,
) -> list[np.ndarray]:
    """
    Generate counterfactual background for each instance in X.
    Returns a list of background arrays, one per instance.
    """
    y_pred = model(data.values, raw_score=False)

    # Create stratified sample with labels
    background, _ = stratified_sampling(
        pd.concat([data, labels], axis=1).values, y_pred, bg_size
    )
    sample_df = pd.DataFrame(
        background,
        columns=(data.columns.to_list() + [labels.name]),
    )
    background_df = sample_df.drop(columns=[labels.name])

    cf_generator = initialize_counterfactual_generator(
        sample_df, str(labels.name), numerical_features, model
    )
    precomputed_cf = compute_counterfactuals(
        cf_generator, background_df, bg_size, desired_class=0
    )

    # Find closest background instance for each X instance
    distances = np.sqrt(
        ((X.values[:, None, :] - background_df.values[None, :, :]) ** 2).sum(axis=2)
    )
    closest_indices = np.argmin(distances, axis=1)

    # Return list of counterfactual backgrounds, one per X instance
    return [precomputed_cf[idx] for idx in closest_indices]


def _generate_conditional_background(
    data: np.ndarray,
    X: np.ndarray,
    bg_size: int,
) -> list[np.ndarray]:
    """Generate conditional background.
    Returns a list of background arrays, one per instance.
    """
    distances = np.sqrt(((X[:, None, :] - data[None, :, :]) ** 2).sum(axis=2))
    # get top b_size closest instances to X in data
    closest_indices = np.argsort(distances, axis=1)[:, :bg_size]
    return [data[idx] for idx in closest_indices]
