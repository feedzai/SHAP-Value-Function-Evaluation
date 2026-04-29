import numpy as np
import pandas as pd

from evaluation.amortiser_alignment_metrics import (
    reconstruction_quality,
    top_k_recall,
)
from evaluation.quantitative_metrics import (
    attribution_sensitivity,
    compute_counterfactual_attributions,
    contrastivity,
    deletion_auc,
    insertion_auc,
    reconstruction_mse,
    sparsity,
)
from src.amortiser import InferenceAmortiser
from src.models.model import Model
from src.value_functions.base_value_function import BaseValueFunction


def compute_metrics_for_sample(
    X: pd.DataFrame,
    preds: np.ndarray,
    preds_cf: np.ndarray,
    ref_shap: np.ndarray,
    attr: np.ndarray,
    attr_cf: np.ndarray,
    classifier: Model,
    amortiser: InferenceAmortiser,
    numerical_features: list[str],
    categorical_features: list[str],
    value_function: BaseValueFunction,
    category_probs: dict[str, dict],
) -> dict:
    """
    Compute all metrics for a single bootstrap sample.

    Args:
        X (pd.DataFrame): Features
        preds (np.ndarray): Predictions
        preds_cf (np.ndarray): Counterfactual predictions
        ref_shap (np.ndarray): Reference Shapley explanations
        attr (np.ndarray): Amortised Shapley explanations
        attr_cf (np.ndarray): Counterfactual amortised Shapley explanations
        classifier (Model): Classifier to explain
        amortiser (InferenceAmortiser): Amortiser for producing explanations
        numerical_features (list[str]): Numerical features
        categorical_features (np.ndarray): Categorical features
        value_function (BaseValueFunction): Value function used for training amortiser
        category_probs (dict[str, dict]): Probabilities for categorical features

    Returns:
        dict with keys "quantitative_metrics" and "agreement_metrics"
    """
    quantitative_metrics = {
        "Reconstruction MSE": reconstruction_mse(
            classifier,
            amortiser,
            X.values,
            attr,
        ),
        "Sparsity": sparsity(attr),
        "Attribution Sensitivity": attribution_sensitivity(
            X,
            classifier,
            amortiser,
            numerical_features,
            categorical_features,
            category_probs,
            n_perturbations=100,
            noise_magnitude_numerical=0.1,
            noise_magnitude_categorical=0.5,
        ),
        "Contrastivity": contrastivity(
            attr,
            preds,
            attr_cf,
            preds_cf,
        ),
        "Deletion AUC": deletion_auc(
            X,
            attr,
            value_function,
            top_k=None,
        ),
        "Insertion AUC": insertion_auc(
            X,
            attr,
            value_function,
            top_k=None,
        ),
    }

    agreement_metrics = {
        "Reconstruction Quality": reconstruction_quality(attr, ref_shap),
        "Recall@1": top_k_recall(attr, ref_shap, 1),
        "Recall@3": top_k_recall(attr, ref_shap, 3),
        "Recall@5": top_k_recall(attr, ref_shap, 5),
    }

    return {
        "quantitative_metrics": quantitative_metrics,
        "agreement_metrics": agreement_metrics,
    }


def bootstrap_metrics(
    n_bootstrap: int,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ref_shap: np.ndarray,
    amortised_attr: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    classifier: Model,
    amortiser: InferenceAmortiser,
    numerical_features: list[str],
    categorical_features: list[str],
    value_function: BaseValueFunction,
    seed: int = 0,
) -> dict:
    """
    Run evaluation on n_bootstrap bootstrap samples and aggregate results.

    Args:
        n_bootstrap (int): Number of bootstrap samples
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        ref_shap (np.ndarray): Reference Shapley explanations
        amortised_attr (np.ndarray): Amortised Shapley explanations
        X_train (pd.DataFrame): Train features
        y_train (pd.Series): Train labels
        classifier (Model): Classifier to explain
        amortiser (InferenceAmortiser): Amortiser for producing explanations
        numerical_features (list[str]): Numerical features
        categorical_features (np.ndarray): Categorical features
        value_function (BaseValueFunction): Value function used for training amortiser
        seed (int): Random seed

    Returns:
        Dictionary with aggregated metrics

    """
    np.random.seed(seed)
    n_test = int(len(X_test) / 2)

    all_quantitative_metrics = {}
    all_agreement_metrics = {}

    X_train_augmented = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_train_augmented = pd.concat([y_train, y_test]).reset_index(drop=True)

    attr_cf, pred_cf = compute_counterfactual_attributions(
        X_train_augmented,
        y_train_augmented,
        categorical_features,
        numerical_features,
        X_test,
        classifier,
        amortiser,
    )

    preds = classifier(X_test.values, raw_score=True)

    category_probs = {
        col: X_train[col].value_counts(normalize=True).to_dict()
        for col in categorical_features
    }

    print(f"Running {n_bootstrap} bootstrap iterations...")

    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_bootstrap} iterations...")

        bootstrap_indices = np.random.choice(len(X_test), size=n_test, replace=False)

        X_test_sample = X_test.iloc[bootstrap_indices].reset_index(drop=True)
        pred_sample = preds[bootstrap_indices]
        pred_cf_sample = pred_cf[bootstrap_indices]
        ref_shap_sample = ref_shap[bootstrap_indices]
        amortised_attr_sample = amortised_attr[bootstrap_indices]
        attr_cf_sample = attr_cf[bootstrap_indices]

        metrics = compute_metrics_for_sample(
            X_test_sample,
            pred_sample,
            pred_cf_sample,
            ref_shap_sample,
            amortised_attr_sample,
            attr_cf_sample,
            classifier,
            amortiser,
            numerical_features,
            categorical_features,
            value_function,
            category_probs,
        )

        for metric_name, metric_value in metrics["quantitative_metrics"].items():
            if metric_name not in all_quantitative_metrics:
                all_quantitative_metrics[metric_name] = []
            all_quantitative_metrics[metric_name].append(metric_value)
        for metric_name, metric_value in metrics["agreement_metrics"].items():
            if metric_name not in all_agreement_metrics:
                all_agreement_metrics[metric_name] = []
            all_agreement_metrics[metric_name].append(metric_value)

    return {
        "quantitative_metrics": all_quantitative_metrics,
        "agreement_metrics": all_agreement_metrics,
    }


def calculate_ranking(values: np.ndarray, ascending: bool):
    if not ascending:
        values = -values
    return np.argsort(np.argsort(values, axis=1), axis=1) + 1
