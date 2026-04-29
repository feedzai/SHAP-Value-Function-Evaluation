from typing import Any, Dict

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from experiment.paths import CACHE_DIR
from src.models.model import Model


class LightGBMModel(Model):
    def __init__(self, dataset_name: str):
        """
        Initialize the LightGBM model.

        Args:
            dataset_name (str): Name of the dataset to train on.
        """
        super().__init__(dataset_name)
        self.model_name = "lightgbm"

    def __call__(self, x: np.ndarray, raw_score: bool = False) -> np.ndarray:
        """
        Make predictions using the trained LightGBM model.

        Args:
            x (np.ndarray): Input features.
            raw_score (bool): Whether to return raw scores or probabilities.

        Returns:
            np.ndarray: Predictions.

        """
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` method first.")
        else:
            y_hat = self.model.predict(x, raw_score=raw_score)
            return np.array(y_hat)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        model_config: Dict[str, Any],
        save: bool = True,
    ):
        """
        Train the LightGBM model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            model_config (Dict[str, Any]): Model training parameters.
            save (bool): Whether to save the model.

        Returns:
            lgb.LGBMModel: Trained LightGBM model.
        """
        if model_config["optim"]:
            best_params = self.optimize_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                model_config["lgbm_params"],
                model_config["objective"],
                model_config["n_trials"],
                model_config["cv_folds"],
                model_config["optimization_metric"],
            )
        else:
            best_params = model_config["lgbm_params"]

        self.model = self._train_lgb(X_train, y_train, X_val, y_val, best_params)

        if save:
            self.save()

        return self.model

    def _train_lgb(self, X_train, y_train, X_val, y_val, params):
        """
        Train LightGBM model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            params (Dict[str, Any]): Parameters for the LightGBM model.

        Returns:
            lgb.LGBMModel: Trained LightGBM model.
        """
        return lgb.train(
            params=params,
            train_set=lgb.Dataset(X_train, label=y_train),
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20)],
        )

    def optimize_hyperparameters(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        base_params: Dict[str, Any],
        optimization_objective: str,
        n_trials: int = 20,
        cv_folds: int = 3,
        optimization_metric: str = "auto",
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Optimize the hyperparameters of the LightGBM model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            base_params (Dict[str, Any]): Base parameters for the LightGBM model.
            n_trials (int): Number of trials for Optuna optimization.
            cv_folds (int): Number of cross-validation folds.
            optimization_metric (str): Optimization metric.
            random_state (int): Random state for reproducibility.

        Returns:
            Dict[str, Any]: Best parameters for the LightGBM model.
        """
        is_classification = optimization_objective in ["binary", "multiclass"]

        if optimization_metric == "auto":
            optimization_metric = "auc" if is_classification else "rmse"

        if is_classification:
            cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        X_cv = pd.concat([X_train, X_val], ignore_index=True)
        y_cv = pd.concat([y_train, y_val], ignore_index=True)

        def objective(trial):
            """Objective function for Optuna optimization."""
            params = base_params.copy()

            # Core hyperparameters to optimize
            params["num_leaves"] = trial.suggest_int("num_leaves", 10, 100)
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            )
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
            params["bagging_fraction"] = trial.suggest_float(
                "bagging_fraction", 0.5, 1.0
            )
            params["feature_fraction"] = trial.suggest_float(
                "feature_fraction", 0.5, 1.0
            )

            # Additional parameters
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            )

            # Early stopping
            params["early_stopping_rounds"] = trial.suggest_int(
                "early_stopping_rounds", 20, 200
            )

            # Random state for reproducibility
            params["random_state"] = random_state
            params["verbosity"] = -1  # Suppress output

            # Cross-validation evaluation
            cv_scores = []

            for train_idx, val_idx in cv.split(X_cv, y_cv):
                X_fold_train, X_fold_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
                y_fold_train, y_fold_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

                # Train model
                model = self._train_lgb(
                    X_fold_train, y_fold_train, X_fold_val, y_fold_val, params
                )

                # Make predictions
                y_pred = np.array(model.predict(X_fold_val))

                # Calculate metric
                if optimization_metric == "auc":
                    score = roc_auc_score(y_fold_val, y_pred)
                elif optimization_metric == "mse":
                    score = mean_squared_error(y_fold_val, y_pred)
                else:
                    raise ValueError(
                        f"Unknown optimization metric: {optimization_metric}"
                    )

                cv_scores.append(score)

            return np.mean(cv_scores)

        # Create study
        study = optuna.create_study(
            direction="maximize" if optimization_metric == "auc" else "minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10, n_warmup_steps=5, interval_steps=1
            ),
        )

        print(f"\n=== Starting Hyperparameter Optimization for {self.dataset_name} ===")
        print(f"Optimization metric: {optimization_metric}")
        print(f"Number of trials: {n_trials}")
        print(f"CV folds: {cv_folds}")

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params.copy()
        best_params.update(base_params)  # Ensure base params are included

        print("\n=== Optimization Results ===")
        print(f"Best {optimization_metric}: {study.best_value:.4f}")
        print("Best parameters:")
        for key, value in best_params.items():
            if key not in base_params:  # Only show optimized parameters
                print(f"  {key}: {value}")

        # Save optimization results
        optimization_results = {
            "best_params": best_params,
            "best_score": study.best_value,
            "optimization_metric": optimization_metric,
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "study": study,
        }

        joblib.dump(
            optimization_results,
            CACHE_DIR / self.dataset_name / "optimization_results_lightgbm.pkl",
        )

        return best_params

    def save(self) -> None:
        joblib.dump(
            self.model, CACHE_DIR / self.dataset_name / f"{self.model_name}.pkl"
        )
