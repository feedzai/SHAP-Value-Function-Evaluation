from typing import Any, Dict

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from experiment.paths import CACHE_DIR
from src.models.model import Model


class LogisticRegressionModel(Model):
    def __init__(self, dataset_name: str):
        """
        Instantiate the LogisticRegressionModel class.

        Args:
            dataset_name (str): Name of the dataset.
        """
        super().__init__(dataset_name)
        self.model_name = "logisticregression"

    def __call__(self, x: np.ndarray, raw_score: bool = False) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            x (np.ndarray): Input features.
            raw_score (bool): If True, return raw scores. Otherwise, probabilities.

        Returns:
            np.ndarray: Predictions.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` method first.")
        else:
            if raw_score:
                prob = self.model.predict_proba(x)[:, 1]
                epsilon = 1e-15
                prob = np.clip(prob, epsilon, 1 - epsilon)
                y_hat = np.log(prob)
            else:
                y_hat = self.model.predict_proba(x)[:, 1]

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
        Train the Logistic Regression model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            model_config (Dict[str, Any]): Model training parameters.
            save (bool): If True, save the trained model.

        Returns:
            LogisticRegression: Trained Logistic Regression model.
        """
        if model_config["optim"]:
            best_params = self.optimize_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                model_config["objective"],
                model_config["n_trials"],
                model_config["cv_folds"],
                model_config["optimization_metric"],
            )
        else:
            best_params = model_config["logreg_params"]

        self.model = LogisticRegression(**best_params)
        self.model.fit(X_train.values, y_train.values)

        if save:
            self.save()

        return self.model

    def optimize_hyperparameters(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        optimization_objective: str,
        n_trials: int = 20,
        cv_folds: int = 3,
        optimization_metric: str = "auto",
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Optimize the hyperparameters of the Logistic Regression model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            base_params (Dict[str, Any]): Base parameters for the model.
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
            params = {}

            # Core hyperparameters to optimize
            params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
            params["tol"] = trial.suggest_float("tol", 1e-4, 1e-1, log=True)
            params["solver"] = trial.suggest_categorical(
                "solver",
                ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            )
            params["max_iter"] = trial.suggest_int("max_iter", 100, 1000)
            params["class_weight"] = trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            )
            params["fit_intercept"] = trial.suggest_categorical(
                "fit_intercept", [True, False]
            )
            params["intercept_scaling"] = trial.suggest_float(
                "intercept_scaling", 0.01, 10.0, log=True
            )
            params["random_state"] = random_state
            params["verbose"] = 0  # Suppress output

            # Cross-validation evaluation
            cv_scores = []

            for train_idx, val_idx in cv.split(X_cv, y_cv):
                X_fold_train, X_fold_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
                y_fold_train, y_fold_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]

                # Train model
                model = LogisticRegression(**params)
                model.fit(X_fold_train, y_fold_train)

                # Make predictions
                y_pred = np.array(model.predict_proba(X_fold_val)[:, 1])

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

        print("\n=== Optimization Results ===")
        print(f"Best {optimization_metric}: {study.best_value:.4f}")
        print("Best parameters:")
        for key, value in best_params.items():
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
            CACHE_DIR / self.dataset_name / "optimization_results.pkl",
        )

        return best_params

    def save(self) -> None:
        joblib.dump(
            self.model, CACHE_DIR / self.dataset_name / f"{self.model_name}.pkl"
        )
