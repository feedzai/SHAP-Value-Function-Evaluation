DATASET_CONFIGS = {
    "MaternalRisk": {
        "target_column": "RiskLevel",
        "target_transform": lambda x: x.apply(
            lambda val: 1 if val == "high risk" else 0
        ),  # Binary encoding
        "kernel_shap_bg_size": 100,
        "model_config": {
            "optim": True,
            "n_trials": 20,
            "cv_folds": 3,
            "optimization_metric": "auc",
            "objective": "binary",
            "lgbm_params": {},
            "logreg_params": {},
        },
    },
    "GermanCredit": {
        "target_column": "class",
        "target_transform": lambda x: x.apply(
            lambda val: 1 if val == 2 else 0
        ),  # Binary encoding
        "categorical_features": [
            "Attribute1",
            "Attribute3",
            "Attribute4",
            "Attribute6",
            "Attribute7",
            "Attribute9",
            "Attribute10",
            "Attribute12",
            "Attribute14",
            "Attribute15",
            "Attribute17",
        ],
        "model_config": {
            "optim": True,
            "n_trials": 20,
            "cv_folds": 3,
            "optimization_metric": "auc",
            "objective": "binary",
            "lgbm_params": {},
            "logreg_params": {},
        },
    },
    "HELOC": {
        "target_column": "RiskPerformance",
        "target_transform": lambda x: x.apply(lambda val: 1 if val == "Bad" else 0),
        "model_config": {
            "optim": True,
            "n_trials": 20,
            "cv_folds": 5,
            "optimization_metric": "auc",
            "objective": "binary",
            "lgbm_params": {},
            "logreg_params": {},
        },
    },
    "Adult": {
        "target_column": "PINCP",
        "target_transform": lambda x: x.apply(lambda val: 1 - val),
        "categorical_features": ["COW", "MAR", "OCCP", "POBP", "RELP", "RAC1P"],
        "model_config": {
            "optim": True,
            "n_trials": 10,
            "cv_folds": 3,
            "optimization_metric": "auc",
            "objective": "binary",
            "lgbm_params": {},
            "logreg_params": {},
        },
    },
}
