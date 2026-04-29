DEFAULTS = {
    "background_size": 50,
    "baseline_type": "zero",
    "num_epochs": 100,
    "lr_adam": 1e-3,
    "lr_sgd": 1e-3,
    "adam_epochs": 30,
    "warmup_epochs": 10,
    "n_repeats": 4,
    "reg": 1e-1,
    "batch_size": 64,
}


DATASETS = [
    "GermanCredit",
    "MaternalRisk",
    "HELOC",
    "Adult",
]

MODELS = ["lightgbm", "logisticregression"]

VALUE_FUNCTIONS = [
    "baseline",
    "marginal",
    "uniform",
    "jointmarginal",
    "conditional",
    "counterfactual",
    "filteredconditional",
]
