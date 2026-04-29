import warnings
from typing import Literal

import numpy as np
import torch

from data.dataset import Dataset
from experiment.paths import RESULTS_DIR
from experiment.utils import generate_experiment_name, save_experiment_config
from experiment.arg_parsers import parse_experiment_args
from src.amortiser import Amortiser
from src.scaler import Scaler
from src.value_functions.utils import get_value_function

warnings.filterwarnings("ignore")


def main(
    dataset_name: str,
    model_name: str,
    batch_size: int,
    value_function_name: str,
    background_size: int,
    baseline_type: Literal["zero", "mean"],
    reg: float,
    num_epochs: int,
    lr_adam: float,
    lr_sgd: float,
    adam_epochs: int,
    warmup_epochs: int,
    n_repeats: int,
    save: bool,
):
    print(
        "Training Amortiser for dataset:",
        dataset_name,
        "with model:",
        model_name,
        "and value function:",
        value_function_name,
    )

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === General Params ======================================================

    dataset = Dataset(dataset_name, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Initialize Amortiser ================================================

    X_train = dataset.X_train
    X_train_num = torch.tensor(
        X_train[dataset.numerical_features].values, dtype=torch.float32, device=device
    )

    value_function = get_value_function(
        value_function_name,
        dataset,
        background_size,
        baseline_type,
        device,
    )

    scaler = Scaler(X_train_num)

    amortiser = Amortiser(
        device,
        len(dataset.numerical_features),
        dataset.cardinalities,
        value_function,
        scaler,
    )

    # === Load data ===========================================================

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = dataset.get_train_dataloader(batch_size, generator=generator)
    val_loader = dataset.get_val_dataloader(batch_size, value_function.name)

    # === Train Amortiser ======================================================

    amortiser.train(
        train_loader,
        val_loader,
        num_epochs,
        lr_adam,
        lr_sgd,
        adam_epochs,
        warmup_epochs,
        n_repeats,
        reg,
    )

    if save:
        experiment_name = generate_experiment_name(
            dataset_name, model_name, value_function.name, num_epochs, n_repeats
        )

        amortiser.save_for_inference(experiment_name)

        X_test = dataset.X_test
        X_test_num = torch.tensor(
            X_test[dataset.numerical_features].values,
            dtype=torch.float32,
            device=device,
        )
        X_test_cat = torch.tensor(
            X_test[dataset.categorical_features].values, dtype=torch.long, device=device
        )

        with torch.no_grad():
            test_attr: torch.Tensor = amortiser(X_test_num, X_test_cat)

        np.save(
            RESULTS_DIR / experiment_name / "amortised_attr_test.npy",
            test_attr.detach().cpu().numpy(),
        )

        save_experiment_config(
            experiment_name,
            dataset_name,
            value_function.name,
            num_epochs,
            len(dataset.numerical_features),
            dataset.cardinalities,
            batch_size,
            reg,
            lr_adam,
            lr_sgd,
            adam_epochs,
            warmup_epochs,
            n_repeats,
        )


if __name__ == "__main__":
    args = parse_experiment_args()

    main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        value_function_name=args.value_function,
        background_size=args.background_size,
        baseline_type=args.baseline_type,
        reg=args.reg,
        num_epochs=args.num_epochs,
        lr_adam=args.lr_adam,
        lr_sgd=args.lr_sgd,
        adam_epochs=args.adam_epochs,
        warmup_epochs=args.warmup_epochs,
        n_repeats=args.n_repeats,
        save=args.save,
    )
