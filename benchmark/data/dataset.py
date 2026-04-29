from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from data.utils import load_data, load_model, load_ref_shap_values
from experiment.paths import CACHE_DIR


class Dataset:
    """
    Dataset class for loading data, model, and reference SHAP values for experiments.

    Args:
        dataset_name (str): Name of the dataset
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training labels
        y_val (pd.Series): Validation labels
        y_test (pd.Series): Test labels
        numerical_features (list[str]): Numerical features
        categorical_features (list[str]): Categorical features
        cardinalities (list[int]): Cardinalities of categorical features
        model_name (str): Name of the model
        model (lgb.Booster): LightGBM model trained on the dataset
    """

    def __init__(self, dataset_name: str, model_name: str):
        """
        Initialize the Dataset class.

        Args:
            dataset_name (str): Name of the dataset
            model_name (str): Name of the model
        """
        self.dataset_name = dataset_name

        # Load data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = (
            load_data(dataset_name)
        )

        # Load feature information
        with open(CACHE_DIR / dataset_name / "features.yaml", "r") as f:
            feature_config = yaml.load(f, Loader=yaml.SafeLoader)

        self.numerical_features = feature_config["numerical_features"]
        self.categorical_features = feature_config["categorical_features"]
        cardinalities_dict = feature_config["cardinalities"]
        if cardinalities_dict is not None:
            self.cardinalities = [
                cardinalities_dict[col] for col in self.categorical_features
            ]
        else:
            self.cardinalities = None

        # Load model
        self.model_name = model_name
        self.model = load_model(dataset_name, model_name)

    def get_train_dataloader(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> DataLoader:
        """
        Get a DataLoader for the training set.

        Args:
            batch_size (int): Batch size
            generator (torch.Generator, optional): Random number generator

        Returns:
            train_loader (DataLoader[TensorDataset]): DataLoader for the training set
        """
        X_train_tensor = torch.tensor(self.X_train.values, dtype=torch.float32)
        X_num_tensor = torch.tensor(
            self.X_train[self.numerical_features].values, dtype=torch.float32
        )
        X_cat_tensor = torch.tensor(
            self.X_train[self.categorical_features].values, dtype=torch.long
        )
        y_hat_train_tensor = torch.tensor(
            self.model(self.X_train.values, raw_score=True), dtype=torch.float32
        )

        train_loader = DataLoader(
            TensorDataset(
                X_train_tensor, X_num_tensor, X_cat_tensor, y_hat_train_tensor
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
        )

        return train_loader

    def get_val_dataloader(
        self, batch_size: int, value_function_name: str
    ) -> DataLoader:
        """
        Get a DataLoader for the validation set.

        Args:
            batch_size (int): Batch size
            value_function_name (str): Name of the value function

        Returns:
            val_loader (DataLoader[TensorDataset]): DataLoader for the validation set
        """
        X_num_tensor = torch.tensor(
            self.X_val[self.numerical_features].values, dtype=torch.float32
        )
        X_cat_tensor = torch.tensor(
            self.X_val[self.categorical_features].values, dtype=torch.long
        )

        ref_shap = load_ref_shap_values(
            self.dataset_name, self.model_name, value_function_name, "val"
        )
        ref_shap_val_tensor = torch.tensor(ref_shap, dtype=torch.float32)

        val_loader = DataLoader(
            TensorDataset(X_num_tensor, X_cat_tensor, ref_shap_val_tensor),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )

        return val_loader
