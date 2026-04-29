import dice_ml
import numpy as np
import pandas as pd

from src.models.model import Model


class DiceModel:
    def __init__(self, model: Model):
        self.model = model

    def predict_proba(self, x) -> np.ndarray:
        return self.model(np.array(x), raw_score=False)


def initialize_counterfactual_generator(
    data: pd.DataFrame,
    target_column: str,
    numerical_features: list[str],
    classifier: Model,
):
    d = dice_ml.Data(
        dataframe=data,
        outcome_name=target_column,
        continuous_features=numerical_features,
    )
    m = dice_ml.Model(model=DiceModel(classifier), backend="sklearn")
    cf_generator = dice_ml.Dice(d, m, method="random")

    return cf_generator


def compute_counterfactuals(
    cf_generator: dice_ml.Dice,
    X: pd.DataFrame,
    background_size: int,
    desired_class: str | int,
):
    cf = cf_generator.generate_counterfactuals(
        query_instances=X,
        total_CFs=background_size,
        desired_class=desired_class,
    )

    examples = []
    for i in range(len(X)):
        cf_df = cf.cf_examples_list[i].final_cfs_df
        if cf_df is not None and len(cf_df) > 0:
            example = cf_df[X.columns]
        else:
            example = X.iloc[[i]]

        if len(example) < background_size:
            if len(example) > 0:
                example = example.sample(n=background_size, replace=True)
            else:
                example = pd.concat([X.iloc[[i]]] * background_size, ignore_index=True)

        examples.append(example)

    return np.array([example.values for example in examples])
