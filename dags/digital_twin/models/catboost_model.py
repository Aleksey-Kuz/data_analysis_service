""" CatBoost regression and classifier """

from catboost import CatBoostRegressor, CatBoostClassifier
import pandas as pd

from typing import Any, Dict, List

from digital_twin.configs.models_conf import CatBoostModelConf
from digital_twin.models.base_model import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost Classification and Regression Model
    """

    def __init__(self, task_type: str, **kwargs):
        self.model = None
        self.task_type = task_type
        self.config = self._load_config()
        self.params = {**self.config, **kwargs}

    def create_new_model(self):
        """ Create new instance by task_type """
        if self.task_type == "regression":
            self.model = CatBoostRegressor(**self.params)
        elif self.task_type == "classifier":
            self.model = CatBoostClassifier(**self.params)
        else:
            raise ValueError("The wrong type of CatBoost model is specified.")

    def _load_config(self) -> Dict[str, Any]:
        """ Load config data form models_conf.CatBoostModelConf class """
        config_data = CatBoostModelConf.model_params
        if self.task_type == "regression":
            config_data["loss_function"] = CatBoostModelConf.reg_loss_function
            config_data["eval_metric"] = CatBoostModelConf.reg_eval_metric
        elif self.task_type == "classifier":
            config_data["loss_function"] = CatBoostModelConf.cla_loss_function
            config_data["eval_metric"] = CatBoostModelConf.cla_eval_metric
        else:
            raise ValueError("The wrong type of CatBoost model is specified.")
        return config_data

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame, cat_features: List[str] = None):
        """ Train CatBoost model """
        self.create_new_model()

        self.model.fit(
            x_train, y_train,
            cat_features=cat_features,
            verbose=100
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """ Predict results from trained CatBoost model """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        predictions = self.model.predict(df).flatten()
        return pd.Series(predictions, index=df.index)

    def save(self, path: str):
        """ Save trained CatBoost model """
        self.model.save_model(path)

    def load(self, path: str):
        """ Load trained CatBoost model """
        self.create_new_model()
        self.model.load_model(path)
