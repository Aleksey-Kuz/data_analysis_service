""" Linear Regression Model """

from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from typing import Any, Dict, List, Optional

from digital_twin.configs.models_conf import LinearRegressionConf
from digital_twin.models.base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """
    Linear Regression Model
    """

    def __init__(self, **kwargs):
        self.model = None
        self.config = self._load_config()
        self.params = {**self.config, **kwargs}
        self.cat_features = list()

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        """ Load configuration from models_conf.LinearRegressionConf class """
        config_data = LinearRegressionConf.model_params
        return config_data

    @staticmethod
    def _preprocess(df: pd.DataFrame, cat_features: List[str] = None) -> np.ndarray:
        """ Pre-processing categorical features """
        preprocessor = ColumnTransformer(
            [("cat", OneHotEncoder(), cat_features)],
            remainder="passthrough"
        )
        return preprocessor.fit_transform(df)

    def create_new_model(self):
        """ Create new LinearRegression instance """
        self.model = LinearRegression(**self.params)

    def train(self, x_train: pd.DataFrame, y_train: pd.Series,
              cat_features: List[str] = None, sample_weight: Optional[List[float]] = None):
        """ Train Linear Regression model """
        self.create_new_model()

        if cat_features:
            x_train = self._preprocess(x_train, cat_features)
            self.cat_features = cat_features
        self.model.fit(x_train, y_train, sample_weight=sample_weight)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """ Predict target values for given features """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        original_index = df.index
        if self.cat_features:
            df = self._preprocess(df, self.cat_features)
        predictions = self.model.predict(df).flatten()
        return pd.Series(predictions, index=original_index)

    def save(self, path: str):
        """ Save model using joblib """
        dump(self.model, path)

    def load(self, path: str):
        """ Load model using joblib """
        self.model = load(path)
