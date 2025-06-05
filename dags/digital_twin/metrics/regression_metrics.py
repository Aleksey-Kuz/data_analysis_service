""" Regression Metrics """

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from typing import Dict, Union


class RegressionMetrics:
    @staticmethod
    def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def r2(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        return r2_score(y_true, y_pred)

    @staticmethod
    def mape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100

    @staticmethod
    def evaluate_all(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """ Evaluate all regression metrics from already predicted data """
        return {
            "mae": RegressionMetrics.mae(y_true, y_pred),
            "mse": RegressionMetrics.mse(y_true, y_pred),
            "rmse": RegressionMetrics.rmse(y_true, y_pred),
            "r2": RegressionMetrics.r2(y_true, y_pred),
            "mape": RegressionMetrics.mape(y_true, y_pred),
        }
