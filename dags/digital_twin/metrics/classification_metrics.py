""" Classification Metrics """

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score
)

from typing import Dict, Optional, Union

class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true: Union[np.ndarray, pd.Series],
                 y_pred: Union[np.ndarray, pd.Series]) -> float:
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true: Union[np.ndarray, pd.Series],
                  y_pred: Union[np.ndarray, pd.Series], average: str = "weighted") -> float:
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def recall(y_true: Union[np.ndarray, pd.Series],
               y_pred: Union[np.ndarray, pd.Series], average: str = "weighted") -> float:
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def f1(y_true: Union[np.ndarray, pd.Series],
           y_pred: Union[np.ndarray, pd.Series], average: str = "weighted") -> float:
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def log_loss_score(y_true: Union[np.ndarray, pd.Series], y_proba: np.ndarray) -> float:
        return log_loss(y_true, y_proba)

    @staticmethod
    def confusion(y_true: Union[np.ndarray, pd.Series],
                  y_pred: Union[np.ndarray, pd.Series], labels: Optional[list] = None) -> pd.DataFrame:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return pd.DataFrame(cm, index=labels or sorted(set(y_true)), columns=labels or sorted(set(y_true)))

    @staticmethod
    def roc_auc(y_true: Union[np.ndarray, pd.Series], y_proba: np.ndarray, multi_class: str = "ovr") -> float:
        return roc_auc_score(y_true, y_proba, multi_class=multi_class)

    @staticmethod
    def evaluate_all(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series],
            y_proba: Optional[Union[np.ndarray, pd.Series]] = None, average: str = "weighted") -> Dict[str, float]:
        """ Evaluate all classification metrics from already predicted data """
        results: Dict[str, float] = {
            "accuracy": ClassificationMetrics.accuracy(y_true, y_pred),
            "precision": ClassificationMetrics.precision(y_true, y_pred, average),
            "recall": ClassificationMetrics.recall(y_true, y_pred, average),
            "f1": ClassificationMetrics.f1(y_true, y_pred, average),
        }

        if y_proba is not None:
            results["log_loss"] = ClassificationMetrics.log_loss_score(y_true, y_proba)
            results["roc_auc"] = ClassificationMetrics.roc_auc(y_true, y_proba)

        return results
