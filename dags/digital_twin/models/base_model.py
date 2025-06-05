""" The basic abstract class for ML models """

import pandas as pd

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
