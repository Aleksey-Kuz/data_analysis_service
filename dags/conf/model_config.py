"""Config for model dags"""

from typing import Dict, Type, Union

from digital_twin.models.base_model import BaseModel
from digital_twin.models.catboost_model import CatBoostModel
from digital_twin.models.linear_regression_model import LinearRegressionModel
from digital_twin.models.logistic_regression_model import LogisticRegressionModel


class Config:
    VAR_MODEL_ROOT_DIR = "MODEL_ROOT_DIR"
    VAR_MODEL_DIR_NAME = "MODEL_DIR_NAME"
    VAR_MODEL_DEFAULT_CLASSIFICATION = "MODEL_DEFAULT_CLASSIFICATION"
    VAR_MODEL_DEFAULT_REGRESSION = "MODEL_DEFAULT_REGRESSION"
    VAR_TARGET_FEATURES = "TARGET_FEATURES"
    VAR_SPLIT_TEST_SIZE = "SPLIT_TEST_SIZE"
    VAR_COMPARE_METRIC_REGRESSION = "COMPARE_METRIC_REGRESSION"
    VAR_COMPARE_METRIC_CLASSIFICATION = "COMPARE_METRIC_CLASSIFICATION"
    VAR_CURRENT_MODEL_REGRESSION_FILE_NAME = "CURRENT_MODEL_REGRESSION_FILE_NAME"
    VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME = "CURRENT_MODEL_CLASSIFICATION_FILE_NAME"

    MODEL_REGISTRY: Dict[str, Dict[str, Type[BaseModel]]] = {
        "regression": {
            "CatBoostModel": CatBoostModel,
            "LinearRegressionModel": LinearRegressionModel,
        },
        "classification": {
            "CatBoostModel": CatBoostModel,
            "LogisticRegressionModel": LogisticRegressionModel,
        }
    }
    BASE_MODEL_TYPE = Union[CatBoostModel, LinearRegressionModel, LogisticRegressionModel]

    RANDOM_STATE = 42
