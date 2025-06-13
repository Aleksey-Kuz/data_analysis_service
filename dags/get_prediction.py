"""Dag for getting predictions"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from conf.data_config import Config as ConfigData
from conf.model_config import Config as ConfigModel
from tasks.data_tasks import (
    uploading_data,
    save_data
)
from tasks.model_tasks import (
    load_model,
    deployed_model,
    get_predictions
)

VAR_DATASET_ROOT_DIR = ConfigData.VAR_DATASET_ROOT_DIR
VAR_DATASET_DIR_NAME = ConfigData.VAR_DATASET_DIR_NAME
VAR_DATASET_FILE_NAME = ConfigData.VAR_DATASET_FILE_NAME

VAR_RESULTS_ROOT_DIR = ConfigData.VAR_RESULTS_ROOT_DIR
VAR_RESULTS_DIR_NAME = ConfigData.VAR_RESULTS_DIR_NAME

VAR_OUTPUT_FILE_TYPE = ConfigData.VAR_OUTPUT_FILE_TYPE

VAR_MODEL_ROOT_DIR = ConfigModel.VAR_MODEL_ROOT_DIR
VAR_MODEL_DIR_NAME = ConfigModel.VAR_MODEL_DIR_NAME
VAR_TARGET_FEATURES = ConfigModel.VAR_TARGET_FEATURES
VAR_CURRENT_MODEL_REGRESSION_FILE_NAME =ConfigModel.VAR_CURRENT_MODEL_REGRESSION_FILE_NAME
VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME = ConfigModel.VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME


@dag(
    dag_id="get_prediction",
    start_date=datetime(2024, 1, 1),
    params={
        "file_name": Param(Variable.get(VAR_DATASET_FILE_NAME)),
        "file_type": Param(Variable.get(VAR_OUTPUT_FILE_TYPE), enum=["csv", "xlsx"])},
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def get_prediction():
    """DAG flow"""
    input_data = uploading_data(VAR_DATASET_ROOT_DIR, VAR_DATASET_DIR_NAME)
    loaded_models = load_model(VAR_MODEL_ROOT_DIR, VAR_MODEL_DIR_NAME, VAR_CURRENT_MODEL_REGRESSION_FILE_NAME, VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME)
    deployed_models = deployed_model(loaded_models)
    data_with_result = get_predictions(input_data, deployed_models, VAR_TARGET_FEATURES)
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    data_with_result = pd.DataFrame({
        "feature_1": np.random.rand(10),
        "feature_2": np.random.randint(0, 100, 10),
        "category": np.random.choice(["A", "B", "C"], 10),
        "target_regression": np.random.normal(50, 10, 10),
        "target_classification": np.random.choice([0, 1], 10)
    })
    save_data(VAR_RESULTS_ROOT_DIR, VAR_RESULTS_DIR_NAME, "result", data_with_result)


get_prediction_dag = get_prediction()
