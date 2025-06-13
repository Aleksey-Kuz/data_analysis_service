"""Dag for getting model metrics"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from conf.data_config import Config as ConfigData
from conf.model_config import Config as ConfigModel
from tasks.data_tasks import (
    uploading_data
)
from tasks.model_tasks import (
    load_model,
    deployed_model,
    get_model_metrics
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
    dag_id="model_metrics",
    start_date=datetime(2024, 1, 1),
    params={"file_name": Param(Variable.get(VAR_DATASET_FILE_NAME))},
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def model_metrics():
    """DAG flow"""
    input_data = uploading_data(VAR_DATASET_ROOT_DIR, VAR_DATASET_DIR_NAME)
    loaded_models = load_model(VAR_MODEL_ROOT_DIR, VAR_MODEL_DIR_NAME, VAR_CURRENT_MODEL_REGRESSION_FILE_NAME, VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME)
    deployed_models = deployed_model(loaded_models)
    get_model_metrics()


model_metrics_dag = model_metrics()
