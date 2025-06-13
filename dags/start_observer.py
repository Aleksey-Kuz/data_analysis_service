"""Dag for start observer"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from conf.data_config import Config as ConfigData
from conf.model_config import Config as ConfigModel
from tasks.data_tasks import (
    uploading_data
)
from tasks.model_tasks import (
    load_model,
    deployed_model,
    get_predictions,
)
from tasks.observer_tasks import (
    check_bias_data_distribution,
    check_model_quality
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
    dag_id="start_observer",
    start_date=datetime(2024, 1, 1),
    params={"file_name": Param(Variable.get(VAR_DATASET_FILE_NAME))},
    schedule_interval="@daily",
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def start_observer():
    """DAG flow"""
    dag_for_data_evaluation = TriggerDagRunOperator(
        task_id="dag_for_data_evaluation",
        trigger_dag_id="data_evaluation",
        wait_for_completion=True,
        deferrable=True,
    )
    dag_for_preparation = TriggerDagRunOperator(
        task_id="dag_for_preparation",
        trigger_dag_id="data_preparation",
        wait_for_completion=True,
        deferrable=True,
    )
    dag_for_model_train = TriggerDagRunOperator(
        task_id="dag_for_model_train",
        trigger_dag_id="model_train",
        wait_for_completion=True,
        deferrable=True,
    )
    dag_for_model_metrics = TriggerDagRunOperator(
        task_id="dag_for_model_metrics",
        trigger_dag_id="model_metrics",
        wait_for_completion=True,
        deferrable=True,
    )
    dag_for_get_prediction = TriggerDagRunOperator(
        task_id="dag_for_get_prediction",
        trigger_dag_id="get_prediction",
        wait_for_completion=True,
        deferrable=True,
    )
    input_data = uploading_data(VAR_DATASET_ROOT_DIR, VAR_DATASET_DIR_NAME)
    loaded_models = load_model(VAR_MODEL_ROOT_DIR, VAR_MODEL_DIR_NAME, VAR_CURRENT_MODEL_REGRESSION_FILE_NAME, VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME)
    deployed_models = deployed_model(loaded_models)
    data_with_result = get_predictions(input_data, deployed_models, VAR_TARGET_FEATURES)
    check_bias_data_distribution()
    check_model_quality()
    v = False
    if v:
        dag_for_data_evaluation >> dag_for_preparation >> dag_for_model_train >> dag_for_model_metrics >> dag_for_get_prediction


start_observer_dag = start_observer()
