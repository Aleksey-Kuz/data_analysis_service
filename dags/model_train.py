"""Dag for model training"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from conf.data_config import Config as ConfigData
from conf.model_config import Config as ConfigModel
from tasks.data_tasks import (
    uploading_data,
    preparate_data
)
from tasks.model_tasks import (
    check_data_quality,
    data_splitting,
    models_training,
    models_evaluating,
    choice_models,
    save_model
)

VAR_DATASET_ROOT_DIR = ConfigData.VAR_DATASET_ROOT_DIR
VAR_DATASET_DIR_NAME = ConfigData.VAR_DATASET_DIR_NAME
VAR_DATASET_FILE_NAME = ConfigData.VAR_DATASET_FILE_NAME

VAR_MODEL_ROOT_DIR = ConfigModel.VAR_MODEL_ROOT_DIR
VAR_MODEL_DIR_NAME = ConfigModel.VAR_MODEL_DIR_NAME
VAR_MODEL_DEFAULT_CLASSIFICATION = ConfigModel.VAR_MODEL_DEFAULT_CLASSIFICATION
VAR_MODEL_DEFAULT_REGRESSION = ConfigModel.VAR_MODEL_DEFAULT_REGRESSION
VAR_TARGET_FEATURES = ConfigModel.VAR_TARGET_FEATURES
VAR_SPLIT_TEST_SIZE = ConfigModel.VAR_SPLIT_TEST_SIZE
VAR_COMPARE_METRIC_REGRESSION = ConfigModel.VAR_COMPARE_METRIC_REGRESSION
VAR_COMPARE_METRIC_CLASSIFICATION = ConfigModel.VAR_COMPARE_METRIC_CLASSIFICATION
VAR_CURRENT_MODEL_REGRESSION_FILE_NAME =ConfigModel.VAR_CURRENT_MODEL_REGRESSION_FILE_NAME
VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME = ConfigModel.VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME


@dag(
    dag_id="model_train",
    start_date=datetime(2024, 1, 1),
    params={"file_name": Param(Variable.get(VAR_DATASET_FILE_NAME))},
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def model_train():
    """DAG flow"""
    input_data = uploading_data(VAR_DATASET_ROOT_DIR, VAR_DATASET_DIR_NAME)
    check_data_quality(input_data)
    prepared_data = preparate_data(input_data)
    splitting_data = data_splitting(prepared_data, VAR_TARGET_FEATURES, VAR_SPLIT_TEST_SIZE)
    models = models_training(splitting_data["data_train"], VAR_MODEL_DEFAULT_CLASSIFICATION, VAR_MODEL_DEFAULT_REGRESSION, VAR_TARGET_FEATURES)
    evaluations = models_evaluating(splitting_data["data_test"], models, VAR_TARGET_FEATURES, VAR_COMPARE_METRIC_REGRESSION, VAR_COMPARE_METRIC_CLASSIFICATION)
    best_models = choice_models(models, evaluations)
    save_model(best_models, VAR_MODEL_ROOT_DIR, VAR_MODEL_DIR_NAME, VAR_CURRENT_MODEL_REGRESSION_FILE_NAME, VAR_CURRENT_MODEL_CLASSIFICATION_FILE_NAME)


model_train_dag = model_train()
