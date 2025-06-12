"""DAG for preparing data for further use in machine learning models"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from conf.data_config import Config
from tasks.data_tasks import (
    uploading_data,
    preparate_data,
    save_data
)

VAR_DATASET_ROOT_DIR = Config.VAR_DATASET_ROOT_DIR
VAR_DATASET_DIR_NAME = Config.VAR_DATASET_DIR_NAME
VAR_DATASET_FILE_NAME = Config.VAR_DATASET_FILE_NAME

VAR_PREPARED_DATA_ROOT_DIR = Config.VAR_PREPARED_DATA_ROOT_DIR
VAR_PREPARED_DATA_DIR_NAME = Config.VAR_PREPARED_DATA_DIR_NAME

VAR_OUTPUT_FILE_TYPE = Config.VAR_OUTPUT_FILE_TYPE

PREPARED_FILE_PREFIX = Config.PREPARED_FILE_PREFIX


@dag(
    dag_id="data_preparation",
    start_date=datetime(2024, 1, 1),
    params={
        "file_name": Param(Variable.get(VAR_DATASET_FILE_NAME)),
        "file_type": Param(Variable.get(VAR_OUTPUT_FILE_TYPE), enum=["csv", "xlsx"]),},
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def data_preparation():
    """DAG flow"""
    input_data = uploading_data(VAR_DATASET_ROOT_DIR, VAR_DATASET_DIR_NAME)
    prepared_data = preparate_data(input_data)
    save_data(VAR_PREPARED_DATA_ROOT_DIR, VAR_PREPARED_DATA_DIR_NAME, PREPARED_FILE_PREFIX, prepared_data)


data_preparation_dag = data_preparation()
