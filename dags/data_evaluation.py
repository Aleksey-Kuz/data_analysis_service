"""DAG to evaluate the quality of the received data"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from conf.data_config import Config
from tasks.data_tasks import (
    uploading_data,
    evaluate_data_quality,
    save_evaluation_results,
)

VAR_DATASET_ROOT_DIR = Config.VAR_DATASET_ROOT_DIR
VAR_DATASET_DIR_NAME = Config.VAR_DATASET_DIR_NAME
VAR_DATASET_FILE_NAME = Config.VAR_DATASET_FILE_NAME

VAR_EVALUATION_ROOT_DIR = Config.VAR_EVALUATION_ROOT_DIR
VAR_EVALUATION_DIR_NAME = Config.VAR_EVALUATION_DIR_NAME


@dag(
    dag_id="data_evaluation",
    start_date=datetime(2024, 1, 1),
    params={"file_name": Param(Variable.get(VAR_DATASET_FILE_NAME))},
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def data_evaluation():
    """DAG to evaluate the quality of the received data"""
    input_data = uploading_data(VAR_DATASET_ROOT_DIR, VAR_DATASET_DIR_NAME)
    evaluation_results = evaluate_data_quality(input_data)
    save_evaluation_results(VAR_EVALUATION_ROOT_DIR, VAR_EVALUATION_DIR_NAME, evaluation_results["file_name"],
                            evaluation_results["date"], evaluation_results["profile"])


data_evaluation_dag = data_evaluation()
