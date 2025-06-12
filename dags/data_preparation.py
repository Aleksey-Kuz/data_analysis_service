"""DAG for preparing data for further use in machine learning models"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from tasks.data_tasks import (
    uploading_data,
    preparate_data,
    save_data
)


@dag(
    dag_id="data_preparation",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def data_preparation():
    """DAG flow"""
    input_data = uploading_data()
    prepare_data = preparate_data()
    save_data(prepare_data)


data_preparation_dag = data_preparation()
