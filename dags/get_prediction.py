"""Dag for getting predictions"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from tasks.data_tasks import (
    uploading_data,
    save_data
)
from tasks.model_tasks import (
    load_model,
    deployed_model,
    get_predictions
)


@dag(
    dag_id="get_prediction",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def get_prediction():
    """DAG flow"""
    input_data = uploading_data()
    load_model()
    deployed_model()
    get_predictions()
    save_data(input_data)


get_prediction_dag = get_prediction()
