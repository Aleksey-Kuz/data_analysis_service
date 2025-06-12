"""Dag for getting model metrics"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from tasks.data_tasks import (
    uploading_data
)
from tasks.model_tasks import (
    load_model,
    deployed_model,
    get_model_metrics
)


@dag(
    dag_id="model_metrics",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def model_metrics():
    """DAG flow"""
    input_data = uploading_data()
    load_model()
    deployed_model()
    get_model_metrics()


model_metrics_dag = model_metrics()
