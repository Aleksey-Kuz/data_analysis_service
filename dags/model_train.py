"""Dag for model training"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from tasks.data_tasks import (
    uploading_data
)
from tasks.model_tasks import (
    models_training,
    models_evaluating,
    choice_model,
    save_model
)


@dag(
    dag_id="model_train",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def model_train():
    """DAG flow"""
    input_data = uploading_data()
    models_training()
    models_evaluating()
    choice_model()
    save_model()


model_train_dag = model_train()
