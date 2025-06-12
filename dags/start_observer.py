"""Dag for start observer"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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


@dag(
    dag_id="start_observer",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
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
    input_data = uploading_data()
    load_model()
    deployed_model()
    get_predictions()
    check_bias_data_distribution()
    check_model_quality()
    v = True
    if v:
        dag_for_data_evaluation >> dag_for_preparation >> dag_for_model_train >> dag_for_model_metrics >> dag_for_get_prediction


start_observer_dag = start_observer()
