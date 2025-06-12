"""DAG to evaluate the quality of the received data"""

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from tasks.data_evaluation_tasks import (
    evaluate_data_quality,
    save_evaluation_results,
)


@dag(
    dag_id="data_evaluation",
    start_date=datetime(2024, 1, 1),
    params={"language": Param(Variable.get(), enum=["ru", "eng"])},
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
        "User": ["can_edit", "can_read"],
    },
)
def data_evaluation():
    """DAG flow"""
    evaluation_results = evaluate_data_quality()
    save_evaluation_results(evaluation_results)


data_evaluation_dag = data_evaluation()
