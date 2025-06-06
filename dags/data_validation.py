""" DAG for checking data quality """

from datetime import datetime, timedelta

from airflow.decorators import dag
from airflow.models import Variable
from airflow.models.param import Param

from tasks.data_validation_tasks import (
    uploading_data,
    check_missing_values,
    check_duplicates,
    check_data_types,
    check_emission_data
)


@dag(
    dag_id="data_validation",
    start_date=datetime(2024, 1, 1),
    params={"filename": "test_car_csv.csv"},
    schedule_interval=None,
    catchup=False,
    dagrun_timeout=timedelta(minutes=30),
    access_control={
            "User": ["can_edit", "can_read"],
        }
)
def data_validation():
    """ DAG flow """
    issues_df = uploading_data()
    issues_df = check_missing_values(issues_df)
    issues_df = check_duplicates(issues_df)
    issues_df = check_data_types(issues_df)
    issues_df = check_emission_data(issues_df)
    return issues_df


data_validation_dag = data_validation()
