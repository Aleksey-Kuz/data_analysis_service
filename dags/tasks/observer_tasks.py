"""Tasks for observer processing"""

from airflow.decorators import task


@task
def check_bias_data_distribution():
    """ """
    return None


@task
def check_model_quality():
    """ """
    return None
