"""Tasks for STT/ASR DAGs"""

from airflow.decorators import task


@task
def evaluate_data_quality():
    """Evaluate the quality of the received data by ydata-profiling"""
    return ""


@task
def save_evaluation_results(evaluation_results):
    """Save the evaluation results to a database or file"""
    # Here you would implement the logic to save the results
    # For example, saving to a database or writing to a file
    return evaluation_results
