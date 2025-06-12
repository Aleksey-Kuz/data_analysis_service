import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

def load_dataset(filename, **kwargs):
    logger.info(f"Loading dataset from file: {filename}")

def deploy_model(model_name, **kwargs):
    logger.info(f"Deploying model with name: {model_name}")

def get_predictions(**kwargs):
    logger.info("Generating predictions using the deployed model")

def save_predictions(results_filename, **kwargs):
    logger.info(f"Saving predictions to file: {results_filename}")

with DAG(
    'predict_model_dag',
    default_args={'owner': 'airflow'},
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    params={
        'dataset_name': 'test_cat_csv.csv',
        'results_filename': 'result_test_car_csv.csv',
        'model_name': 'default_model',
    },
) as dag:

    load = PythonOperator(
        task_id='load_dataset',
        python_callable=load_dataset,
        op_kwargs={'filename': "{{ params.dataset_name }}"},
    )

    deploy = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        op_kwargs={'model_name': "{{ params.model_name }}"},
    )

    predict = PythonOperator(
        task_id='get_predictions',
        python_callable=get_predictions,
    )

    save = PythonOperator(
        task_id='save_predictions',
        python_callable=save_predictions,
        op_kwargs={'results_filename': "{{ params.results_filename }}"},
    )

    load >> deploy >> predict >> save
