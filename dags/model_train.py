import logging
import random
import string
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

def load_dataset(filename, **kwargs):
    logger.info(f"Loading dataset from file: {filename}")

def train_models(model_name, **kwargs):
    logger.info(f"Training models for model: {model_name}")

def evaluate_models(**kwargs):
    logger.info("Evaluating models to select the best one")

def save_model(**kwargs):
    model_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    logger.info(f"Saving model with generated ID: {model_id}")
    kwargs['ti'].xcom_push(key='saved_model_id', value=model_id)

with DAG(
    'train_model',
    default_args={'owner': 'airflow'},
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    params={
        'model_name': 'default_model',
        'filename': 'test_car_csv.csv',
    },
) as dag:

    load = PythonOperator(
        task_id='load_dataset',
        python_callable=load_dataset,
        op_kwargs={'filename': "{{ params.filename }}"},
    )

    train = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
        op_kwargs={'model_name': "{{ params.model_name }}"},
    )

    evaluate = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
    )

    save = PythonOperator(
        task_id='save_model',
        python_callable=save_model,
    )

    load >> train >> evaluate >> save
