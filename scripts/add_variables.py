import logging
import os
from airflow.models import Variable


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


ENV_VARS = {
    "MODEL_ROOT_DIR": os.getenv("MODEL_ROOT_DIR", "/opt/airflow/models"),
    "MODEL_DIR_NAME": os.getenv("MODEL_DIR_NAME", "demo_models"),
    "MODEL_DEFAULT": os.getenv("MODEL_DEFAULT", "catboost"),
    "DATASET_ROOT_DIR": os.getenv("DATASET_ROOT_DIR", "/opt/airflow/data"),
    "DATASET_DIR_NAME": os.getenv("DATASET_DIR_NAME", "demo_datasets"),
    "DATASET_FILE_NAME": os.getenv("DATASET_FILE_NAME", "demo_dataset.csv"),
    "EVALUATION_ROOT_DIR": os.getenv("EVALUATION_ROOT_DIR", "/opt/airflow/evaluations"),
    "EVALUATION_DIR_NAME": os.getenv("EVALUATION_DIR_NAME", "data_evaluation"),
}


def add_variadle(key, val):
    try:
        Variable.set(key, val)
        logging.info(f"Variable {key} set successfully to {val}")
    except Exception:
        logging.exception("Failed to set variable")
        exit(1)


def main():
    for key, val in ENV_VARS.items():
        if val is not None:
            add_variadle(key, val)


if __name__ == "__main__":
    main()
