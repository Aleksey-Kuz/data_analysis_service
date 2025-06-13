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
    "MODEL_DEFAULT_CLASSIFICATION": os.getenv("MODEL_DEFAULT_CLASSIFICATION", "CatBoostModel"),
    "MODEL_DEFAULT_REGRESSION": os.getenv("MODEL_DEFAULT_REGRESSION", "CatBoostModel"),
    "TARGET_FEATURES": os.getenv("TARGET_FEATURES", '{"target_column_name": "regression" | "classification"}'),
    "SPLIT_TEST_SIZE": os.getenv("SPLIT_TEST_SIZE", 0.2),
    "COMPARE_METRIC_REGRESSION": os.getenv("COMPARE_METRIC_REGRESSION", "r2"),
    "COMPARE_METRIC_CLASSIFICATION": os.getenv("COMPARE_METRIC_CLASSIFICATION", "f1"),
    "CURRENT_MODEL_REGRESSION_FILE_NAME": os.getenv("CURRENT_MODEL_REGRESSION_FILE_NAME", ""),
    "CURRENT_MODEL_CLASSIFICATION_FILE_NAME": os.getenv("CURRENT_MODEL_CLASSIFICATION_FILE_NAME", ""),
    "DATASET_ROOT_DIR": os.getenv("DATASET_ROOT_DIR", "/opt/airflow/data"),
    "DATASET_DIR_NAME": os.getenv("DATASET_DIR_NAME", "demo_datasets"),
    "DATASET_FILE_NAME": os.getenv("DATASET_FILE_NAME", "demo_dataset.csv"),
    "EVALUATION_ROOT_DIR": os.getenv("EVALUATION_ROOT_DIR", "/opt/airflow/evaluations"),
    "EVALUATION_DIR_NAME": os.getenv("EVALUATION_DIR_NAME", "data_evaluation"),
    "RESULTS_ROOT_DIR": os.getenv("RESULTS_ROOT_DIR", "/opt/airflow/data"),
    "RESULTS_DIR_NAME": os.getenv("RESULTS_DIR_NAME", "model_results"),
    "PREPARED_DATA_ROOT_DIR": os.getenv("PREPARED_DATA_ROOT_DIR", "/opt/airflow/data"),
    "PREPARED_DATA_DIR_NAME": os.getenv("PREPARED_DATA_DIR_NAME", "prepared_data"),
    "OUTPUT_FILE_TYPE": os.getenv("OUTPUT_FILE_TYPE", "csv"),
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
