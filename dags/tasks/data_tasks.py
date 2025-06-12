"""Tasks for data processing"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from airflow.decorators import task
from airflow.models import Variable

import pandas as pd
import numpy as np
from loguru import logger
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder


@task
def uploading_data(var_dataset_root_dir: str, var_dataset_dir_name: str, **context) -> pd.DataFrame:
    """Upload data to a storage system"""
    dataset_root_dir = Variable.get(var_dataset_root_dir)
    dataset_dir_name = Variable.get(var_dataset_dir_name)
    file_name = context.get("params").get("file_name")
    file_path = Path(dataset_root_dir) / dataset_dir_name / file_name
    logger.info(f"Uploading data from {file_path}.")
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    logger.info(f"File {file_path} found. Reading data.")
    data = pd.read_csv(file_path)
    logger.info(f"Data read successfully with shape {data.shape}.")
    return data


@task
def evaluate_data_quality(data: pd.DataFrame, **context) -> Dict[str, Any]:
    """Evaluate the quality of the received data by ydata-profiling"""
    file_name = context.get("params").get("file_name")
    date_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logger.info(f"Evaluating data quality for {file_name} at {date_now}.")
    profile = ProfileReport(
        data,
        title=f"Data Quality Report for {file_name} | {date_now}",
        explorative=True,
        minimal=False
    )
    logger.info("Data quality evaluation completed.")
    result = {
        "file_name": file_name,
        "date": date_now,
        "profile": profile
    }
    return result


@task
def save_evaluation_results(var_evaluation_root_dir: str, var_evaluation_dir_name: str, 
                            file_name: str, date: str, profile: ProfileReport) -> None:
    """Save the evaluation results to a database or file"""
    evaluation_root_dir = Variable.get(var_evaluation_root_dir)
    evaluation_dir_name = Variable.get(var_evaluation_dir_name)
    output_file = f"data_evaluation_{file_name}_{date}.html"
    output_path = Path(evaluation_root_dir) / evaluation_dir_name / output_file
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Directory {output_path.parent} does not exist.")
    profile.to_file(output_path)
    logger.info(f"Evaluation results saved to {output_path}.")


@task
def preparate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for further use in machine learning models"""
    pass


@task
def save_data(data):
    """Save data to a database"""
    # Here you would implement the logic to save data to a database
    # For example, using SQLAlchemy or a similar library
    return "Data saved to database successfully"
