""" Tasks for checking data quality """

from pathlib import Path

from airflow.decorators import task
from airflow.models import Variable
import pandas as pd
from loguru import logger


@task
def uploading_data(**context) -> pd.DataFrame:
    """ Task to upload data """
    filename = context.get("params").get("filename")
    path_to_datasets = "/opt/airflow/data/datasets"
    path_to_dataset = Path(path_to_datasets) / filename
    if not path_to_dataset.exists():
        raise FileNotFoundError(f"File {filename} not found in {path_to_datasets}.")
    logger.info(f"File {filename} found in {path_to_datasets}.")
    df = pd.read_csv(filename)
    logger.info(f"Uploading data from {path_to_dataset}")
    return df


@task
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """ Task to check for missing values in the dataframe """
    logger.info("Checking for missing values in the dataframe.")
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
    else:
        logger.info("No missing values found.")
    return df


@task
def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """ Task to check for duplicate rows in the dataframe """
    logger.info("Checking for duplicate rows in the dataframe.")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")
    return df


@task
def check_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """ Task to check data types of the dataframe """
    logger.info("Checking data types of the dataframe.")
    data_types = df.dtypes
    logger.info(f"Data types:\n{data_types}")
    return df


@task
def check_emission_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Task to check emission data for specific conditions """
    logger.info("Checking emission data for specific conditions.")
    
    # Example condition: Check if 'emission' column exists and has valid values
    if 'emission' not in df.columns:
        raise ValueError("Column 'emission' does not exist in the dataframe.")
    
    invalid_emissions = df[df['emission'] < 0]
    if not invalid_emissions.empty:
        logger.warning(f"Found {len(invalid_emissions)} rows with invalid emissions (negative values).")
    else:
        logger.info("All emission values are valid (non-negative).")
    
    return df
