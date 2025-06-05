""" Functions for downloading data from files """

import pandas as pd

from pathlib import Path


def check_exist_file(filepath: Path) -> None:
    """ Checking the file path for existence """
    if Path.exists(filepath):
        raise FileExistsError(f"The file was not found in the specified path {filepath}.")


def load_from_csv(filepath: Path) -> pd.DataFrame:
    """ Downloading data from a CSV file """
    check_exist_file(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        data = pd.read_csv(file)
    return data


def load_from_excel(filepath: Path) -> pd.DataFrame:
    """ Downloading data from an Excel file """
    check_exist_file(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        data = pd.read_excel(file)
    return data
