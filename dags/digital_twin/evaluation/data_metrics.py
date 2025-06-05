""" Calculating metrics for data """

import pandas as pd

from typing import List

from digital_twin.configs.constants import DataMetricsConstants


class DataMetrics:
    """
    A class containing a set of various metrics for data
    """

    def __init__(self):
        self.completeness_threshold = DataMetricsConstants.COMPLETENESS_THRESHOLD

    def validation(self, df: pd.DataFrame, target_features: List[str]) -> bool:
        """ Validation by all implemented check methods """
        self.check_dataframe_is_not_empty(df)
        self.check_dataframe_contains_target_features(df, target_features)
        self.check_dataframe_completeness(df)
        return True

    @staticmethod
    def check_dataframe_is_not_empty(df: pd.DataFrame) -> None:
        """ Checking the DataFrame for emptiness """
        if df.empty:
            raise ValueError("The resulting DataFrame is empty.")

    @staticmethod
    def check_dataframe_contains_target_features(df: pd.DataFrame, target_features: List[str]) -> None:
        """ Checking the DataFrame for the content of target features """
        columns = df.columns
        if columns.empty:
            raise ValueError("There are no columns in the dataframe.")
        elif len(columns) <= len(target_features):
            raise ValueError(
                "The number of target features is equal to or greater than the number of columns in the DataFrame."
            )
        for target_feature in target_features:
            if target_feature not in columns:
                raise ValueError(
                    f"The specified {target_feature} target feature was not found in the features DataFrame."
                )

    def check_dataframe_completeness(self, df: pd.DataFrame) -> None:
        """ Checking the DataFrame for completeness """
        completeness = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if completeness > self.completeness_threshold:
            raise ValueError(
                f"The number of empty values in the DataFrame exceeds {self.completeness_threshold}. "
                "To change the completeness threshold, refer to the config files."
            )
