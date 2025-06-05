""" Functions for different data partitioning """

import pandas as pd
from sklearn.model_selection import train_test_split

from typing import Dict, List, Union


RANDOM_STATE = 42


def simple_split(data: pd.DataFrame, target_features: List[str],
                 test_size: Union[float, int]) -> Dict[str, pd.DataFrame]:
    """ Primitive data set partitioning function """
    x = data.drop(target_features, axis=1)
    y = data[target_features]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=RANDOM_STATE)
    result = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test
    }
    return result
