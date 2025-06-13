"""Tasks for models processing"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Union, Type
import joblib

from airflow.decorators import task
from airflow.models import Variable
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from conf.model_config import Config
from digital_twin.models.base_model import BaseModel
from digital_twin.evaluation.data_metrics import DataMetrics
from digital_twin.models.catboost_model import CatBoostModel
from digital_twin.metrics.regression_metrics import RegressionMetrics
from digital_twin.metrics.classification_metrics import ClassificationMetrics

BASE_MODEL_TYPE = Config.BASE_MODEL_TYPE
MODEL_REGISTRY = Config.MODEL_REGISTRY
RANDOM_STATE = Config.RANDOM_STATE


def train_model(model_cls: Type[BaseModel], task_type: str,
                x_train: pd.DataFrame, y_train: pd.Series,
                cat_features: List[str]) -> BASE_MODEL_TYPE:
    """Helper function to instantiate and train a model based on its class."""
    if model_cls == CatBoostModel:
        model = model_cls(task_type=task_type)
        model.train(x_train, y_train, cat_features=cat_features)
    else:
        model = model_cls()
        model.train(x_train, y_train, cat_features=cat_features)
    return model


@task
def check_data_quality(data: pd.DataFrame) -> None:
    """Check the quality of the data"""
    data_metrics = DataMetrics()
    logger.info("Starting data quality check.")
    logger.info("Checking if the DataFrame is not empty.")
    data_metrics.check_dataframe_is_not_empty(data)
    logger.info("Checking if the DataFrame completeness is within the threshold.")
    data_metrics.check_dataframe_completeness(data)
    logger.info("Checking stage is completed successfully.")


@task
def data_splitting(
    data: pd.DataFrame,
    var_target_features: str,
    var_split_test_size: str
) -> Dict[str, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.

    Parameters:
        data (pd.DataFrame): Input dataset.
        var_target_features (str): Airflow Variable containing a JSON string mapping target column names to task types.
        var_split_test_size (str): Airflow Variable specifying the proportion of the dataset to include in the test split.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing training and testing sets.
    """
    target_features = list(json.loads(Variable.get(var_target_features)).keys())
    if not isinstance(target_features, list):
        raise ValueError(f"Expected target_features to be a dict, got {type(target_features)} instead.")
    test_size = float(Variable.get(var_split_test_size))
    logger.info(f"Target features: {target_features}")
    logger.info(f"Test size: {test_size}")

    logger.info("Starting data splitting.")
    x = data.drop(columns=target_features)
    y = data[target_features]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=RANDOM_STATE
    )

    data_train = pd.concat([x_train, y_train], axis=1)
    data_test = pd.concat([x_test, y_test], axis=1)

    result = {
        "data_train": data_train,
        "data_test": data_test,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test
    }
    
    logger.info("Data splitting completed successfully.")
    return result


@task
def models_training(
    data: pd.DataFrame,
    var_model_default_classification: str,
    var_model_default_regression: str,
    var_target_features: str
) -> Dict[str, List[BASE_MODEL_TYPE]]:
    """
    Train models for specified target features and task types.

    This task dynamically selects and trains ML models for each target column
    based on the task type (regression or classification) and model selection.

    Parameters:
        data (pd.DataFrame): Input dataset including features and target columns.
        var_model_default_classification (str): Airflow Variable specifying the default model for classification tasks.
        var_model_default_regression (str): Airflow Variable specifying the default model for regression tasks.
            Use "default" to train all models available for that task type.
        var_target_features (str): Airflow Variable containing a JSON string mapping target column names to task types.

    Returns:
        Dict[str, List[BaseModel]]:
            Dictionary mapping each target column to a list of trained model instances.
            Example:
            {
                "price": [CatBoostModel(), LinearRegressionModel()],
                "churn": [CatBoostModel()]
            }
    """
    logger.info("Starting model training.")

    model_regression = Variable.get(var_model_default_regression)
    model_classification = Variable.get(var_model_default_classification)
    model_names = {
        "regression": model_regression,
        "classification": model_classification
    }
    logger.info(f"Model names provided: {model_names}.")

    target_features = json.loads(Variable.get(var_target_features))
    logger.info(f"Target features provided: {target_features}")

    cat_features = [col for col in data.columns if data[col].dtype == "object"]
    results: Dict[str, List[BASE_MODEL_TYPE]] = dict()

    for target_name, task_type in target_features.items():
        logger.info(f"Processing target '{target_name}' with task type '{task_type}'.")

        if task_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported task type: '{task_type}'")
        if target_name not in data.columns:
            raise ValueError(f"Target column '{target_name}' not found in dataset.")

        # Use all columns except all targets as features
        x_train = data.drop(columns=target_features.keys())
        y_train = data[target_name]

        # Get requested model or default
        requested_model_key = model_names.get(f"{task_type}", "default")
        if requested_model_key == "default":
            selected_models = MODEL_REGISTRY[task_type].values()
        else:
            if requested_model_key not in MODEL_REGISTRY[task_type]:
                raise ValueError(f"Model '{requested_model_key}' is not supported for task type '{task_type}'")
            selected_models = [MODEL_REGISTRY[task_type][requested_model_key]]
        logger.info(f"Selected models for target '{target_name}': {[model.__name__ for model in selected_models]}")

        # Train each selected model
        trained = [
            train_model(model_cls, task_type, x_train, y_train, cat_features)
            for model_cls in selected_models
        ]
        logger.info(f"Trained models for target '{target_name}': {[model.__class__.__name__ for model in trained]}")

        results[target_name] = trained

    logger.info("Model training completed successfully.")
    logger.debug(f"Trained models: {results}")

    return results


@task
def models_evaluating(
    data: pd.DataFrame,
    models: Dict[str, List[BASE_MODEL_TYPE]],
    var_target_features: str,
    var_compare_metric_regression: str,
    var_compare_metric_classification: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all trained models on the given dataset and return their scores grouped by task type.

    Parameters:
        data (pd.DataFrame): The test dataset including feature and target columns.
        models (Dict[str, List[BaseModel]]): Dictionary with model lists per target feature.
            Example: {"market_value": [model1], "failure_category": [model2]}
        var_target_features (str): Airflow Variable name with JSON dict {"target_feature": "regression"/"classification"}.
        var_compare_metric_regression (str): Regression metric name (e.g., "rmse", "r2").
        var_compare_metric_classification (str): Classification metric name (e.g., "f1", "accuracy").

    Returns:
        Dict[str, Dict[str, float]]: Dictionary with structure like:
            {
                "regression": {"CatBoostModel": 0.95, "LinearRegressionModel": 0.88},
                "classification": {"CatBoostModel": 0.92}
            }
    """
    target_features = json.loads(Variable.get(var_target_features))
    metric_reg = Variable.get(var_compare_metric_regression)
    metric_cls = Variable.get(var_compare_metric_classification)

    logger.info("Starting model evaluation.")
    logger.info(f"Target features: {list(target_features.keys())}")
    logger.info(f"Regression metric: {metric_reg}")
    logger.info(f"Classification metric: {metric_cls}")

    results: Dict[str, Dict[str, float]] = {
        "regression": {},
        "classification": {}
    }

    all_targets = list(target_features.keys())
    features_only = data.drop(columns=all_targets)

    for target, model_list in models.items():
        if target not in target_features:
            raise ValueError(f"Target '{target}' not found in target features mapping.")

        task_type = target_features[target]
        if task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task type: '{task_type}' for target '{target}'")

        logger.info(f"Evaluating models for target '{target}' with task type '{task_type}'.")

        y_true = data[target]

        for model in model_list:
            model_name = model.__class__.__name__
            logger.info(f"Evaluating model '{model_name}' on target '{target}'.")

            y_pred = model.predict(features_only)

            if task_type == "regression":
                metric_fn = getattr(RegressionMetrics, metric_reg, None)
                if not metric_fn:
                    raise ValueError(f"Unsupported regression metric: '{metric_reg}'")
                score = metric_fn(y_true, y_pred)
            else:
                metric_fn = getattr(ClassificationMetrics, metric_cls, None)
                if not metric_fn:
                    raise ValueError(f"Unsupported classification metric: '{metric_cls}'")
                try:
                    y_proba = model.predict_proba(features_only)
                except Exception:
                    y_proba = None
                if metric_cls in {"log_loss", "roc_auc"} and y_proba is not None:
                    score = metric_fn(y_true, y_proba)
                else:
                    score = metric_fn(y_true, y_pred)

            logger.info(f"Model '{model_name}' score for target '{target}': {score:.4f}")
            results[task_type][model_name] = score

    return results


@task
def choice_models(
    models: Dict[str, List[BASE_MODEL_TYPE]],
    evaluations: Dict[str, Dict[str, float]],
    var_target_features: Dict[str, str]
) -> Dict[str, BASE_MODEL_TYPE]:
    """
    Select the best model instance for each task type based on evaluation scores.

    Parameters:
        models: Dict with target names as keys and lists of model instances as values.
            Example: {"market_value": [model1], "failure_category": [model2]}
        evaluations: Dict with task types as keys ("regression", "classification")
            and dicts of {model_class_name: score} as values.
        var_target_features (str): Airflow Variable name with JSON dict {"target_feature": "regression"/"classification"}.

    Returns:
        Dict mapping task type ("regression", "classification") to best model instance.
    """
    logger.info("Starting model selection based on evaluations.")
    target_features = json.loads(Variable.get(var_target_features))
    best_models: Dict[str, BASE_MODEL_TYPE] = {}

    for task_type, evals in evaluations.items():
        logger.info(f"Selecting best model for task type '{task_type}'.")
        if not evals:
            continue

        if task_type == "regression":
            best_metric = min(evals.values())
        elif task_type == "classification":
            best_metric = max(evals.values())
        else:
            raise ValueError(f"Unsupported task type: '{task_type}'")

        best_model_name = next(name for name, score in evals.items() if score == best_metric)
        targets_for_task = [t for t, tt in target_features.items() if tt == task_type]

        found_model = None
        for target in targets_for_task:
            logger.info(f"Searching for best model '{best_model_name}' for target '{target}'.")
            for model in models.get(target, []):
                if model.__class__.__name__.lower() == best_model_name.lower():
                    found_model = model
                    break
            if found_model:
                break
        if not found_model:
            raise ValueError(f"Model '{best_model_name}' for task type '{task_type}' not found among provided models.")

        logger.info(f"Best model for task type '{task_type}' is '{found_model.__class__.__name__}' with score {best_metric:.4f}.")
        best_models[task_type] = found_model

    return best_models


@task
def save_model(
    best_models: Dict[str, "BASE_MODEL_TYPE"],
    var_model_root_dir: str,
    var_model_dir_name: str,
    var_current_model_regression_file_name: str,
    var_current_model_classification_file_name: str
) -> None:
    """
    Save the best models to disk and update Airflow variables with their filenames.

    Args:
        best_models (Dict[str, BASE_MODEL_TYPE]): Dictionary with keys like 'regression', 'classification'
            and values are the best trained model objects.
        var_model_root_dir (str): Root directory path to save models.
        var_model_dir_name (str): Subdirectory name inside the root directory to save models.
        var_current_model_regression_file_name (str): Airflow variable key to save latest regression model filename.
        var_current_model_classification_file_name (str): Airflow variable key to save latest classification model filename.

    Returns:
        None
    """
    model_root_dir = Variable.get(var_model_root_dir)
    model_dir_name = Variable.get(var_model_dir_name)
    models_path = Path(model_root_dir) / model_dir_name
    if not models_path.exists():
        raise FileNotFoundError(f"Model directory {models_path} does not exist.")
    logger.info(f"Saving models to {models_path}.")
    date_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    saved_filenames = {}

    for model_type, model in best_models.items():
        model_class_name = model.__class__.__name__.lower()
        filename = f"{model_type}_{model_class_name}_{date_now}.pkl"
        file_path = models_path / filename
        joblib.dump(model, file_path)

        saved_filenames[model_type] = filename

    if "regression" in saved_filenames:
        Variable.set(var_current_model_regression_file_name, saved_filenames["regression"])
    if "classification" in saved_filenames:
        Variable.set(var_current_model_classification_file_name, saved_filenames["classification"])



@task
def load_model():
    """ """
    return None


@task
def deployed_model():
    """ """
    return None


@task
def get_predictions():
    """ """
    return None


@task
def get_model_metrics():
    """ """
    return None
