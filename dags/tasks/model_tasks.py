"""Tasks for models processing"""

from airflow.decorators import task


@task
def models_training():
    """ """
    return list()


@task
def models_evaluating():
    """ """
    return list()


@task
def choice_model():
    """ """
    return dict()


@task
def save_model():
    """ """
    return None


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
