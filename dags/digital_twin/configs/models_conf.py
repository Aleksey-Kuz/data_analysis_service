""" Config for CatBoost Model """


class CatBoostModelConf:
    """
    Config class for CatBoost Classification and Regression Model
    """

    model_params = {
        "iterations": 1000,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": None,
        "eval_metric": None,
        "early_stopping_rounds": 50
    }
    reg_loss_function = "RMSE"
    reg_eval_metric = "RMSE"
    cla_loss_function = "MultiClass"
    cla_eval_metric = "MultiClass"


class LinearRegressionConf:
    """
    Config class for Linear Regression Model
    """

    model_params = {
        "fit_intercept": True,
        "copy_X": True,
        "n_jobs": -1,
        "positive": False
    }


class LogisticRegressionConf:
    """
    Config class for Logistic Regression Model
    """

    model_params = {
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
        "n_jobs": -1
    }
