import logging

import mlflow
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from zenml.client import Client

from src.model import LinearRegressionModel
from src.config import ModelNameConfig


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: ModelNameConfig
) -> RegressorMixin:
    """
    Training model
    :param df: pandas dataframe
    :return:
    """
    logging.info("Training model...")
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error occurred while training model: {str(e)}")
        raise e
