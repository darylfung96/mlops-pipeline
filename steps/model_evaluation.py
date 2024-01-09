import logging

import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

from src.evaluation import MSE, R2, RMSE


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_evaluation(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[
    Annotated[float, "mse"],
    Annotated[float, "r2"],
    Annotated[float, "rmse"],
]:
    """
    Model evaluation
    :param model: regression model
    :param X_test: test data
    :param y_test: target values for test data
    :return: tuple of mse, r2, and rmse
    """
    try:
        logging.info("Model evaluation...")
        prediction = model.predict(X_test.values)
        mse = MSE().calculate_scores(y_test.values, prediction)
        mlflow.log_metric("MSE", mse)
        r2 = R2().calculate_scores(y_test.values, prediction)
        mlflow.log_metric("R2", r2)
        rmse = RMSE().calculate_scores(y_test.values, prediction)
        mlflow.log_metric("RMSE", rmse)

        return mse, r2, rmse
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        return None, None, None
