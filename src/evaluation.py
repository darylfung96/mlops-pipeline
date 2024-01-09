import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(Evaluation):
    """
    Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate Mean Squared Error
        :param y_true:
        :param y_pred:
        :return:
        """
        try:
            logging.info("Calculating Mean Squared Error...")
            mse = np.mean((y_true - y_pred) ** 2)
            logging.info(f"Mean Squared Error: {mse}")
            return mse

        except Exception as e:
            logging.error(f"Error calculating Mean Squared Error: {e}")
            raise e


class R2(Evaluation):
    """
    R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate R2 Score
        :param y_true:
        :param y_pred:
        :return:
        """
        try:
            logging.info("Calculating R2 Score...")
            r2 = np.mean((y_true - y_pred) ** 2)
            logging.info(f"R2 Score: {r2}")
            return r2

        except Exception as e:
            logging.error(f"Error calculating R2 Score: {e}")
            raise e


class RMSE(Evaluation):
    """
    Root Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate Root Mean Squared Error
        :param y_true:
        :param y_pred:
        :return:
        """
        try:
            logging.info("Calculating Root Mean Squared Error...")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse

        except Exception as e:
            logging.error(f"Error calculating Root Mean Squared Error: {e}")
            raise e