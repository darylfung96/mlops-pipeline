import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Model abstraction
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Training model
        :param X_train:
        :param y_train:
        :return:
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear regression model
    """

    def train(self, X_train, y_train):
        """
        Training model
        :param X_train:
        :param y_train:
        :return:
        """
        try:
            logging.info("Training model...")
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error occurred while training model: {str(e)}")
            return None