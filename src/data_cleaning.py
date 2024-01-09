import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Preprocess data
        :param data:
        :return:
        """

        try:
            logging.info("Preprocessing data...")

            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    'customer_zip_code_prefix',
                    'order_item_id'
                ], axis=1
            )

            data['product_weight_g'] = data['product_weight_g'].fillna(data['product_weight_g'].median())
            data['product_length_cm'] = data['product_length_cm'].fillna(data['product_length_cm'].median())
            data['product_height_cm'] = data['product_height_cm'].fillna(data['product_height_cm'].median())
            data['product_width_cm'] = data['product_width_cm'].fillna(data['product_width_cm'].median())
            data['review_comment_message'] = data['review_comment_message'].fillna('No review')

            data = data.select_dtypes(include=[np.number])
            return data
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise e


class DataSplitStrategy(DataStrategy):
    """
    Split data
    """

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the given DataFrame into training and testing sets.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            Tuple[Any, Any, Any, Any]: A tuple containing the training and testing sets of the features
            (X_train, X_test) and the target variable (y_train, y_test).
        """
        try:
            logging.info("Splitting data...")

            # Separate the features and the target variable
            X = data.drop("review_score", axis=1)
            y = data["review_score"]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error splitting data: {e}")
            raise e


class DataHandler:

    def __init__(self, data: pd.DataFrame, data_strategy: DataStrategy):
        """
        Initialize the class.

        Args:
            data (pd.DataFrame): The input data as a pandas DataFrame.
            data_strategy (DataStrategy): The data strategy object.
        """
        # Set the data strategy
        self.data = data
        self.data_strategy = data_strategy

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data
        :param data: pandas dataframe
        :return: cleaned dataframe
        """
        try:
            logging.info("Cleaning data...")
            return self.data_strategy.handle_data(data)
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise e


if __name__ == '__main__':
    data = pd.read_csv('data/olist_customers_dataset.csv')
    data_cleaning = DataHandler(data, DataPreprocessStrategy())
    cleaned_data = data_cleaning.handle_data(data)
    print(cleaned_data)