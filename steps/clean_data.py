import logging

import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple

from src.data_cleaning import DataHandler, DataPreprocessStrategy, DataSplitStrategy


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleaning data
    :param df: pandas dataframe
    :return:
    X_train: Training data
    X_test: Testing data
    y_train: Training labels
    y_test: Testing labels
    """
    try:
        logging.info("Cleaning data...")
        data_strategy = DataPreprocessStrategy()
        df = DataHandler(df, data_strategy).handle_data(df)
        data_strategy = DataSplitStrategy()
        X_train, X_test, y_train, y_test = DataHandler(df, data_strategy).handle_data(df)
        logging.info("Data cleaned!")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e
