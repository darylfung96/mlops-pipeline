import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from the data path
    """
    def __init__(self, data_path: str):
        """
        Ingesting data from the data path
        :param data_path: Path to the data
        """
        self.data_path = data_path

    def ingest_data(self) -> pd.DataFrame:
        """
        Ingesting data
        :return: Pandas dataframe
        """
        logging.info("Ingesting data...")
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    try:
        ingest_data = IngestData(data_path)
        return ingest_data.ingest_data()
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise e