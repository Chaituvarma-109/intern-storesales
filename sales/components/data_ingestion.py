import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from sales.exception import CustomException
from sales.config.configuration import DataIngestionConfig
from sales.logger import logging


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config

    def start_data_ingestion_config(self):
        logging.info("Entered the data ingestion method or component")
        try:
            train_df = pd.read_csv(self.ingestion_config.data_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(train_df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data iss completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
