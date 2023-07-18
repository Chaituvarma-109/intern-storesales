import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from sqlalchemy.engine import create_engine

from sales.exception import CustomException
from sales.config.configuration import DataIngestionConfig
from sales.logger import logging

load_dotenv()

uri: str = os.environ.get("DB_URL")
pwd: str = os.environ.get("DB_PWD")


class Database:
    conn = f"postgresql://postgres:{pwd}@db.{uri}:5432/postgres"

    def __init__(self):
        pass

    def initialize_engine(self):
        engine = create_engine(self.conn)

        return engine

    def download_data(self, table):
        engine = self.initialize_engine()
        df = pd.read_sql(table, engine)
        return df


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config

    def start_data_ingestion_config(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info("Downloading Train Data from Database")
            train = Database().download_data(table='train')

            logging.info("Downloading Test Data from Database")
            test = Database().download_data(table='test')

            os.makedirs(os.path.dirname(self.ingestion_config.data_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.data_test_path), exist_ok=True)

            logging.info("saving train and test data in csv format")
            train.to_csv(self.ingestion_config.data_train_path, index=False, header=True)
            test.to_csv(self.ingestion_config.data_test_path, index=False, header=True)

            train_df = pd.read_csv(self.ingestion_config.data_train_path)
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
