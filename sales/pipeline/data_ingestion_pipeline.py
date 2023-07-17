from sales.config.configuration import ConfigManager
from sales.components.data_ingestion import DataIngestion
from sales.logger import logging

STAGE_NAME = 'DATA INGESTION STAGE'


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def start(self):
        # initializing config manager class
        logging.info("initialized config manager")
        config = ConfigManager()

        # data ingestion pipeline
        logging.info("started data ingestion pipeline")
        data_ingestion_config = config.get_data_ingestion()
        data_ingestion_ = DataIngestion(data_ingestion_config)
        train_data_path, test_data_path = data_ingestion_.start_data_ingestion_config()
        logging.info("completed data ingestion pipeline")

        return train_data_path, train_data_path


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        train_path, test_path = obj.start()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
