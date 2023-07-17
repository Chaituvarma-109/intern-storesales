from sales.config.configuration import ConfigManager
from sales.components.data_transformation import DataTransformation
from sales.logger import logging

STAGE_NAME = 'DATA TRANSFORMATION STAGE'


class DataTransformationPipeline:
    def __init__(self):
        pass

    def start(self):
        # initializing config manager class
        logging.info("initialized config manager")
        config = ConfigManager()

        # data ingestion pipeline
        logging.info("started data ingestion pipeline")
        data_transformation_config = config.get_data_transformation()
        data_transformation_ = DataTransformation(data_transformation_config)
        data_transformation_.initiate_data_transformation()
        logging.info("completed data ingestion pipeline")


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.start()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
