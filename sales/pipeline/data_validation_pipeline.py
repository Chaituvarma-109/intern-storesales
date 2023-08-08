from sales.config.configuration import ConfigManager
from sales.components.data_validation import DataValidation, Schema
from sales.logger import logging

STAGE_NAME = 'DATA VALIDATION STAGE'


class DataValidationPipeline:
    def __init__(self):
        pass

    def start(self):
        # initializing config manager class
        logging.info("initialized config manager")
        config = ConfigManager()

        # data validation pipeline
        logging.info("started data validation pipeline")
        data_validation_config = config.get_data_validation()
        data_validation_ = DataValidation(data_validation_config)
        train_file, test_file = data_validation_.get_train_test_file_path()

        Schema.validate(train_file)
        Schema.validate(test_file)
        logging.info("completed data validation pipeline")


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationPipeline()
        obj.start()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        logging.info(">>>>>>>>>> train pipeline completed <<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
