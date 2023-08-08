from sales.config.configuration import ConfigManager
from sales.components.data_ingestion import DataIngestion
from sales.components.data_validation import DataValidation, Schema
from sales.components.data_transformation import DataTransformation
from sales.components.model_trainer import ModelTrainer
from sales.logger import logging


def train_pipeline():
    logging.info(">>>>>>>>>> train pipeline started <<<<<<<<<<")

    # initializing config manager class
    logging.info("initialized config manager")
    config = ConfigManager()

    # data ingestion pipeline
    logging.info("started data ingestion pipeline")
    data_ingestion_config = config.get_data_ingestion()
    data_ingestion = DataIngestion(data_ingestion_config)
    train_data, test_data = data_ingestion.start_data_ingestion_config()
    logging.info("completed data ingestion pipeline")

    # data validation pipeline
    logging.info("started data validation pipeline")
    data_validation_config = config.get_data_validation()
    data_validation = DataValidation(data_validation_config)
    train_data_, test_data_ = data_validation.get_train_test_file_path()

    Schema.validate(train_data_)
    Schema.validate(test_data_)
    logging.info("completed data validation pipeline")

    # data transformation pipeline
    logging.info("started data transformation pipeline")
    data_transformation_config = config.get_data_transformation()
    data_transformation = DataTransformation(data_transformation_config)
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation()
    logging.info("completed data transformation pipeline")

    # model trainer pipeline
    logging.info("started model trainer pipeline")
    model_trainer_config = config.get_model_trainer()
    model_trainer = ModelTrainer(model_trainer_config)
    model_trainer.initiate_model_trainer()
    logging.info("completed model trainer pipeline")

    logging.info(">>>>>>>>>> train pipeline completed <<<<<<<<<<")


if __name__ == "__main__":
    train_pipeline()
