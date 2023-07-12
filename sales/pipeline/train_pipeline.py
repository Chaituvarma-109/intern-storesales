from sales.config.configuration import ConfigManager
from sales.components.data_ingestion import DataIngestion
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

    # data transformation pipeline
    logging.info("started data transformation pipeline")
    data_transformation_config = config.get_data_transformation()
    data_transformation = DataTransformation(data_transformation_config)
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    logging.info("completed data transformation pipeline")

    # model trainer pipeline
    logging.info("started model trainer pipeline")
    model_trainer_config = config.get_model_trainer()
    model_trainer = ModelTrainer(model_trainer_config)
    score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    logging.info("completed model trainer pipeline")

    logging.info(f"model score: {score}")
    logging.info(">>>>>>>>>> train pipeline completed <<<<<<<<<<")

    return score


if __name__ == "__main__":
    res = train_pipeline()
    print(res)
