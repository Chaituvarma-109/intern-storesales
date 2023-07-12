from sales.components.data_ingestion import DataIngestion
from sales.components.data_transformation import DataTransformation
from sales.components.model_trainer import ModelTrainer
from sales.logger import logging


def train_pipeline():
    logging.info(">>>>>>>>>> train pipeline started <<<<<<<<<<")

    obj = DataIngestion()
    train_data, test_data = obj.start_data_ingestion_config()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_arr, test_arr)

    logging.info(f"model score: {score}")
    logging.info(">>>>>>>>>> train pipeline completed <<<<<<<<<<")

    return score


if __name__ == "__main__":
    res = train_pipeline()
    print(res)
