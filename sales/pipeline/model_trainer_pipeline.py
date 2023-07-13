from sales.config.configuration import ConfigManager
from sales.components.model_trainer import ModelTrainer
from sales.logger import logging

STAGE_NAME = 'MODEL TRAINING STAGE'


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def start(self):
        # initializing config manager class
        logging.info("initialized config manager")
        config = ConfigManager()

        # model trainer pipeline
        logging.info("started model trainer pipeline")
        model_trainer_config = config.get_model_trainer()
        model_trainer = ModelTrainer(model_trainer_config)
        score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("completed model trainer pipeline")

        logging.info(f"model score: {score}")


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.start()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        logging.info(">>>>>>>>>> train pipeline completed <<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
