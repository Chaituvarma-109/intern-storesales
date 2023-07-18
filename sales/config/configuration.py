from pathlib import Path

from sales.constants import CONFIG_FILE_PATH
from sales.utils import read_yaml, create_directories
from sales.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig


class ConfigManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_train_path=Path(config.data_train_path),
            data_test_path=Path(config.data_test_path),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
        )

        return data_ingestion_config

    def get_data_transformation(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            preprocessed_obj_file_path=Path(config.preprocessed_obj_file_path),
            train_arr_path=Path(config.train_arr_path),
            test_arr_path=Path(config.test_arr_path),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path)
        )

        return data_transformation_config

    def get_model_trainer(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            model_file_path=Path(config.model_file_path),
            train_arr_path=Path(config.train_arr_path),
            test_arr_path=Path(config.test_arr_path),
        )

        return model_trainer_config
