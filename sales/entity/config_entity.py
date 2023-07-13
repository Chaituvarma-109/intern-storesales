from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_path: Path
    train_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    preprocessed_obj_file_path: Path
    train_arr_path: Path
    test_arr_path: Path
    train_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_file_path: Path
    train_arr_path: Path
    test_arr_path: Path
