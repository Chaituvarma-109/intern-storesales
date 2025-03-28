import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sales.config.configuration import DataTransformationConfig
from sales.exception import CustomException
from sales.logger import logging
from sales.utils import save_object, save_numpy_array_data
from sales.constants import NUMERICAL_COLS, TARGET_COL, CATEGORICAL_COLS


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, col_type, col_names=None):
        self.col_type = col_type
        self.col_names = col_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.col_names)
            if self.col_type == 'categorical':
                X['Item_Type_Combined'] = X['Item_Identifier'].apply(lambda x: x[0:2]).map({
                    'FD': 'Food',
                    'NC': 'Non-Consumable',
                    'DR': 'Drinks'
                })
                X['Item_Fat_Content'] = X['Item_Fat_Content'].replace(['low fat', 'LF', 'reg'],
                                                                      ['Low Fat', 'Low Fat', 'Regular'], inplace=True)
                X.drop(['Item_Type', 'Outlet_Identifier'], axis=1, inplace=True)
            if self.col_type == 'numerical':
                X['Years_Established'] = X['Outlet_Establishment_Year'].apply(lambda x: 2022 - x)
                X['Item_Visibility'] = X['Item_Visibility'].replace(0, X['Item_Visibility'].mean())
                X.drop(['Outlet_Establishment_Year'], axis=1, inplace=True)

            return X
        except Exception as e:
            raise CustomException(e, sys) from e


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.transformation_config = config

    @staticmethod
    def get_data_transformer_object():
        try:
            num_cols, cat_cols = NUMERICAL_COLS, CATEGORICAL_COLS

            num_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('feature_generator', FeatureGenerator(
                    col_type='numerical',
                    col_names=num_cols,
                )),
                ('scaling', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('feature_generator', FeatureGenerator(
                    col_type='categorical',
                    col_names=cat_cols,
                )),
                ('one hot encoding', OneHotEncoder(sparse=False, handle_unknown='ignore')),
                ('scaling', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols),
            ])

            return preprocessing

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):

        try:
            train_df = pd.read_csv(self.transformation_config.train_data_path)
            test_df = pd.read_csv(self.transformation_config.test_data_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_col = TARGET_COL

            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saving train and test arrays")
            save_numpy_array_data(self.transformation_config.train_arr_path, train_arr)
            save_numpy_array_data(self.transformation_config.test_arr_path, test_arr)

            logging.info("Saving preprocessing object.")

            save_object(file_path=self.transformation_config.preprocessed_obj_file_path, obj=preprocessing_obj)

            return train_arr, test_arr, self.transformation_config.preprocessed_obj_file_path,

        except Exception as e:
            raise CustomException(e, sys)
