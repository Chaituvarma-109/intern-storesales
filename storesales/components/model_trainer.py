import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from storesales.exception import CustomException
from storesales.logger import logging
from storesales.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    train_model_file_path: Path = os.path.join("../../artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                # "Decision Tree": DecisionTreeRegressor(),
                # "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                # "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Linear Regression": {},
                # "Decision Tree": {
                #     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                #     # 'splitter':['best','random'],
                #     # 'max_features':['sqrt','log2'],
                # },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                # "Gradient Boosting": {
                #     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                #     'learning_rate': [.1, .01, .05, .001],
                #     'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                #     # 'criterion':['squared_error', 'friedman_mse'],
                #     # 'max_features':['auto','sqrt','log2'],
                #     'n_estimators': [8, 16, 32, 64, 128, 256]
                # },
                # "AdaBoost Regressor": {
                #     'learning_rate': [.1, .01, 0.5, .001],
                #     # 'loss':['linear','square','exponential'],
                #     'n_estimators': [8, 16, 32, 64, 128, 256]
                # }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"best model: {best_model_name}, best model_score: {best_model_score}")

            if best_model_score < 0.5:
                raise Exception("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
