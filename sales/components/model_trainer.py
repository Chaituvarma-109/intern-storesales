import sys
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

from sales.config.configuration import ModelTrainerConfig
from sales.exception import CustomException
from sales.logger import logging
from sales.utils import save_object, load_numpy_array_data


def evaluate_regression_models(best_models_list: list, X_train, y_train, X_test, y_test, base_accuracy):
    try:
        accepted_model: dict = {}
        for model in best_models_list:
            # Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculating r squared score on training and testing dataset
            train_acc = r2_score(y_train, y_train_pred)
            test_acc = r2_score(y_test, y_test_pred)

            # Calculating mean squared error on training and testing dataset
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Calculating harmonic mean of train_accuracy and test_accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)

            # logging all important metric
            logging.info(f"{'>>' * 30} Score {'<<' * 30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

            logging.info(f"{'>>' * 30} Loss {'<<' * 30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].")
            logging.info(f"Train root mean squared error: [{train_rmse}].")
            logging.info(f"Test root mean squared error: [{test_rmse}].")

            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.5:
                base_accuracy = model_accuracy
                logging.info(f"model accuracy is: {base_accuracy}")
                accepted_model['model_name'] = str(model)
                accepted_model['model_obj'] = model
                accepted_model['train_accuracy'] = train_acc
                accepted_model['test_accuracy'] = test_acc
                accepted_model['train_rmse'] = train_rmse
                accepted_model['test_rmse'] = test_rmse
                accepted_model['model_accuracy'] = base_accuracy

        else:
            logging.info(f"No model found with higher accuracy than base accuracy")
        return accepted_model
    except Exception as e:
        raise CustomException(e, sys)


def get_models_list(X_train, y_train, models: dict, param: dict) -> list[dict]:
    try:
        model_report: dict = {}
        grid_search_models_list: list = []

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_params = param[list(models.keys())[i]]

            logging.info(f"training model: {model}")
            logging.info(f"with params: {model_params}.")
            gs = GridSearchCV(model, model_params, cv=5)
            gs.fit(X_train, y_train)

            model_report['model_name'] = model
            model_report['best_model_estimator'] = gs.best_estimator_
            model_report['best_params'] = gs.best_params_
            model_report['best_score'] = gs.best_score_

            grid_search_models_list.append(model_report)

        return grid_search_models_list

    except Exception as e:
        raise CustomException(e, sys)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, base_accuracy: float = 0.5):
        self.model_trainer_config = config
        self.base_accuracy = base_accuracy

    def initiate_model_trainer(self) -> None:
        try:
            logging.info("loading train and test arrays")
            train_arr = load_numpy_array_data(self.model_trainer_config.train_arr_path)
            test_arr = load_numpy_array_data(self.model_trainer_config.test_arr_path)
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # TODO: include more models like svr and metrics like precision and recall.
            # TODO: create params.yaml and load the parameters.
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Linear Regression": {},
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [128, 256, 512]
                },
                # "Random Forest": {
                #     'max_depth': [5],
                #     'min_samples_leaf': [6],
                #     'min_samples_split': [2],
                #     'n_estimators': [500],
                # },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [128, 256, 512]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [128, 256, 512]
                }
            }

            logging.info(f"Initiating operation model selection")
            models_report: list[dict] = get_models_list(X_train=X_train, y_train=y_train, models=models, param=params)

            best_model_list: list = [model['best_model_estimator'] for model in models_report]
            logging.info(f"extracting trained model list: {best_model_list}")

            accepted_model: dict = (
                evaluate_regression_models(best_model_list, X_train, y_train, X_test, y_test, self.base_accuracy))
            logging.info(f"Accepted model found: {accepted_model}")

            save_object(file_path=self.model_trainer_config.model_file_path, obj=accepted_model)

        except Exception as e:
            raise CustomException(e, sys)
