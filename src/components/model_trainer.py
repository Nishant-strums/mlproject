
import os
import sys
from dataclasses import dataclass

import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


def evaluate_models(X_train, y_train, models):
    try:
        report = {}

        for model_name, model in models.items():

            logging.info(f"Training model: {model_name}")

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )

            mean_score = np.mean(scores)

            report[model_name] = mean_score

            logging.info(
                f"{model_name}: Cross Validation R2 = {mean_score}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        preprocessor_path
    ):

        try:

            logging.info("Splitting training and test input data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(
                    random_state=42
                ),

                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),

                "Linear Regression": LinearRegression(),

                "K-Nearest Neighbors": KNeighborsRegressor(),

                "XGBRegressor": XGBRegressor(
                    random_state=42,
                    n_jobs=-1
                ),

                "AdaBoost Regressor": AdaBoostRegressor(
                    random_state=42
                ),
            }

            logging.info("Starting model evaluation")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                models=models
            )

            logging.info(f"Model Report: {model_report}")

            # Select best model using cross-validation
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            logging.info(
                f"Best Model: {best_model_name}"
            )

            logging.info(
                f"Best Cross Validation R2 Score: {best_model_score}"
            )

            # Train best model on complete training data
            best_model.fit(X_train, y_train)

            # Evaluate on unseen test data
            y_test_pred = best_model.predict(X_test)

            test_r2_score = r2_score(
                y_test,
                y_test_pred
            )

            logging.info(
                f"Final Test R2 Score: {test_r2_score}"
            )

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                ),
                exist_ok=True
            )

            # Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return test_r2_score

        except Exception as e:
            raise CustomException(e, sys)
