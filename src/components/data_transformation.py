import os
import sys

# Ensure project root is in sys.path for direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Configures preprocessing pipelines for numerical and categorical features.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head: \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# ==========================================
# MANUAL SETTING / DIRECT EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Manual file paths
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")

    # 2. Check if train/test artifacts exist; if not, run data ingestion first
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Artifacts not found. Running Data Ingestion first...")
        from src.components.data_ingestion import DataIngestion

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print(f"Data ingestion completed. Train: {train_path}, Test: {test_path}")

    # 3. Trigger Data Transformation manually
    print("\nStarting Data Transformation with manual settings...")
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path,
    )

    print("\n" + "=" * 50)
    print("DATA TRANSFORMATION COMPLETED SUCCESSFULLY!")
    print(f"Train Array Shape: {train_arr.shape}")
    print(f"Test Array Shape:  {test_arr.shape}")
    print(f"Preprocessor Saved: {preprocessor_path}")
    print("=" * 50)