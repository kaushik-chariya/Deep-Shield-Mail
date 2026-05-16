import json
import sys
import os

import pandas as pd

from pandas import DataFrame
from src.utils.exception import MyException
from src.utils.logger import logger
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.entity.config_entity import DataValidationConfig

from  constants import SCHEMA_FILE_PATH


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            self._schema_config = read_yaml_file(
                file_path=SCHEMA_FILE_PATH
            )

        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(
        self,
        dataframe: DataFrame
    ) -> bool:

        try:

            print("\n================ DEBUG ================")
            print("DataFrame Columns:")
            print(list(dataframe.columns))

            print("\nSchema Columns:")
            print(list(self._schema_config["columns"].keys()))
            print("=======================================\n")

            status = (
                len(dataframe.columns)
                == len(self._schema_config["columns"])
            )

            logger.info(
                f"Is required column present: [{status}]"
            )

            return status

        except Exception as e:
            raise MyException(e, sys)

    def is_column_exist(
        self,
        df: DataFrame
    ) -> bool:

        try:

            dataframe_columns = df.columns

            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:

                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:

                logger.info(
                    f"Missing numerical column: "
                    f"{missing_numerical_columns}"
                )

            for column in self._schema_config["categorical_columns"]:

                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:

                logger.info(
                    f"Missing categorical column: "
                    f"{missing_categorical_columns}"
                )

            return (
                False
                if len(missing_categorical_columns) > 0
                or len(missing_numerical_columns) > 0
                else True
            )

        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:

        try:

            df = pd.read_csv(file_path)

            # Remove unwanted index column
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

            return df

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(
        self
    ) -> DataValidationArtifact:

        try:

            validation_error_msg = ""

            logger.info("Starting data validation")

            train_df, test_df = (
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.trained_file_path
                ),

                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                )
            )

            # Validate number of columns
            status = self.validate_number_of_columns(
                dataframe=train_df
            )

            if not status:
                validation_error_msg += (
                    "Columns are missing in training dataframe. "
                )

            else:
                logger.info(
                    f"All required columns present "
                    f"in training dataframe: {status}"
                )

            status = self.validate_number_of_columns(
                dataframe=test_df
            )

            if not status:
                validation_error_msg += (
                    "Columns are missing in test dataframe. "
                )

            else:
                logger.info(
                    f"All required columns present "
                    f"in testing dataframe: {status}"
                )

            # Validate column names
            status = self.is_column_exist(df=train_df)

            if not status:
                validation_error_msg += (
                    "Columns are missing in training dataframe. "
                )

            else:
                logger.info(
                    f"All categorical/int columns present "
                    f"in training dataframe: {status}"
                )

            status = self.is_column_exist(df=test_df)

            if not status:
                validation_error_msg += (
                    "Columns are missing in test dataframe."
                )

            else:
                logger.info(
                    f"All categorical/int columns present "
                    f"in testing dataframe: {status}"
                )

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            report_dir = os.path.dirname(
                self.data_validation_config.validation_report_file_path
            )

            os.makedirs(report_dir, exist_ok=True)

            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(
                self.data_validation_config.validation_report_file_path,
                "w"
            ) as report_file:

                json.dump(
                    validation_report,
                    report_file,
                    indent=4
                )

            logger.info(
                "Data validation artifact created "
                "and saved to JSON file."
            )

            logger.info(
                f"Data validation artifact: "
                f"{data_validation_artifact}"
            )

            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e

if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
    from src.entity.artifact_entity import DataIngestionArtifact

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path="artifact/data_ingestion/ingested/train.csv",
        test_file_path="artifact/data_ingestion/ingested/test.csv"
    )

    config = DataValidationConfig()
    obj = DataValidation(ingestion_artifact, config)
    artifact = obj.initiate_data_validation()
    print(artifact)
