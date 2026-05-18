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

from constants import SCHEMA_FILE_PATH


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config  = data_validation_config
            self._schema_config          = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logger.info("DataValidation: schema loaded from '%s'", SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(
        self,
        dataframe: DataFrame
    ) -> bool:
        try:
            actual   = len(dataframe.columns)
            expected = len(self._schema_config["columns"])
            status   = actual == expected

            logger.info(
                "validate_number_of_columns: actual=%d, expected=%d, status=%s",
                actual, expected, status,
            )
            logger.info(
                "validate_number_of_columns: dataframe columns — %s",
                list(dataframe.columns),
            )
            logger.info(
                "validate_number_of_columns: schema columns — %s",
                list(self._schema_config["columns"].keys()),
            )

            return status

        except Exception as e:
            raise MyException(e, sys)

    def is_column_exist(
        self,
        df: DataFrame
    ) -> bool:
        try:
            dataframe_columns            = df.columns
            missing_numerical_columns    = []
            missing_categorical_columns  = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logger.info(
                    "is_column_exist: missing numerical columns — %s",
                    missing_numerical_columns,
                )

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logger.info(
                    "is_column_exist: missing categorical columns — %s",
                    missing_categorical_columns,
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
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)
            logger.info("read_data: loaded '%s' — shape=%s", file_path, df.shape)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("=" * 70)
            logger.info("Data Validation Pipeline: STARTED")
            logger.info("=" * 70)

            validation_error_msg = ""

            # Step 1: Load data
            logger.info("[Step 1/3] Loading train and test CSVs")
            train_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )
            test_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            # Step 2: Validate number of columns
            logger.info("[Step 2/3] Validating number of columns")
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "
            else:
                logger.info(
                    "validate_number_of_columns: all required columns present in train — %s",
                    status,
                )

            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += "Columns are missing in test dataframe. "
            else:
                logger.info(
                    "validate_number_of_columns: all required columns present in test — %s",
                    status,
                )

            # Step 3: Validate column names
            logger.info("[Step 3/3] Validating column names")
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "
            else:
                logger.info(
                    "is_column_exist: all categorical/numerical columns present in train — %s",
                    status,
                )

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += "Columns are missing in test dataframe."
            else:
                logger.info(
                    "is_column_exist: all categorical/numerical columns present in test — %s",
                    status,
                )

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            # Save report
            report_dir = os.path.dirname(
                self.data_validation_config.validation_report_file_path
            )
            os.makedirs(report_dir, exist_ok=True)

            validation_report = {
                "validation_status": validation_status,
                "message"          : validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logger.info(
                "Data validation report saved to '%s'",
                self.data_validation_config.validation_report_file_path,
            )

            logger.info("=" * 70)
            logger.info(
                "Data Validation Pipeline: COMPLETED — status=%s",
                validation_status,
            )
            logger.info("=" * 70)

            return data_validation_artifact

        except Exception as e:
            logger.error("Data Validation: FAILED — %s", str(e), exc_info=True)
            raise MyException(e, sys) from e


if __name__ == "__main__":
    from src.entity.config_entity   import DataIngestionConfig, DataValidationConfig
    from src.entity.artifact_entity import DataIngestionArtifact

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path="artifact/data_ingestion/ingested/train.csv",
        test_file_path   ="artifact/data_ingestion/ingested/test.csv"
    )

    config   = DataValidationConfig()
    obj      = DataValidation(ingestion_artifact, config)
    artifact = obj.initiate_data_validation()
    print(artifact)