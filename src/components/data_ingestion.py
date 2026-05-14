import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.exception import MyException
from src.utils.logger import logger
from src.data_access.load_data import LoadData


class DataIngestion:
    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        """
        :param data_ingestion_config: configuration for data ingestion
        """

        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name : export_data_into_feature_store

        Description :
        This method exports data from PostgreSQL to csv file

        Output :
        Data is returned as artifact of data ingestion components

        On Failure :
        Write an exception log and then raise an exception
        """

        try:

            logger.info("Exporting data from PostgreSQL")

            my_data = LoadData()
            

            dataframe = my_data.fetch_and_save(
                table_name=self.data_ingestion_config.table_name
            )

            logger.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = (
                self.data_ingestion_config.feature_store_file_path
            )

            dir_path = os.path.dirname(feature_store_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logger.info(
                f"Saving exported data into feature store path: "
                f"{feature_store_file_path}"
            )

            dataframe.to_csv(
                feature_store_file_path,
                index=False,
                header=True
            )

            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def split_data_as_train_test(
        self,
        dataframe: DataFrame
    ) -> None:
        """
        Method Name : split_data_as_train_test

        Description :
        This method splits the dataframe into train and test set

        Output :
        Train and test csv files are created

        On Failure :
        Write an exception log and then raise an exception
        """

        logger.info(
            "Entered split_data_as_train_test method "
            "of DataIngestion class"
        )

        try:

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logger.info(
                "Performed train test split on dataframe"
            )

            dir_path = os.path.dirname(
                self.data_ingestion_config.training_file_path
            )

            os.makedirs(dir_path, exist_ok=True)

            logger.info(
                "Exporting train and test files"
            )

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logger.info(
                "Successfully exported train and test files"
            )

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_ingestion(
        self
    ) -> DataIngestionArtifact:
        """
        Method Name : initiate_data_ingestion

        Description :
        This method initiates data ingestion pipeline

        Output :
        Returns data ingestion artifact

        On Failure :
        Write an exception log and then raise an exception
        """

        logger.info(
            "Entered initiate_data_ingestion method "
            "of DataIngestion class"
        )

        try:

            dataframe = self.export_data_into_feature_store()

            logger.info(
                "Successfully fetched data from PostgreSQL"
            )

            self.split_data_as_train_test(dataframe)

            logger.info(
                "Performed train test split successfully"
            )

            logger.info(
                "Exited initiate_data_ingestion method "
                "of DataIngestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=
                self.data_ingestion_config.training_file_path,

                test_file_path=
                self.data_ingestion_config.testing_file_path
            )

            logger.info(
                f"Data ingestion artifact: {data_ingestion_artifact}"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e