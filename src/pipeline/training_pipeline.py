import sys

from src.utils.logger import logger
from src.utils.exception import MyException

from src.components.data_ingestion import DataIngestion        # ✅ add
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
    DataIngestionConfig,       # ✅ add
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

import pandas as pd  # ✅ add


class TrainPipeline:

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self):

        self.data_ingestion_config = (        # ✅ add
            DataIngestionConfig()
        )

        self.data_validation_config = (
            DataValidationConfig()
        )

        self.data_transformation_config = (
            DataTransformationConfig()
        )

        self.model_trainer_config = (
            ModelTrainerConfig()
        )

    # =====================================================
    # DATA INGESTION  ✅
    # =====================================================

    def start_data_ingestion(
        self,
        df: pd.DataFrame
    ) -> DataIngestionArtifact:

        try:

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = (
                data_ingestion.initiate_data_ingestion(df)
            )

            logger.info("Performed data ingestion successfully")
            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # DATA VALIDATION
    # =====================================================

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:

        logger.info(
            "Entered start_data_validation "
            "method of TrainPipeline class"
        )

        try:

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            data_validation_artifact = (
                data_validation.initiate_data_validation()
            )

            logger.info("Performed data validation successfully")
            logger.info(
                "Exited start_data_validation "
                "method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    # =====================================================
    # DATA TRANSFORMATION
    # =====================================================

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:

        try:

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logger.info("Performed data transformation successfully")
            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # MODEL TRAINER
    # =====================================================

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:

        try:

            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )

            model_trainer_artifact = (
                model_trainer.initiate_model_trainer()
            )

            logger.info("Performed model training successfully")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)

    # =====================================================
    # RUN COMPLETE PIPELINE
    # =====================================================

    def run_pipeline(
        self,
        df: pd.DataFrame          # ✅ data_ingestion_artifact ki jagah df
    ) -> ModelTrainerArtifact:

        try:

            logger.info("Starting complete training pipeline 🏃🏻‍♂️")

            # DATA INGESTION ✅
            data_ingestion_artifact = self.start_data_ingestion(df=df)

            # DATA VALIDATION
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # DATA TRANSFORMATION
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )

            # MODEL TRAINER
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logger.info("Training pipeline completed successfully ✅")

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)