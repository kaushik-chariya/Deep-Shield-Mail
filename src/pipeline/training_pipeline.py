import sys
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import MyException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation   # ✅
from src.components.model_pusher import ModelPusher           # ✅

from src.entity.config_entity import (
    DataIngestionConfig,
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


class TrainPipeline:

    def __init__(self):
        self.data_ingestion_config      = DataIngestionConfig()
        self.data_validation_config     = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config       = ModelTrainerConfig()

    # ── Data Ingestion ────────────────────────────────────
    def start_data_ingestion(self, df: pd.DataFrame) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            artifact = data_ingestion.initiate_data_ingestion(df)
            logger.info("Data Ingestion: COMPLETED ✅")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    # ── Data Validation ───────────────────────────────────
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            artifact = data_validation.initiate_data_validation()
            logger.info("Data Validation: COMPLETED ✅")
            return artifact
        except Exception as e:
            raise MyException(e, sys) from e

    # ── Data Transformation ───────────────────────────────
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
            artifact = data_transformation.initiate_data_transformation()
            logger.info("Data Transformation: COMPLETED ✅")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    # ── Model Trainer ─────────────────────────────────────
    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            artifact = model_trainer.initiate_model_trainer()
            logger.info("Model Trainer: COMPLETED ✅")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    # ── Model Evaluation ──────────────────────────────────
    def start_model_evaluation(self) -> dict:
        try:
            model_evaluation = ModelEvaluation()          # ✅ class use
            model_info = model_evaluation.initiate_model_evaluation()
            logger.info("Model Evaluation: COMPLETED ✅")
            return model_info
        except Exception as e:
            raise MyException(e, sys)

    # ── Model Pusher ──────────────────────────────────────
    def start_model_pusher(self, model_info: dict) -> None:
        try:
            model_pusher = ModelPusher()                  # ✅ class use
            model_pusher.initiate_model_pusher(model_info=model_info)
            logger.info("Model Pusher: COMPLETED ✅")
        except Exception as e:
            raise MyException(e, sys)

    # ── Run Complete Pipeline ─────────────────────────────
    def run_pipeline(self, df: pd.DataFrame) -> None:
        try:
            logger.info("=" * 60)
            logger.info("Training Pipeline: STARTED 🏃🏻‍♂️")
            logger.info("=" * 60)

            # 1️⃣ Ingestion
            data_ingestion_artifact = self.start_data_ingestion(df=df)

            # 2️⃣ Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # 3️⃣ Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )

            # 4️⃣ Training
            self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            # 5️⃣ Evaluation
            model_info = self.start_model_evaluation()

            # 6️⃣ Pusher
            self.start_model_pusher(model_info=model_info)

            logger.info("=" * 60)
            logger.info("Training Pipeline: COMPLETED ✅ 👑")
            logger.info("=" * 60)

        except Exception as e:
            raise MyException(e, sys)