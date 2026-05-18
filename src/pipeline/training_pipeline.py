# ═══════════════════════════════════════════════════════════════
# Training Pipeline
# ═══════════════════════════════════════════════════════════════

import sys
import pandas as pd
from typing import Optional

from src.utils.logger    import logger
from src.utils.exception import MyException

from src.components.data_ingestion      import DataIngestion
from src.components.data_validation     import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer       import ModelTrainer
from src.components.model_evaluation    import ModelEvaluation
from src.components.model_pusher        import ModelPusher

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

    # ── Stage 1: Data Ingestion ───────────────────────────────
    def start_data_ingestion(
        self,
        df: Optional[pd.DataFrame] = None,   # None → khud fetch karo (Postgres/S3)
    ) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            artifact = data_ingestion.initiate_data_ingestion(df)
            logger.info("✅ Data Ingestion: COMPLETED")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    # ── Stage 2: Data Validation ──────────────────────────────
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config =self.data_validation_config,
            )
            artifact = data_validation.initiate_data_validation()
            logger.info("✅ Data Validation: COMPLETED")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    # ── Stage 3: Data Transformation ─────────────────────────
    def start_data_transformation(
        self,
        data_ingestion_artifact : DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact   =data_ingestion_artifact,
                data_validation_artifact  =data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            artifact = data_transformation.initiate_data_transformation()
            logger.info("✅ Data Transformation: COMPLETED")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    # ── Stage 4: Model Trainer ────────────────────────────────
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config        =self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            artifact = model_trainer.initiate_model_trainer()
            logger.info("✅ Model Trainer: COMPLETED")
            return artifact
        except Exception as e:
            raise MyException(e, sys)

    # ── Stage 5: Model Evaluation ─────────────────────────────
    def start_model_evaluation(self) -> dict:
        try:
            model_evaluation = ModelEvaluation()
            model_info       = model_evaluation.initiate_model_evaluation()
            logger.info("✅ Model Evaluation: COMPLETED")
            return model_info
        except Exception as e:
            raise MyException(e, sys)

    # ── Stage 6: Model Pusher ─────────────────────────────────
    def start_model_pusher(self, model_info: dict):
        try:
            model_pusher = ModelPusher()
            artifact     = model_pusher.initiate_model_pusher(model_info=model_info)
            logger.info("✅ Model Pusher: COMPLETED")
            return artifact         # ModelPusherArtifact — pushed/skipped info
        except Exception as e:
            raise MyException(e, sys)

    # ── Run Complete Pipeline ─────────────────────────────────
    def run_pipeline(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame, optional
            Bahar se data pass karo (testing/manual run ke liye).
            None (default) → DataIngestion khud Postgres/S3 se fetch karega.
        """
        try:
            logger.info("=" * 60)
            logger.info("🏃 Training Pipeline: STARTED")
            logger.info("=" * 60)

            # 1️⃣ Ingestion
            data_ingestion_artifact = self.start_data_ingestion(df=df)

            # 2️⃣ Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact,
            )

            # 3️⃣ Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact =data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )

            # 4️⃣ Training
            self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact,
            )

            # 5️⃣ Evaluation
            model_info = self.start_model_evaluation()

            # 6️⃣ Pusher
            pusher_artifact = self.start_model_pusher(model_info=model_info)

            # ── Summary ───────────────────────────────────────
            logger.info("=" * 60)
            logger.info("👑 Training Pipeline: COMPLETED SUCCESSFULLY")
            if pusher_artifact and pusher_artifact.pushed:
                logger.info(
                    "🚀 New model PUSHED → '%s' v%s @%s (score: %.4f)",
                    pusher_artifact.model_name,
                    pusher_artifact.model_version,
                    pusher_artifact.model_alias,
                    pusher_artifact.new_score,
                )
            else:
                logger.info(
                    "⏭️  Model NOT pushed — existing champion is better "
                    "(new=%.4f  best=%.4f)",
                    model_info.get("new_score",  0),
                    model_info.get("best_score", 0),
                )
            logger.info("=" * 60)

        except Exception as e:
            logger.error("❌ Training Pipeline: FAILED — %s", str(e), exc_info=True)
            raise MyException(e, sys)


# ───────────────────────────────────────────────────────────────
# Standalone run
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()   # df=None → DataIngestion khud fetch karega