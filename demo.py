import sys
from src.data_access.email_data import EmailData
from src.data_access.load_data import (
    load_data_from_postgres,
    load_data_from_s3,
    load_data_from_both,
    load_params
)
from src.utils.logger import logger
from src.utils.exception import MyException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation          # ✅ added
from src.entity.config_entity import DataValidationConfig          # ✅ added
from src.pipeline.training_pipeline import TrainPipeline


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Demo pipeline: STARTED...✈️")
        logger.info("=" * 60)

        # ── Step 1 → Insert CSV into PostgreSQL ──────────────────────
        logger.info("[Step 1/5] Inserting CSV into PostgreSQL ✔️")
        csv_path = "data/spam_assassin.csv"
        logger.info("Source file: %s", csv_path)
        email_obj = EmailData()
        email_obj.insert_csv_to_postgres(csv_path)
        logger.info("[Step 1/5] CSV insertion completed successfully ✔️")

        # ── Step 2 → Fetch data from PostgreSQL + S3 simultaneously ──
        logger.info("[Step 2/5] Fetching data from PostgreSQL + S3 ✔️")

        params    = load_params("params.yaml")
        pg_config = params["data_ingestion"]["postgres"]
        s3_config = params["data_ingestion"]["s3"]
        source    = params["data_ingestion"]["source"]

        if source == "postgres":
            logger.info("Source mode: PostgreSQL only")
            df = load_data_from_postgres(pg_config)

        elif source == "s3":
            logger.info("Source mode: S3 only")
            df = load_data_from_s3(s3_config)

        elif source == "both":
            logger.info("Source mode: PostgreSQL + S3 simultaneously 🚀")
            df = load_data_from_both(pg_config, s3_config)
            logger.info("Combined DataFrame shape: %s", df.shape)
            logger.info("Source breakdown: %s", df["_source"].value_counts().to_dict())

        else:
            raise ValueError(
                f"Unknown source '{source}'. Use 'postgres', 's3', or 'both'."
            )

        logger.info(
            "[Step 2/5] Data fetched — shape: %s, columns: %s",
            df.shape, list(df.columns)
        )
        logger.info(
            "[Step 2/5] Null counts: %s",
            df.isnull().sum().to_dict()
        )
        logger.info(
            "[Step 2/5] Source breakdown:\n%s",
            df['_source'].value_counts().to_dict() if '_source' in df.columns else "single source"
        )

        # ── Step 3 → Data Ingestion ───────────────────────────────────
        logger.info("[Step 3/5] Starting Data Ingestion...")
        data_ingestion = DataIngestion()
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion(df)
        logger.info("[Step 3/5] Data Ingestion Completed ✅")
        logger.info("Train File Path: %s", data_ingestion_artifact.trained_file_path)
        logger.info("Test File Path:  %s", data_ingestion_artifact.test_file_path)

        # ── Step 4 → Data Validation ──────────────────────────────────  ✅ NEW
        logger.info("[Step 4/5] Starting Data Validation...")
        data_validation_config = DataValidationConfig()
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logger.info("[Step 4/5] Data Validation Completed ✅")
        logger.info(
            "Validation status : %s", data_validation_artifact.validation_status
        )
        logger.info(
            "Validation message: %s",
            data_validation_artifact.message or "All checks passed."
        )
        logger.info(
            "Validation report : %s",
            data_validation_artifact.validation_report_file_path
        )

        if not data_validation_artifact.validation_status:
            raise ValueError(
                f"Data Validation Failed: {data_validation_artifact.message}"
            )

        # ── Step 5 → Training Pipeline ────────────────────────────────
        logger.info("[Step 5/5] Initialising TrainPipeline...")
        pipeline = TrainPipeline()
        logger.info("[Step 5/5] Running training pipeline...🏃🏻‍♂️")
        pipeline.run_pipeline(data_ingestion_artifact)
        logger.info("[Step 5/5] Training pipeline completed successfully ✔️")

        logger.info("=" * 60)
        logger.info("Demo pipeline: COMPLETED SUCCESSFULLY....🏆")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(
            "Demo pipeline: FAILED at runtime — %s", str(e), exc_info=True
        )
        raise MyException(e, sys)