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
from src.pipeline.training_pipeline import TrainPipeline


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Demo pipeline: STARTED...✈️")
        logger.info("=" * 60)

        # ── Step 1 → Insert CSV into PostgreSQL ──────────────────────
        logger.info("[Step 1/3] Inserting CSV into PostgreSQL ✔️")
        csv_path = "data/spam_assassin.csv"
        logger.info("Source file: %s", csv_path)
        email_obj = EmailData()
        email_obj.insert_csv_to_postgres(csv_path)
        logger.info("[Step 1/3] CSV insertion completed successfully ✔️")

        # ── Step 2 → Fetch data from PostgreSQL + S3 ─────────────────
        logger.info("[Step 2/3] Fetching data from PostgreSQL + S3 ✔️")

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
            "[Step 2/3] Data fetched — shape: %s, columns: %s",
            df.shape, list(df.columns)
        )
        logger.info(
            "[Step 2/3] Null counts: %s",
            df.isnull().sum().to_dict()
        )
        logger.info(
            "[Step 2/3] Source breakdown:\n%s",
            df['_source'].value_counts().to_dict() if '_source' in df.columns else "single source"
        )

        # ── Step 3 → Training Pipeline ───────────────────────────────
        logger.info("[Step 3/3] Initialising TrainPipeline...")
        pipeline = TrainPipeline()
        logger.info("[Step 3/3] Running training pipeline...🏃🏻‍♂️")
        model_trainer_artifact = pipeline.run_pipeline(df)  # ✅ df pass karo
        logger.info("[Step 3/3] Training pipeline completed successfully ✔️")

        # Model Trainer Logging
        logger.info("Trained Model Path : %s", model_trainer_artifact.trained_model_file_path)
        logger.info("F1 Score           : %s", model_trainer_artifact.metric_artifact.f1_score)
        logger.info("Precision Score    : %s", model_trainer_artifact.metric_artifact.precision_score)
        logger.info("Recall Score       : %s", model_trainer_artifact.metric_artifact.recall_score)

        logger.info("=" * 60)
        logger.info("Demo pipeline: COMPLETED SUCCESSFULLY....🏆")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(
            "Demo pipeline: FAILED at runtime — %s", str(e), exc_info=True
        )
        raise MyException(e, sys)