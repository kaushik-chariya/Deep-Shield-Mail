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
        logger.info("✈️  Demo pipeline: STARTED...")
        logger.info("=" * 60)

        # ── Step 1 → Insert CSV into PostgreSQL ──────────────────────
        logger.info("📂 [Step 1/3] Inserting CSV into PostgreSQL...")
        csv_path  = "data/spam_assassin.csv"
        email_obj = EmailData()
        email_obj.insert_csv_to_postgres(csv_path)
        logger.info("✅ [Step 1/3] CSV insertion completed!")

        # ── Step 2 → Fetch data ───────────────────────────────────────
        logger.info("🔄 [Step 2/3] Fetching data...")

        params    = load_params("params.yaml")
        pg_config = params["data_ingestion"]["postgres"]
        s3_config = params["data_ingestion"]["s3"]
        source    = params["data_ingestion"]["source"]

        if source == "postgres":
            logger.info("🐘 Source mode: PostgreSQL only")
            df = load_data_from_postgres(pg_config)

        elif source == "s3":
            logger.info("☁️  Source mode: S3 only")
            df = load_data_from_s3(s3_config)

        elif source == "both":
            logger.info("🚀 Source mode: PostgreSQL + S3 simultaneously")
            df = load_data_from_both(pg_config, s3_config)
            logger.info("📊 Combined shape: %s", df.shape)

        else:
            raise ValueError(
                f"❌ Unknown source '{source}'. Use 'postgres', 's3', or 'both'."
            )

        logger.info("✅ [Step 2/3] Data fetched successfully!")
        logger.info("📐 Shape: %s | 🏷️  Columns: %s", df.shape, list(df.columns))

        # ── Step 3 → Full Training Pipeline ──────────────────────────
        logger.info("🏃🏻‍♂️ [Step 3/3] Running Full Training Pipeline...")
        logger.info("🔁 Flow: Ingestion → Validation → Transformation → Training → Evaluation → Pusher")

        pipeline = TrainPipeline()
        pipeline.run_pipeline(df)

        logger.info("✅ [Step 3/3] Full pipeline completed!")

        logger.info("=" * 60)
        logger.info("🏆 Demo pipeline: COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("💥 Demo pipeline: FAILED — %s", str(e), exc_info=True)
        raise MyException(e, sys)  