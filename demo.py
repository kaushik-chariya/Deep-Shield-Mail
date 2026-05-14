import sys
from src.data_access.email_data import EmailData
from src.data_access.load_data import LoadData
from src.utils.logger import logger
from src.utils.exception import MyException
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Demo pipeline: STARTED...✈️")
        logger.info("=" * 60)

        # ── Step 1 → Insert CSV into PostgreSQL ──────────────────────
        logger.info("[Step 1/4] Inserting CSV into PostgreSQL ✔️")
        csv_path = "data/spam_assassin.csv"
        logger.info("Source file: %s", csv_path)
        email_obj = EmailData()
        email_obj.insert_csv_to_postgres(csv_path)
        logger.info("[Step 1/4] CSV insertion completed successfully ✔️")

        # ── Step 2 → Fetch data from PostgreSQL ──────────────────────
        logger.info("[Step 2/4] Fetching data from PostgreSQL ✔️")
        load_obj = LoadData()
        df = load_obj.fetch_data()
        logger.info(
            "[Step 2/4] Data fetched — shape: %s, columns: %s",
            df.shape, list(df.columns)
        )
        logger.info(
            "[Step 2/4] Null counts: %s",
            df.isnull().sum().to_dict()
        )

        # ── Step 3 → Run training pipeline ───────────────────────────
        logger.info("[Step 3/4] Initialising TrainPipeline...")
        pipeline = TrainPipeline()
        logger.info("[Step 3/4] Running training pipeline...🏃🏻‍♂️")
        pipeline.run_pipeline()
        logger.info("[Step 3/4] Training pipeline completed successfully ✔️")

        # ── Step 4 → Done ─────────────────────────────────────────────
        logger.info("[Step 4/4] All steps finished ✔️")
        logger.info("=" * 60)
        logger.info("Demo pipeline: COMPLETED SUCCESSFULLY....🏆")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(
            "Demo pipeline: FAILED at runtime — %s", str(e), exc_info=True
        )
        raise MyException(e, sys)