import sys

from src.data_access.email_data import EmailData
from src.data_access.load_data import LoadData

from src.utils.logger import logger
from src.utils.exception import CustomException


if __name__ == "__main__":

    try:

        logger.info("🚀 Demo started")

        # ─────────────────────────────────────────────
        # Step 1 → Insert CSV into PostgreSQL
        # ─────────────────────────────────────────────
        logger.info(
            "📂 Starting CSV insertion process"
        )

        email_obj = EmailData()

        email_obj.insert_csv_to_postgres(
            "data/raw/archive/spam_assassin.csv"
        )

        logger.info(
            "✅ CSV insertion process completed"
        )

        # ─────────────────────────────────────────────
        # Step 2 → Fetch data from PostgreSQL
        # ─────────────────────────────────────────────
        logger.info(
            "📥 Starting data loading process"
        )

        load_obj = LoadData()

        df = load_obj.fetch_data()

        logger.info(
            f"✅ Data fetched successfully | Shape: {df.shape}"
        )

        logger.info("🏁 Demo completed")

    except Exception as e:

        logger.exception(
            "❌ Error occurred in demo.py"
        )

        raise CustomException(
            e,
            sys
        )