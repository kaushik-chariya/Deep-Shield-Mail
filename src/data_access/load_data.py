import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from src.utils.logger import logger


class LoadData:

    def __init__(self):
        self.db_url = self.db_url = os.getenv("POSTGRES_URL")
        self.table_name = "emails"
        logger.info("✅ LoadData class initialized")

    def create_connection(self):
        try:
            logger.info("🔌 Creating PostgreSQL connection")
            engine = create_engine(self.db_url)
            logger.info("✅ PostgreSQL connection established")
            return engine
        except SQLAlchemyError as e:
            logger.error(f"❌ Database connection error: {e}")
            raise e

    def fetch_data(self) -> pd.DataFrame:
        try:
            logger.info(f"📥 Fetching data from table: {self.table_name}")
            engine = self.create_connection()
            query = f"SELECT * FROM {self.table_name}"
            df = pd.read_sql(query, engine)
            logger.info(f"✅ Successfully fetched {len(df)} rows")
            logger.info(f"📊 Data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Error while fetching data: {e}")
            raise e

    def save_to_csv(self, df: pd.DataFrame, file_path: str = "data/emails.csv") -> str:
        """
        Save a DataFrame to a CSV file.

        Args:
            df        : DataFrame to save.
            file_path : Destination path (including filename). Default: data/emails.csv

        Returns:
            Absolute path of the saved file.
        """
        try:
            # Create parent directories if they don't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            df.to_csv(file_path, index=False, encoding="utf-8")

            abs_path = os.path.abspath(file_path)
            logger.info(f"✅ Data saved to CSV | Path: {abs_path} | Rows: {len(df)} | Columns: {len(df.columns)}")
            return abs_path

        except PermissionError as e:
            logger.error(f"❌ Permission denied while saving CSV: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to save CSV: {e}")
            raise

    def fetch_and_save(self, file_path: str = "data/raw/emails.csv") -> pd.DataFrame:
        """
        Convenience method: fetch data from DB and immediately save to CSV.

        Args:
            file_path : Destination CSV path. Default: data/emails.csv

        Returns:
            The fetched DataFrame.
        """
        logger.info("🔄 Starting fetch-and-save pipeline")
        df = self.fetch_data()
        self.save_to_csv(df, file_path)
        logger.info("🏁 Fetch-and-save pipeline completed")
        return df


if __name__ == "__main__":
    logger.info("🚀 Data loading process started")

    obj = LoadData()

    # ── Option 1: fetch + save in one call ──────────────────────
    data = obj.fetch_and_save(file_path="data/raw/emails.csv")

    # ── Option 2: fetch first, save separately ──────────────────
    # data = obj.fetch_data()
    # obj.save_to_csv(data, file_path="data/emails.csv")

    logger.info("📌 Displaying first 5 rows")
    print("\n📌 First 5 Rows:\n")
    print(data.head())
    logger.info("✅ Program executed successfully")