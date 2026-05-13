import os
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from src.utils.logger import logger


class LoadData:

    def __init__(self):

        self.db_url = os.getenv("POSTGRES_URL")

        self.table_name = "emails"

        logger.info(
            "✅ LoadData class initialized"
        )

    def create_connection(self):

        try:

            logger.info(
                "🔌 Creating PostgreSQL connection"
            )

            engine = create_engine(self.db_url)

            logger.info(
                "✅ PostgreSQL connection established"
            )

            return engine

        except SQLAlchemyError as e:

            logger.error(
                f"❌ Database connection error: {e}"
            )

            raise e

    def fetch_data(self):

        try:

            logger.info(
                f"📥 Fetching data from table: {self.table_name}"
            )

            engine = self.create_connection()

            query = f"SELECT * FROM {self.table_name}"

            df = pd.read_sql(query, engine)

            logger.info(
                f"✅ Successfully fetched {len(df)} rows"
            )

            logger.info(
                f"📊 Data shape: {df.shape}"
            )

            return df

        except Exception as e:

            logger.error(
                f"❌ Error while fetching data: {e}"
            )

            raise e


if __name__ == "__main__":

    logger.info("🚀 Data loading process started")

    obj = LoadData()

    data = obj.fetch_data()

    logger.info("📌 Displaying first 5 rows")

    print("\n📌 First 5 Rows:\n")

    print(data.head())

    logger.info("✅ Program executed successfully")