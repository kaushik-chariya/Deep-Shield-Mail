import sys
import pandas as pd

from src.configuration.postgres_db_connection import PostgreSQLClient
from src.utils.logger import logger
from src.utils.exception import MyException


class EmailData:

    def insert_csv_to_postgres(self, csv_path):

        conn = None

        try:

            logger.info(f"📂 Reading CSV file: {csv_path}")

            df = pd.read_csv(
                csv_path,
                encoding="utf-8"
            )

            df = df.fillna("")

            logger.info("✅ CSV loaded")
            logger.info(f"📊 Total rows: {len(df)}")

            conn = PostgreSQLClient()

            logger.info("✅ PostgreSQL connected")

            logger.info("📤 Uploading data to PostgreSQL")

            df.to_sql(
                name="emails",
                con=conn.engine,
                if_exists="replace",
                index=False
            )

            logger.info("✅ Data inserted successfully")

        except Exception as e:

            raise MyException(e, sys)

        finally:

            if conn:
                conn.close()

            logger.info("🔒 Connection closed")