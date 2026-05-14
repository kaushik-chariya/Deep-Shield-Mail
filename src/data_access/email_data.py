import pandas as pd

from psycopg2.extras import execute_batch

from src.configuration.postgres_db_connection import (
    get_postgres_connection
)

from src.utils.logger import logger


class EmailData:

    def insert_csv_to_postgres(self, csv_path):

        conn = None
        cur = None

        try:

            logger.info(f"📂 Reading CSV file: {csv_path}")

            df = pd.read_csv(
                csv_path,
                encoding="utf-8"
            )

            df = df.fillna("")

            logger.info(f"✅ CSV loaded")
            logger.info(f"📊 Total rows: {len(df)}")

            conn = get_postgres_connection()

            cur = conn.cursor()

            logger.info("✅ PostgreSQL connected")

            cur.execute(
                "TRUNCATE TABLE emails RESTART IDENTITY;"
            )

            logger.info("🗑️ Old data removed")

            data = list(
                zip(
                    df["text"],
                    df["target"]
                )
            )

            execute_batch(
                cur,
                """
                INSERT INTO emails
                (texts, label)

                VALUES (%s, %s)
                """,
                data
            )

            conn.commit()

            logger.info("✅ Data inserted successfully")

        except Exception as e:

            logger.error(f"❌ Error: {e}")

            raise e

        finally:

            if cur:
                cur.close()

            if conn:
                conn.close()

            logger.info("🔒 Connection closed")