import pandas as pd

from src.configuration.postgres_db_connection import (
    get_postgres_connection
)

from src.utils.logger import logger


class EmailData:

    def insert_csv_to_postgres(self, csv_path):

        try:

            logger.info(f"📂 Reading CSV file: {csv_path}")

            # Read CSV
            df = pd.read_csv(csv_path)

            logger.info(f"✅ CSV loaded successfully")
            logger.info(f"📊 Total rows found: {len(df)}")

            # PostgreSQL connection
            conn = get_postgres_connection()

            logger.info("✅ PostgreSQL connection established")

            # Cursor
            cur = conn.cursor()

            logger.info("✅ Cursor created")

            # Delete old data
            logger.warning(
                "🗑️ Deleting old data from emails table"
            )

            cur.execute(
                "TRUNCATE TABLE emails RESTART IDENTITY;"
            )

            conn.commit()

            logger.info("✅ Old data deleted")

            logger.info("🚀 Starting data insertion process")

            # Insert data row by row
            for index, row in df.iterrows():

                cur.execute(
                    """
                    INSERT INTO emails
                    (message, label)

                    VALUES (%s, %s)
                    """,
                    (
                        row["text"],
                        row["target"]
                    )
                )

                # Progress log every 10000 rows
                if index % 10000 == 0:

                    logger.info(
                        f"📥 Inserted {index} rows"
                    )

            # Save changes
            conn.commit()

            logger.info(
                "✅ Data inserted successfully into PostgreSQL"
            )

            # Close connection
            cur.close()
            conn.close()

            logger.info("🔒 Database connection closed")

        except Exception as e:

            logger.error(
                f"❌ Error occurred: {e}"
            )

            raise e