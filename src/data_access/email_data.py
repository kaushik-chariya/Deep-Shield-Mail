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

            # Check existing data
            cur.execute("SELECT COUNT(*) FROM emails")

            count = cur.fetchone()[0]

            logger.info(f"📦 Existing rows in table: {count}")

            if count > 0:

                logger.warning(
                    "⚠️ Data already exists in table!"
                )

                cur.close()
                conn.close()

                logger.info("🔒 Database connection closed")

                return

            logger.info("🚀 Starting data insertion process")

            # Insert data row by row
            for index, row in df.iterrows():

                cur.execute(
                    """
                    INSERT INTO emails
                    (subject, message, label, email_date)

                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        row["Subject"],
                        row["Message"],
                        row["Spam/Ham"],
                        row["Date"]
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