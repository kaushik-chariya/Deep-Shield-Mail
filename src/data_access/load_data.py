import os
import sys
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from src.utils.logger import logger
from src.utils.exception import MyException


class LoadData:

    def __init__(self):

        self.db_url = os.getenv("POSTGRES_URL")
        self.table_name = "emails"

        logger.info("✅ LoadData class initialized")

    # ─────────────────────────────────────────────
    # Create PostgreSQL Connection
    # ─────────────────────────────────────────────
    def create_connection(self):

        try:

            logger.info("🔌 Creating PostgreSQL connection")
            engine = create_engine(self.db_url)
            logger.info("✅ PostgreSQL connection established")
            return engine

        except SQLAlchemyError as e:
            logger.error( f"❌ Database connection error: {e}")
            raise MyException(e, sys)

    # ─────────────────────────────────────────────
    # Fetch Data From PostgreSQL
    # ─────────────────────────────────────────────
    def fetch_data(self) -> pd.DataFrame:

        try:
            logger.info(
                f"📥 Fetching data from table: {self.table_name}"
            )
            engine = self.create_connection()
            query = f"SELECT * FROM {self.table_name}"
            df = pd.read_sql(query, engine)
            logger.info(f"✅ Successfully fetched {len(df)} rows")
            logger.info(f"📊 Data shape: {df.shape}")

            return df

        except Exception as e:

            logger.error( f"❌ Error while fetching data: {e}")

            raise MyException(e, sys)

    # ─────────────────────────────────────────────
    # Save DataFrame To CSV
    # ─────────────────────────────────────────────
    def save_to_csv( self, df: pd.DataFrame, file_path: str = "data/emails.csv" ) -> str:

        try:

            logger.info( "💾 Saving dataframe to CSV")
            # Create folder if not exists
            dir_path = os.path.dirname(file_path)

            if dir_path:

                os.makedirs(
                    dir_path,
                    exist_ok=True
                )

            # Save CSV
            df.to_csv(
                file_path,
                index=False,
                encoding="utf-8"
            )

            abs_path = os.path.abspath(file_path)

            logger.info(
                f"✅ CSV saved successfully at: {abs_path}"
            )

            logger.info(
                f"📊 Saved Rows: {len(df)} | Columns: {len(df.columns)}"
            )

            return abs_path

        except PermissionError as e:

            logger.error(
                f"❌ Permission denied while saving CSV: {e}"
            )

            raise MyException(e, sys)

        except Exception as e:

            logger.error(
                f"❌ Failed to save CSV: {e}"
            )

            raise MyException(e, sys)

    # ─────────────────────────────────────────────
    # Fetch + Save Pipeline
    # ─────────────────────────────────────────────
    def fetch_and_save(
        self,
        table_name: str = "emails",
        file_path: str = "data/emails.csv"
    ) -> pd.DataFrame:

        try:

            logger.info(
                "🔄 Starting fetch-and-save pipeline"
            )

            # Dynamic table name
            self.table_name = table_name

            # Fetch data
            df = self.fetch_data()

            # Save CSV
            self.save_to_csv(df, file_path)

            logger.info(
                "🏁 Fetch-and-save pipeline completed"
            )

            return df

        except Exception as e:

            logger.error(
                f"❌ Pipeline failed: {e}"
            )

            raise MyException(e, sys)


# ─────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────
if __name__ == "__main__":

    try:

        logger.info(
            "🚀 Data loading process started"
        )

        obj = LoadData()

        # Fetch data and save CSV
        data = obj.fetch_and_save(
            table_name="emails",
            file_path="data/emails.csv"
        )

        logger.info(
            "📌 Displaying first 5 rows"
        )

        print("\n📌 First 5 Rows:\n")

        print(data.head())

        logger.info(
            "✅ Program executed successfully"
        )

    except Exception as e:

        logger.error(
            "❌ Main execution failed"
        )

        raise MyException(e, sys)