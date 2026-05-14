from src.utils.logger import logger
from src.utils.exception import MyException

import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from constants import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD
)


class PostgreSQLClient:
    """
    PostgreSQLClient is responsible for establishing
    connection with PostgreSQL database.
    """

    engine = None
    SessionLocal = None

    def __init__(self) -> None:

        try:

            logger.info("🚀 Initializing PostgreSQLClient")

            # Create connection only once
            if PostgreSQLClient.engine is None:

                logger.info("📡 Creating PostgreSQL connection")

                DATABASE_URL = (
                    f"postgresql://{POSTGRES_USER}:"
                    f"{POSTGRES_PASSWORD}@"
                    f"{POSTGRES_HOST}:"
                    f"{POSTGRES_PORT}/"
                    f"{POSTGRES_DB}"
                )

                logger.info("⚙️ Creating SQLAlchemy engine")

                PostgreSQLClient.engine = create_engine(
                    DATABASE_URL
                )

                logger.info("⚙️ Creating session maker")

                PostgreSQLClient.SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=PostgreSQLClient.engine
                )

                logger.info(
                    "✅ PostgreSQL connection established successfully"
                )

            self.engine = PostgreSQLClient.engine

            logger.info("🛠️ Creating database session")

            self.session = PostgreSQLClient.SessionLocal()

            logger.info("✅ Database session created successfully")

        except Exception as e:

            logger.exception("❌ Failed to connect PostgreSQL")

            raise MyException(e, sys)

    def close(self):

        try:

            if self.session:

                self.session.close()

                logger.info("🔒 PostgreSQL session closed")

        except Exception as e:

            raise MyException(e, sys)