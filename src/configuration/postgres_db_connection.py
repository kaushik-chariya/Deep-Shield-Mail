import psycopg2

from src.utils.logger import logger


def get_postgres_connection():

    try:

        logger.info(
            "🔌 Connecting to PostgreSQL database"
        )

        conn = psycopg2.connect(
            host="localhost",
            database="spam_db",
            user="postgres",
            password="Kaushik13"
        )

        logger.info(
            "✅ PostgreSQL connection established"
        )

        return conn

    except Exception as e:

        logger.error(
            f"❌ Database connection failed: {e}"
        )

        raise e