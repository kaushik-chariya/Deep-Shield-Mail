# ─────────────────────────────────────────────
# Data Ingestion — Load from PostgreSQL + S3 simultaneously
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import os
import concurrent.futures                          # ✅ run both sources in parallel
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.utils.logger import logging
from src.configuration.aws_connection import s3_operations

import psycopg2
from sqlalchemy import create_engine 
from dotenv import load_dotenv



# ─────────────────────────────────────────────
# Load .env file
# ─────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────
# Load Params from YAML
# ─────────────────────────────────────────────
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


# ─────────────────────────────────────────────
# Load from PostgreSQL
# ─────────────────────────────────────────────
def load_data_from_postgres(pg_config: dict) -> pd.DataFrame:
    """Load data from PostgreSQL — credentials from .env"""
    try:
        # ── Credentials from .env ──────────────────────────────────
        host     = os.getenv("POSTGRES_HOST")
        port     = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB")
        user     = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")

        # ── Non-sensitive config from params.yaml ──────────────────
        table = pg_config['table']
        query = pg_config.get('query', f"SELECT * FROM {table}")

        # ── Validate env vars ───────────────────────────────────────
        missing = [
            k for k, v in {
                "POSTGRES_HOST"    : host,
                "POSTGRES_DB"      : database,
                "POSTGRES_USER"    : user,
                "POSTGRES_PASSWORD": password,
            }.items() if not v
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {missing}"
            )

        # ── Connect and fetch ───────────────────────────────────────
        connection_string = (
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )

        logging.info(
            "Connecting to PostgreSQL at %s:%s/%s as user '%s'",
            host, port, database, user
        )

        engine = create_engine(connection_string)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        logging.info(
            "✅ PostgreSQL — loaded table '%s' — shape: %s",
            table, df.shape
        )
        return df

    except EnvironmentError:
        raise
    except Exception as e:
        logging.error("Failed to load data from PostgreSQL: %s", e)
        raise


# ─────────────────────────────────────────────
# Load from S3
# ─────────────────────────────────────────────
def load_data_from_s3(s3_config: dict) -> pd.DataFrame:
    """Load CSV from S3 — credentials from .env"""
    try:
        # ── Credentials from .env ──────────────────────────────────
        access_key = os.getenv("AWS_ACCESS_KEY")
        secret_key = os.getenv("AWS_SECRET_KEY")
        region     = os.getenv("AWS_REGION")

        # ── Non-sensitive config from params.yaml ──────────────────
        bucket_name = s3_config['bucket_name']
        file_key    = s3_config['file_key']

        # ── Validate env vars ───────────────────────────────────────
        # missing = [
        #     k for k, v in {
        #         "AWS_ACCESS_KEY_ID"    : access_key,
        #         "AWS_SECRET_ACCESS_KEY": secret_key,
        #     }.items() if not v
        # ]
        # if missing:
        #     raise EnvironmentError(
        #         f"Missing required environment variables: {missing}"
        #     )

        logging.info(
            "Connecting to S3 bucket '%s' in region '%s'",
            bucket_name, region
        )

        s3 = s3_operations(bucket_name, access_key, secret_key)
        df = s3.fetch_file_from_s3(file_key)

        logging.info(
            "✅ S3 — loaded '%s/%s' — shape: %s",
            bucket_name, file_key, df.shape
        )
        return df

    except EnvironmentError:
        raise
    except Exception as e:
        logging.error("Failed to load data from S3: %s", e)
        raise


# ─────────────────────────────────────────────
# ✅ NEW: Load from BOTH simultaneously
# ─────────────────────────────────────────────
def load_data_from_both(pg_config: dict, s3_config: dict) -> pd.DataFrame:
    """
    Fetch from PostgreSQL and S3 at the same time using
    ThreadPoolExecutor, then merge both DataFrames into one.
    """
    logging.info("🚀 Fetching from PostgreSQL and S3 simultaneously...")

    # ── Run both loaders in parallel threads ───────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_pg = executor.submit(load_data_from_postgres, pg_config)
        future_s3 = executor.submit(load_data_from_s3, s3_config)

        # ── Collect results (raises exception if either failed) ─────
        df_pg = future_pg.result()
        df_s3 = future_s3.result()

    logging.info("PostgreSQL rows : %d", len(df_pg))
    logging.info("S3 rows         : %d", len(df_s3))

    # ── Tag source so you know where each row came from ────────────
    df_pg['_source'] = 'postgres'
    df_s3['_source'] = 's3'

    # ── Combine both DataFrames ─────────────────────────────────────
    df_combined = pd.concat([df_pg, df_s3], ignore_index=True)

    # ── Drop exact duplicates (same email in both sources) ─────────
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(
        subset=[c for c in df_combined.columns if c != '_source']
    )
    after = len(df_combined)

    logging.info(
        "✅ Combined shape: %s  (dropped %d duplicates)",
        df_combined.shape, before - after
    )
    return df_combined
