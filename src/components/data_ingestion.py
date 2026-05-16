# ═══════════════════════════════════════════════════════════════
# Data Ingestion
# ─ Accepts DataFrame from PostgreSQL / S3 / both
# ─ Saves feature store CSV
# ─ Performs train-test split
# ─ Returns DataIngestionArtifact
# ═══════════════════════════════════════════════════════════════

import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity   import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from src.utils.exception import MyException
from src.utils.logger    import logger


class DataIngestion:

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info(
                "DataIngestion.__init__: config loaded — "
                "split_ratio=%.2f, feature_store='%s'",
                self.data_ingestion_config.train_test_split_ratio,
                self.data_ingestion_config.feature_store_file_path
            )
        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────
    # Private — Train Test Split
    # ──────────────────────────────────────────────────────────
    def _split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """Split dataframe and save train/test CSVs."""
        try:
            logger.info(
                "✂️  Performing train-test split — "
                "ratio=%.2f, input shape=%s",
                self.data_ingestion_config.train_test_split_ratio,
                dataframe.shape
            )

            train_set, test_set = train_test_split(
                dataframe,
                test_size   = self.data_ingestion_config.train_test_split_ratio,
                random_state= 42
            )

            logger.info(
                "✅ Split done — train=%s, test=%s",
                train_set.shape, test_set.shape
            )

            # ── Target distribution check ──────────────────────
            if "target" in train_set.columns:
                logger.info(
                    "📊 Train target distribution : %s",
                    train_set["target"].value_counts().to_dict()
                )
                logger.info(
                    "📊 Test  target distribution : %s",
                    test_set["target"].value_counts().to_dict()
                )

            # ── Save CSVs ──────────────────────────────────────
            dir_path = os.path.dirname(
                self.data_ingestion_config.training_file_path
            )
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False, header=True
            )
            logger.info(
                "💾 Train CSV saved → '%s'",
                self.data_ingestion_config.training_file_path
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False, header=True
            )
            logger.info(
                "💾 Test  CSV saved → '%s'",
                self.data_ingestion_config.testing_file_path
            )

        except Exception as e:
            raise MyException(e, sys)

    # ──────────────────────────────────────────────────────────
    # Private — Basic Validation
    # ──────────────────────────────────────────────────────────
    def _validate_dataframe(self, dataframe: DataFrame) -> None:
        """Basic checks before processing."""
        if dataframe is None:
            raise ValueError("Incoming DataFrame is None — check your data source")

        if dataframe.empty:
            raise ValueError("Incoming DataFrame is empty — check your data source")

        if "target" not in dataframe.columns:
            raise ValueError(
                "Target column 'target' missing — "
                f"available columns: {dataframe.columns.tolist()}"
            )

        logger.info(
            "✅ DataFrame validation passed — shape=%s, nulls=%s",
            dataframe.shape,
            dataframe.isnull().sum().to_dict()
        )

    # ──────────────────────────────────────────────────────────
    # Public — Main Entry Point
    # ──────────────────────────────────────────────────────────
    def initiate_data_ingestion(
        self,
        dataframe: DataFrame        # ← comes from PostgreSQL / S3 / both
    ) -> DataIngestionArtifact:

        try:
            logger.info("=" * 60)
            logger.info("Data Ingestion: STARTED")
            logger.info("=" * 60)

            # ── Step 1 → Validate ──────────────────────────────
            logger.info("[Step 1/4] Validating incoming DataFrame...")
            self._validate_dataframe(dataframe)
            logger.info(
                "[Step 1/4] Incoming shape   : %s",
                dataframe.shape
            )
            logger.info(
                "[Step 1/4] Incoming columns : %s",
                dataframe.columns.tolist()
            )

            # ── Step 2 → Drop internal columns ────────────────
            logger.info("[Step 2/4] Cleaning internal columns...")

            if "_source" in dataframe.columns:
                logger.info(
                    "[Step 2/4] Dropping '_source' column "
                    "(added by load_data_from_both)"
                )
                dataframe = dataframe.drop(columns=["_source"])

            logger.info(
                "[Step 2/4] Final shape   : %s",
                dataframe.shape
            )
            logger.info(
                "[Step 2/4] Final columns : %s",
                dataframe.columns.tolist()
            )

            # ── Step 3 → Save feature store ───────────────────
            logger.info("[Step 3/4] Saving feature store CSV...")

            feature_store_file_path = (
                self.data_ingestion_config.feature_store_file_path
            )

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(
                feature_store_file_path,
                index=False, header=True
            )
            logger.info(
                "[Step 3/4] Feature store saved → 💾 '%s'",
                feature_store_file_path
            )

            # ── Step 4 → Train-test split ──────────────────────
            logger.info("[Step 4/4] Performing train-test split...")
            self._split_data_as_train_test(dataframe)
            logger.info("[Step 4/4] Train-test split complete ✔️")

            # ── Build Artifact ─────────────────────────────────
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path = self.data_ingestion_config.training_file_path,
                test_file_path    = self.data_ingestion_config.testing_file_path
            )

            logger.info("=" * 60)
            logger.info("Data Ingestion: COMPLETED SUCCESSFULLY ✅")
            logger.info("=" * 60)
            logger.info(
                "📦 Artifact → train='%s', test='%s'",
                data_ingestion_artifact.trained_file_path,
                data_ingestion_artifact.test_file_path
            )

            return data_ingestion_artifact

        except Exception as e:
            logger.error(
                "Data Ingestion: FAILED — %s", str(e),
                exc_info=True
            )
            raise MyException(e, sys)


# ═══════════════════════════════════════════════════════════════
# MAIN — DVC ke liye entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.data_access.load_data import (
        load_params,
        load_data_from_postgres,
        load_data_from_s3,
        load_data_from_both,
    )

    # ── Params load karo ──────────────────────────────────────
    params = load_params("params.yaml")
    source = params["data_ingestion"]["source"]

    logger.info("📦 Data source: %s", source)

    # ── Source ke hisaab se data fetch karo ──────────────────
    if source == "postgres":
        df = load_data_from_postgres(params["data_ingestion"]["postgres"])

    elif source == "s3":
        df = load_data_from_s3(params["data_ingestion"]["s3"])

    elif source == "both":
        df = load_data_from_both(
            params["data_ingestion"]["postgres"],
            params["data_ingestion"]["s3"],
        )

    else:
        raise ValueError(
            f"Invalid source '{source}' in params.yaml — use postgres / s3 / both"
        )

    # ── Ingestion chalaao ─────────────────────────────────────
    obj      = DataIngestion()
    artifact = obj.initiate_data_ingestion(dataframe=df)

    logger.info(
        "✅ Done! train='%s', test='%s'",
        artifact.trained_file_path,
        artifact.test_file_path
    )