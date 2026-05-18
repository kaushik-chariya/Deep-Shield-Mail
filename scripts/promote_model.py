"""
Model ko S3 production bucket mein promote karta hai
agar metrics threshold pass kare toh.
"""

import json
import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import (
    ARTIFACT_DIR,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_FILE_NAME,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREPROCESSING_OBJECT_FILE_NAME,
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    REGION_NAME,
)
from src.aws_storage.s3_ops import s3_operations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
METRICS_PATH   = os.path.join("reports", "metrics.json")
MODEL_PATH     = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME,
                              MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
PREPROC_PATH   = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME,
                              DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                              PREPROCESSING_OBJECT_FILE_NAME)

# ── S3 ─────────────────────────────────────────────────────────
S3_BUCKET           = os.getenv("AWS_BUCKET_NAME", "deepshield-models")
S3_MODEL_KEY        = "production/model.pkl"
S3_PREPROC_KEY      = "production/preprocessing.pkl"

# ── Thresholds (metrics.json keys: accuracy, f1, precision, recall, auc) ──
THRESHOLDS = {
    "accuracy":  0.95,
    "f1":        0.95,
    "precision": 0.95,
    "recall":    0.95,
    "auc":       0.97,
}


def load_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def is_model_good_enough(metrics: dict) -> bool:
    passed = True
    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric, 0)
        status = "✅" if value >= threshold else "❌"
        logger.info(f"{status}  {metric}: {value:.4f}  (threshold: {threshold})")
        if value < threshold:
            passed = False
    return passed


def promote_model():
    logger.info("═══════════════════════════════════")
    logger.info("   Model Promotion Script Start    ")
    logger.info("═══════════════════════════════════")

    # 1. Metrics check
    metrics = load_metrics()
    logger.info(f"Loaded metrics: {metrics}")

    if not is_model_good_enough(metrics):
        logger.error("❌ Model thresholds pass nahi kiye. Promotion FAILED.")
        sys.exit(1)

    logger.info("✅ Saare thresholds pass! S3 pe push kar raha hoon...")

    # 2. S3 client init
    s3 = s3_operations(
        bucket_name   = S3_BUCKET,
        aws_access_key = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY),
        aws_secret_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY),
        region_name    = REGION_NAME,
    )

    # 3. Upload files
    uploads = {
        MODEL_PATH:   S3_MODEL_KEY,
        PREPROC_PATH: S3_PREPROC_KEY,
    }

    for local_path, s3_key in uploads.items():
        if not os.path.exists(local_path):
            logger.error(f"File not found: {local_path}")
            sys.exit(1)
        s3.upload_file(local_path, s3_key)

    logger.info("🚀 Model successfully promoted to production!")


if __name__ == "__main__":
    promote_model()