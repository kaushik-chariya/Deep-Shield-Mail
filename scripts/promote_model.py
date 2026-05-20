"""
Model ko S3 production bucket mein promote karta hai
agar metrics threshold pass kare toh.

Flow:
  1. reports/metrics.json load karo
  2. Har metric threshold se compare karo
  3. Pass → S3 pe model + preprocessor upload karo
  4. Pass → MLflow registry mein @production alias set karo
  5. Fail → sys.exit(1) se CI rok do
"""

import json
import os
import sys
import logging

import mlflow
from mlflow import MlflowClient

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
    MODEL_EVALUATION_MODEL_NAME,
    MODEL_PUSHER_ALIAS,
)
from src.aws_storage.s3_ops import s3_operations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
METRICS_PATH = os.path.join("reports", "metrics.json")

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_FILE_NAME,
)

PREPROC_PATH = os.path.join(
    ARTIFACT_DIR,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREPROCESSING_OBJECT_FILE_NAME,
)

# ── S3 ─────────────────────────────────────────────────────────
S3_BUCKET      = os.getenv("AWS_BUCKET_NAME", "deepshield-models")
S3_MODEL_KEY   = "production/model.pkl"
S3_PREPROC_KEY = "production/preprocessing.pkl"

# ── Thresholds ─────────────────────────────────────────────────
THRESHOLDS = {
    "accuracy":  0.95,
    "f1":        0.95,
    "precision": 0.95,
    "recall":    0.95,
    "auc":       0.97,
}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def load_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}\n"
            "Ensure the DVC pipeline ran successfully and "
            "reports/metrics.json was generated."
        )
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def is_model_good_enough(metrics: dict) -> bool:
    logger.info("── Metric Evaluation ──────────────────────")
    passed = True
    for metric, threshold in THRESHOLDS.items():
        value  = metrics.get(metric, 0)
        status = "✅" if value >= threshold else "❌"
        logger.info(
            "%s  %-12s : %.4f  (threshold ≥ %.4f)",
            status, metric, value, threshold,
        )
        if value < threshold:
            passed = False
    logger.info("───────────────────────────────────────────")
    return passed


def setup_mlflow() -> MlflowClient:
    """DagShub + MLflow init — same pattern as trainer & tests."""
    dagshub_token = os.getenv("DEEPSHIELD_TEST")
    if not dagshub_token:
        raise EnvironmentError(
            "DEEPSHIELD_TEST environment variable not set"
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = "kaushik-chariya"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
    # ✅ FIX 1: dagshub.init() required so MLflow can resolve
    # artifact paths and write to the registry correctly.
    try:
        import dagshub
        dagshub.init(
            repo_owner="kaushik-chariya",
            repo_name="Deep-Shield-Mail",
            mlflow=True,
        )
        logger.info("✅ DagShub initialized")
    except Exception as e:
        raise EnvironmentError(f"DagShub init failed: {e}")

    mlflow.set_tracking_uri(
        "https://dagshub.com/kaushik-chariya/Deep-Shield-Mail.mlflow"
    )

    return MlflowClient()


def get_latest_version(client: MlflowClient) -> str:
    """Registry se sabse latest version number fetch karo."""
    versions = client.search_model_versions(
        filter_string=f"name='{MODEL_EVALUATION_MODEL_NAME}'",
        order_by=["version_number DESC"],
        max_results=1,
    )
    if not versions:
        raise Exception(
            f"No versions found for model '{MODEL_EVALUATION_MODEL_NAME}' "
            "in MLflow registry. Ensure the trainer ran first."
        )
    return versions[0].version


def set_production_alias(client: MlflowClient, version: str) -> None:
    """
    ✅ FIX 2: Latest version ko MODEL_PUSHER_ALIAS
    (e.g. 'production') assign karo.
    Bina iske MLflow registry aur S3 out-of-sync rehte hain —
    registry mein koi 'production' model nahi hota.
    """
    client.set_registered_model_alias(
        name=MODEL_EVALUATION_MODEL_NAME,
        alias=MODEL_PUSHER_ALIAS,
        version=version,
    )
    logger.info(
        "✅ MLflow alias '@%s' → version %s  (model: %s)",
        MODEL_PUSHER_ALIAS,
        version,
        MODEL_EVALUATION_MODEL_NAME,
    )


def upload_to_s3() -> None:
    """model.pkl + preprocessor S3 pe upload karo."""

    # ✅ FIX 3: Region env var se lo, constants se nahi.
    # CI mein AWS_DEFAULT_REGION set hota hai; hardcoded
    # constant CI environment ko reflect nahi karta.
    region = os.getenv("AWS_DEFAULT_REGION", REGION_NAME)

    s3 = s3_operations(
        bucket_name    = S3_BUCKET,
        aws_access_key = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY),
        aws_secret_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY),
        region_name    = region,
    )

    uploads = {
        MODEL_PATH:   S3_MODEL_KEY,
        PREPROC_PATH: S3_PREPROC_KEY,
    }

    for local_path, s3_key in uploads.items():
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Upload aborted — file not found: {local_path}"
            )
        s3.upload_file(local_path, s3_key)
        logger.info("📤 Uploaded  %s  →  s3://%s/%s", local_path, S3_BUCKET, s3_key)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def promote_model() -> None:
    logger.info("═══════════════════════════════════════════")
    logger.info("        Model Promotion Script Start       ")
    logger.info("═══════════════════════════════════════════")

    # ── Step 1 : Metrics load + evaluate ───────────────────────
    metrics = load_metrics()
    logger.info("Loaded metrics: %s", metrics)

    if not is_model_good_enough(metrics):
        logger.error(
            "❌ Model thresholds pass nahi kiye. "
            "Promotion FAILED — CI rok diya gaya."
        )
        sys.exit(1)

    logger.info("✅ Saare thresholds pass!")

    # ── Step 2 : MLflow setup ───────────────────────────────────
    client  = setup_mlflow()
    version = get_latest_version(client)
    logger.info("Latest registry version : %s", version)

    # ── Step 3 : S3 upload ─────────────────────────────────────
    logger.info("📦 S3 pe files upload kar raha hoon...")
    upload_to_s3()

    # ── Step 4 : MLflow alias set ──────────────────────────────
    set_production_alias(client, version)

    logger.info("═══════════════════════════════════════════")
    logger.info("🚀 Model successfully promoted to production!")
    logger.info("   S3  : s3://%s/production/", S3_BUCKET)
    logger.info("   MLflow alias @%s → v%s", MODEL_PUSHER_ALIAS, version)
    logger.info("═══════════════════════════════════════════")


if __name__ == "__main__":
    promote_model()