import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_NAME = "src"

FILES = [
    # ── Package roots ──────────────────────────────────────────────
    f"{PROJECT_NAME}/__init__.py",

    # ── Constants ──────────────────────────────────────────────────
    "constants/__init__.py",

    # ── Config ─────────────────────────────────────────────────────
    "config/schema.yaml",

    # ── Utils ──────────────────────────────────────────────────────
    f"{PROJECT_NAME}/utils/__init__.py",
    f"{PROJECT_NAME}/utils/logger.py",
    f"{PROJECT_NAME}/utils/exception.py",
    f"{PROJECT_NAME}/utils/main_utils.py",

    # ── Configuration (DB + AWS connections) ───────────────────────
    f"{PROJECT_NAME}/configuration/__init__.py",
    f"{PROJECT_NAME}/configuration/postgres_db_connection.py",
    f"{PROJECT_NAME}/configuration/aws_connection.py",

    # ── Data Access ────────────────────────────────────────────────
    f"{PROJECT_NAME}/data_access/__init__.py",
    f"{PROJECT_NAME}/data_access/email_data.py",

    # ── Entity (config + artifact + estimator) ─────────────────────
    f"{PROJECT_NAME}/entity/__init__.py",
    f"{PROJECT_NAME}/entity/config_entity.py",
    f"{PROJECT_NAME}/entity/artifact_entity.py",
    f"{PROJECT_NAME}/entity/estimator.py",
    f"{PROJECT_NAME}/entity/s3_estimator.py",

    # ── Components ─────────────────────────────────────────────────
    f"{PROJECT_NAME}/components/__init__.py",
    f"{PROJECT_NAME}/components/data_ingestion.py",
    f"{PROJECT_NAME}/components/data_validation.py",
    f"{PROJECT_NAME}/components/data_transformation.py",
    f"{PROJECT_NAME}/components/model_trainer.py",
    f"{PROJECT_NAME}/components/model_evaluation.py",
    f"{PROJECT_NAME}/components/model_pusher.py",

    # ── Pipelines ──────────────────────────────────────────────────
    f"{PROJECT_NAME}/pipeline/__init__.py",
    f"{PROJECT_NAME}/pipeline/training_pipeline.py",
    f"{PROJECT_NAME}/pipeline/prediction_pipeline.py",

    # ── AWS Storage ────────────────────────────────────────────────
    f"{PROJECT_NAME}/aws_storage/__init__.py",
    f"{PROJECT_NAME}/aws_storage/s3_ops.py",

    # ── Serving / FastAPI ──────────────────────────────────────────
    "serving/__init__.py",
    "serving/api/__init__.py",
    "serving/api/app.py",

    # ── Web UI ─────────────────────────────────────────────────────
    "templates/index.html",
    "static/style.css",
    "static/script.js",

    # ── Monitoring ─────────────────────────────────────────────────
    "monitoring/__init__.py",
    "monitoring/data_drift.py",
    "monitoring/model_performance.py",

    # ── Infrastructure — Docker ────────────────────────────────────
    "infra/docker/Dockerfile.serve",
    ".dockerignore",

    # ── Infrastructure — Kubernetes ────────────────────────────────
    "infra/kubernetes/namespace.yaml",
    "infra/kubernetes/deployment.yaml",
    "infra/kubernetes/service.yaml",
    "infra/kubernetes/hpa.yaml",
    "infra/kubernetes/ingress.yaml",
    "infra/kubernetes/configmap.yaml",
    "infra/kubernetes/monitoring/prometheus-values.yaml",
    "infra/kubernetes/monitoring/servicemonitor.yaml",
    "infra/kubernetes/monitoring/drift-cronjob.yaml",

    # ── CI/CD ──────────────────────────────────────────────────────
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",

    # ── Tests ──────────────────────────────────────────────────────
    "tests/__init__.py",
    "tests/test_prediction.py",

    # ── Notebooks ──────────────────────────────────────────────────
    "notebooks/postgres_demo.ipynb",
    "notebooks/01_eda.ipynb",
    "notebooks/02_feature_engineering.ipynb",

    # ── Root-level files ───────────────────────────────────────────
    "setup.py",
    "pyproject.toml",
    "requirements.txt",
    "demo.py",
    ".gitignore",
    "README.md",

    # ── Keep empty artifact / log folders in git ───────────────────
    "artifacts/.gitkeep",
    "logs/.gitkeep",
    "mlruns/.gitkeep",
    "data/live_emails.csv",
]


def create_project():
    for file_path_str in FILES:
        file_path = Path(file_path_str)
        directory = file_path.parent

        # ── Create parent directory if needed ──────────────────────
        if str(directory) != ".":
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Directory ready : {directory}")

        # ── Create file only if it doesn't exist ───────────────────
        if not file_path.exists():
            file_path.touch()
            logger.info(f"📄 File created    : {file_path}")
        else:
            logger.info(f"⏩ Already exists  : {file_path}")


if __name__ == "__main__":
    create_project()
    logger.info("\n✅ Project scaffold created successfully!")
    logger.info("   Next step → fill in setup.py and pyproject.toml, then run:")
    logger.info("   conda create -n spamdetector python=3.10 -y")
    logger.info("   conda activate spamdetector")
    logger.info("   pip install -r requirements.txt")