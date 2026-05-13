import os
from pathlib import Path

project_name = "src"

list_of_files = [

    # ==============================
    # Root Files
    # ==============================
    ".github/workflows/.gitkeep",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".gitignore",
    "demo.py",

    # ==============================
    # Config
    # ==============================
    "config/schema.yaml",

    # ==============================
    # Constants
    # ==============================
    "constants/__init__.py",

    # ==============================
    # Main Package
    # ==============================
    f"{project_name}/__init__.py",

    # ==============================
    # Components
    # ==============================
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",

    # ==============================
    # Pipeline
    # ==============================
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",

    # ==============================
    # Entity
    # ==============================
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/estimator.py",
    f"{project_name}/entity/s3_estimator.py",

    # ==============================
    # Configuration
    # ==============================
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/postgres_db_connection.py",
    f"{project_name}/configuration/aws_connection.py",

    # ==============================
    # Data Access
    # ==============================
    f"{project_name}/data_access/__init__.py",
    f"{project_name}/data_access/email_data.py",

    # ==============================
    # AWS Storage
    # ==============================
    f"{project_name}/aws_storage/__init__.py",
    f"{project_name}/aws_storage/s3_operations.py",

    # ==============================
    # Utils
    # ==============================
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/logger.py",
    f"{project_name}/utils/exception.py",
    f"{project_name}/utils/main_utils.py",

    # ==============================
    # Serving API
    # ==============================
    "serving/api/app.py",
    "serving/__init__.py",

    # ==============================
    # Monitoring
    # ==============================
    "monitoring/data_drift.py",
    "monitoring/model_performance.py",

    # ==============================
    # Infrastructure
    # ==============================
    "infra/docker/Dockerfile.serve",
    "infra/kubernetes/namespace.yaml",
    "infra/kubernetes/deployment.yaml",
    "infra/kubernetes/service.yaml",
    "infra/kubernetes/hpa.yaml",
    "infra/kubernetes/ingress.yaml",
    "infra/kubernetes/configmap.yaml",

    # ==============================
    # Monitoring Infra
    # ==============================
    "infra/kubernetes/monitoring/prometheus-values.yaml",
    "infra/kubernetes/monitoring/servicemonitor.yaml",
    "infra/kubernetes/monitoring/drift-cronjob.yaml",

    # ==============================
    # Templates & Static
    # ==============================
    "templates/index.html",
    "static/style.css",

    # ==============================
    # Notebooks
    # ==============================
    "notebooks/postgres_demo.ipynb",
    "notebooks/01_eda.ipynb",
    "notebooks/02_feature_engineering.ipynb",

    # ==============================
    # Tests
    # ==============================
    "tests/__init__.py",

    # ==============================
    # Logs & Artifacts
    # ==============================
    "logs/.gitkeep",
    "artifacts/.gitkeep",
]

for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass

    else:
        print(f"{filename} already exists")