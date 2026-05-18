# ═══════════════════════════════════════════════════════════════
# Constants — Single Source of Truth
# ═══════════════════════════════════════════════════════════════

import os
import yaml
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

# ── Environment ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ── PostgreSQL ─────────────────────────────────────────────────
POSTGRES_HOST     = os.getenv("POSTGRES_HOST")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT")
POSTGRES_DB       = os.getenv("POSTGRES_DB")
POSTGRES_USER     = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# ── General ────────────────────────────────────────────────────
PIPELINE_NAME                  : str = ""
ARTIFACT_DIR                   : str = "artifact"
MODEL_FILE_NAME                : str = "model.pkl"
TRANSFORMERS_FILE_NAME         : str = "transformers.pkl"
PREPROCESSING_OBJECT_FILE_NAME : str = "preprocessing.pkl"
TARGET_COLUMN                  : str = "target"
FILE_NAME                      : str = "data.csv"
TRAIN_FILE_NAME                : str = "train.csv"
TEST_FILE_NAME                 : str = "test.csv"
SCHEMA_FILE_PATH               : str = os.path.join("config", "schema.yaml")
CURRENT_YEAR                         = date.today().year

# ── AWS ────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID_ENV_KEY     = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME                   = "us-east-1"

# ── Data Ingestion ─────────────────────────────────────────────
DATA_INGESTION_TABLE_NAME             : str   = "emails"
DATA_INGESTION_DIR_NAME               : str   = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR      : str   = "feature_store"
DATA_INGESTION_INGESTED_DIR           : str   = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : float = 0.2

# ── Data Validation ────────────────────────────────────────────
DATA_VALIDATION_DIR_NAME         : str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME : str = "report.yaml"

# ── Data Transformation ────────────────────────────────────────
DATA_TRANSFORMATION_DIR_NAME               : str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR   : str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR : str = "transformed_object"

# ── Hand-crafted features — SINGLE SOURCE OF TRUTH ────────────
# ⚠️  Naya feature SIRF yahan add karo.
#     data_transformation.py → BodyFeatureExtractor/MetaFeatureExtractor
#     predict.py → _build_feature_matrix()
#     schema.yaml → num_features
#     — teeno automatically sync ho jaate hain is list se.

HAND_FEAT_COLS = [
    # ── Body features (BodyFeatureExtractor) ──────────────────
    'caps_ratio',           # uppercase letters ka ratio
    'exclamation_count',    # '!' kitni baar
    'url_count',            # http/www links
    'dollar_count',         # '$' signs
    'html_flag',            # HTML tags present?
    'word_count',           # total words
    'avg_word_length',      # average word length
    'digit_ratio',          # digits ka ratio
    'unique_word_ratio',    # vocabulary richness
    # ── Meta features (EmailMetaFeatureExtractor) ─────────────
    'same_domain',          # from aur to same domain?
    'is_weekend',           # Saturday/Sunday?
    'to_is_generic',        # noreply/admin type address?
]

# ── Model Trainer ──────────────────────────────────────────────
MODEL_TRAINER_DIR_NAME               : str   = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR      : str   = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE         : float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH : str   = os.path.join("config", "model.yaml")

# ── Model Evaluation ───────────────────────────────────────────
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE : float = 0.02

# ── App ────────────────────────────────────────────────────────
APP_HOST : str = "0.0.0.0"
APP_PORT : int = 8000

# ── MLflow / DagsHub — params.yaml se read karo ───────────────
def _read_params() -> dict:
    params_path = os.path.join(os.getcwd(), "params.yaml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

_params = _read_params()

MODEL_EVALUATION_EXPERIMENT_NAME : str = _params["model_evaluation"]["experiment_name"]
MODEL_EVALUATION_MODEL_NAME      : str = _params["model_evaluation"]["model_name"]
MODEL_PUSHER_ALIAS               : str = _params["model_pusher"]["alias"]