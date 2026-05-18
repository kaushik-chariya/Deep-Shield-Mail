import os
from dataclasses import dataclass
from datetime import datetime
from constants import *

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# ── Training Pipeline ─────────────────────────────────────────

@dataclass
class TrainingPipelineConfig:
    pipeline_name : str = PIPELINE_NAME
    artifact_dir  : str = ARTIFACT_DIR
    timestamp     : str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


# ── Data Ingestion ────────────────────────────────────────────

@dataclass
class DataIngestionConfig:
    data_ingestion_dir      : str   = ""
    feature_store_file_path : str   = ""
    training_file_path      : str   = ""
    testing_file_path       : str   = ""
    train_test_split_ratio  : float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    table_name              : str   = DATA_INGESTION_TABLE_NAME

    def __post_init__(self):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME,
        )
        # → artifact/data_ingestion/

        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            FILE_NAME,
        )
        # → artifact/data_ingestion/feature_store/data.csv

        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TRAIN_FILE_NAME,
        )
        # → artifact/data_ingestion/ingested/train.csv

        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TEST_FILE_NAME,
        )
        # → artifact/data_ingestion/ingested/test.csv


# ── Data Validation ───────────────────────────────────────────

@dataclass
class DataValidationConfig:
    data_validation_dir         : str = ""
    validation_report_file_path : str = ""

    def __post_init__(self):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME,
        )
        # → artifact/data_validation/

        self.validation_report_file_path = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_REPORT_FILE_NAME,
        )
        # → artifact/data_validation/report.yaml


# ── Data Transformation ───────────────────────────────────────

@dataclass
class DataTransformationConfig:
    data_transformation_dir      : str = ""
    transformed_train_file_path  : str = ""
    transformed_test_file_path   : str = ""
    transformed_object_file_path : str = ""

    def __post_init__(self):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME,
        )
        # → artifact/data_transformation/

        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        # → artifact/data_transformation/transformed/train.npy

        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy"),
        )
        # → artifact/data_transformation/transformed/test.npy

        self.transformed_object_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCESSING_OBJECT_FILE_NAME,
        )
        # → artifact/data_transformation/transformed_object/preprocessing.pkl


# ── Model Trainer ─────────────────────────────────────────────

@dataclass
class ModelTrainerConfig:
    model_trainer_dir       : str   = ""
    trained_model_file_path : str   = ""
    transformers_file_path  : str   = ""
    expected_accuracy       : float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path  : str   = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

    def __post_init__(self):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            MODEL_TRAINER_DIR_NAME,
        )
        # → artifact/model_trainer/

        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_DIR,
            MODEL_FILE_NAME,
        )
        # → artifact/model_trainer/trained_model/model.pkl

        self.transformers_file_path = os.path.join(
        self.model_trainer_dir,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        TRANSFORMERS_FILE_NAME
)
        # → artifact/model_trainer/trained_model/transformers.pkl