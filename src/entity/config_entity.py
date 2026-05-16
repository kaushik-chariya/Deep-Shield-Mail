import os
from constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR          # ✅ TIMESTAMP hata diya — DVC ke liye fixed path
    timestamp: str = TIMESTAMP                # ✅ Logs ke liye abhi bhi available hai


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_INGESTION_DIR_NAME
    )
    # → artifact/data_ingestion/  ✅

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_FEATURE_STORE_DIR,
        FILE_NAME
    )
    # → artifact/data_ingestion/feature_store/data.csv  ✅

    training_file_path: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_INGESTED_DIR,
        TRAIN_FILE_NAME
    )
    # → artifact/data_ingestion/ingested/train.csv  ✅

    testing_file_path: str = os.path.join(
        data_ingestion_dir,
        DATA_INGESTION_INGESTED_DIR,
        TEST_FILE_NAME
    )
    # → artifact/data_ingestion/ingested/test.csv  ✅

    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    table_name: str = DATA_INGESTION_TABLE_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_VALIDATION_DIR_NAME
    )
    # → artifact/data_validation/  ✅

    validation_report_file_path: str = os.path.join(
        data_validation_dir,
        DATA_VALIDATION_REPORT_FILE_NAME
    )
    # → artifact/data_validation/report.yaml  ✅


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        DATA_TRANSFORMATION_DIR_NAME
    )
    # → artifact/data_transformation/  ✅

    transformed_train_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TRAIN_FILE_NAME.replace("csv", "npy")
    )
    # → artifact/data_transformation/transformed/train.npy  ✅

    transformed_test_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TEST_FILE_NAME.replace("csv", "npy")
    )
    # → artifact/data_transformation/transformed/test.npy  ✅

    transformed_object_file_path: str = os.path.join(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCESSING_OBJECT_FILE_NAME
    )
    # → artifact/data_transformation/transformed_object/preprocessing.pkl  ✅


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR_NAME
    )
    # → artifact/model_trainer/  ✅

    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_FILE_NAME
    )
    # → artifact/model_trainer/trained_model/model.pkl  ✅

    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

    n_estimators: int = MODEL_TRAINER_N_ESTIMATORS
    max_depth: int = MODEL_TRAINER_MAX_DEPTH
    learning_rate: float = MODEL_TRAINER_LEARNING_RATE
    subsample: float = MODEL_TRAINER_SUBSAMPLE
    colsample_bytree: float = MODEL_TRAINER_COLSAMPLE_BYTREE
    random_state: int = MODEL_TRAINER_RANDOM_STATE