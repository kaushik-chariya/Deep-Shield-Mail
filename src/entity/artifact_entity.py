from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    trained_file_path : str
    test_file_path    : str


@dataclass
class DataValidationArtifact:
    validation_status           : bool
    message                     : str
    validation_report_file_path : str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path : str
    transformed_train_file_path  : str
    transformed_test_file_path   : str


@dataclass
class ClassificationMetricArtifact:
    f1_score        : float
    precision_score : float
    recall_score    : float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path : str
    metric_artifact         : ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    push_model  : bool
    run_id      : str
    new_score   : float
    best_score  : float
    model_path  : str


@dataclass
class ModelPusherArtifact:
    pushed        : bool
    model_name    : str
    model_version : Optional[str]   # None agar push nahi hua
    model_alias   : Optional[str]   # None agar push nahi hua
    run_id        : Optional[str]
    new_score     : float
    best_score    : float