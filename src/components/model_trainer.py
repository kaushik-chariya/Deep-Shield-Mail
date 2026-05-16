import os
import sys
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils.logger import logger
from src.utils.exception import MyException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
        try:
            clf = XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample, 
                colsample_bytree=self.config.colsample_bytree,
                eval_metric='logloss',
                random_state=self.config.random_state,
                # use_label_encoder=False
            )
            clf.fit(X_train, y_train)
            logger.info("XGBoost model training completed ✅")
            return clf
        except Exception as e:
            raise MyException(e, sys)

    def get_model_object_and_report(
        self,
        train_arr: np.ndarray,
    ):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]

            model = self.train_model(X_train, y_train)

            return model

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("=" * 60)
            logger.info("Model Trainer: STARTED...🤖")
            logger.info("=" * 60)

            # Load .npy files
            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )

            logger.info(f"Train shape: {train_arr.shape}")

            # Train only
            model = self.get_model_object_and_report(train_arr)

            # Save model
            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path),
                exist_ok=True
            )
            with open(self.config.trained_model_file_path, 'wb') as f:
                pickle.dump(model, f)

            logger.info(f"Model saved: {self.config.trained_model_file_path}")
            logger.info("Model Trainer: COMPLETED ✅")

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path,
                metric_artifact=None
            )

        except Exception as e:
            logger.error(f"Model Trainer failed: {e}")
            raise MyException(e, sys)