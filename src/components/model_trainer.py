# ═══════════════════════════════════════════════════════════════
# Model Trainer
# ═══════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import pickle
import dill

from sklearn.naive_bayes import MultinomialNB

from src.utils.logger      import logger
from src.utils.exception   import MyException
from src.utils.main_utils  import read_yaml_file, save_object
from src.entity.config_entity    import ModelTrainerConfig
from src.entity.artifact_entity  import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)

import warnings
warnings.filterwarnings("ignore")


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config         : ModelTrainerConfig,
        data_transformation_artifact : DataTransformationArtifact,
    ):
        try:
            self.config                       = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self._params                      = read_yaml_file(file_path="params.yaml")
            logger.info("🔧 ModelTrainer: params loaded from 'params.yaml'")
        except Exception as e:
            raise MyException(e, sys)

    # ── Train ───────────────────────────────────────────────────
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> MultinomialNB:
        try:
            nb_cfg = self._params["model_trainer"]["naive_bayes"]
            logger.info(
                "🤖 MultinomialNB — alpha=%s, fit_prior=%s",
                nb_cfg["alpha"], nb_cfg["fit_prior"],
            )
            clf = MultinomialNB(
                alpha     =nb_cfg["alpha"],
                fit_prior =nb_cfg["fit_prior"],
            )
            clf.fit(X_train, y_train)
            logger.info("✅ MultinomialNB training completed")
            return clf
        except Exception as e:
            raise MyException(e, sys)

    # ── Main ────────────────────────────────────────────────────
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("=" * 60)
            logger.info("🚀 Model Trainer: STARTED")
            logger.info("=" * 60)

            # ── Step 1: Load train array ────────────────────────
            logger.info(
                "📂 Loading train array from '%s'",
                self.data_transformation_artifact.transformed_train_file_path,
            )
            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )
            logger.info("📐 Train shape: %s", train_arr.shape)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            logger.info(
                "📐 X_train=%s | y_train=%s", X_train.shape, y_train.shape
            )

            # ── Step 2: Train model ────────────────────────────
            clf = self.train_model(X_train, y_train)

            # ── Step 3: Save model.pkl ─────────────────────────
            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path),
                exist_ok=True,
            )
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(clf, f)
            logger.info("💾 model.pkl saved → '%s'", self.config.trained_model_file_path)

            # ── Step 4: Save transformers.pkl (FIX) ────────────
            # Data transformation step mein jo transformers bane hain
            # unhe bhi saath save karo — prediction ke waqt zarurat padegi
            transformers_src = self.data_transformation_artifact.transformed_object_file_path
            transformers_dst = os.path.join(
                os.path.dirname(self.config.trained_model_file_path),
                "transformers.pkl",
            )
            with open(transformers_src, "rb") as f:
                transformers = dill.load(f)
            with open(transformers_dst, "wb") as f:
                dill.dump(transformers, f)
            logger.info("💾 transformers.pkl saved → '%s'", transformers_dst)

            logger.info("=" * 60)
            logger.info("✅ Model Trainer: COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path,
                metric_artifact        =None,
            )

        except Exception as e:
            logger.error("❌ Model Trainer: FAILED — %s", str(e), exc_info=True)
            raise MyException(e, sys)


# ───────────────────────────────────────────────────────────────
# Standalone run
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    transformation_artifact = DataTransformationArtifact(
        transformed_object_file_path="artifact/data_transformation/transformed_object/preprocessing.pkl",
        transformed_train_file_path ="artifact/data_transformation/transformed/train.npy",
        transformed_test_file_path  ="artifact/data_transformation/transformed/test.npy",
    )

    trainer  = ModelTrainer(
        data_transformation_artifact=transformation_artifact,
        model_trainer_config        =ModelTrainerConfig(),
    )
    artifact = trainer.initiate_model_trainer()
    print(artifact)