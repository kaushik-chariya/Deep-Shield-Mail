# ═══════════════════════════════════════════════════════════════
# Model Trainer
# ═══════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import pickle
import dill
import warnings

import mlflow
import mlflow.sklearn

from sklearn.naive_bayes import MultinomialNB

from src.utils.logger import logger
from src.utils.exception import MyException
from src.utils.main_utils import read_yaml_file

from src.entity.config_entity import ModelTrainerConfig

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from constants import (
    MODEL_EVALUATION_MODEL_NAME,
)

warnings.filterwarnings("ignore")


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.config = model_trainer_config

            self.data_transformation_artifact = (
                data_transformation_artifact
            )

            self._params = read_yaml_file(
                file_path="params.yaml"
            )

            logger.info(
                "🔧 ModelTrainer initialized successfully"
            )

        except Exception as e:
            raise MyException(e, sys)

    # ───────────────────────────────────────────────────────────
    # Train Model
    # ───────────────────────────────────────────────────────────

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> MultinomialNB:

        try:
            nb_cfg = self._params["model_trainer"]["naive_bayes"]

            logger.info(
                "🤖 Training MultinomialNB | alpha=%s | fit_prior=%s",
                nb_cfg["alpha"],
                nb_cfg["fit_prior"],
            )

            clf = MultinomialNB(
                alpha=nb_cfg["alpha"],
                fit_prior=nb_cfg["fit_prior"],
            )

            clf.fit(X_train, y_train)

            logger.info(
                "✅ MultinomialNB training completed"
            )

            return clf

        except Exception as e:
            raise MyException(e, sys)

    # ───────────────────────────────────────────────────────────
    # Main Training Pipeline
    # ───────────────────────────────────────────────────────────

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:

        try:
            logger.info("=" * 60)
            logger.info("🚀 MODEL TRAINER STARTED")
            logger.info("=" * 60)

            # ───────────────────────────────────────────────
            # Step 1 : Load Train Array
            # ───────────────────────────────────────────────

            logger.info(
                "📂 Loading train array from : %s",
                self.data_transformation_artifact.transformed_train_file_path,
            )

            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )

            logger.info(
                "📐 Train array shape : %s",
                train_arr.shape,
            )

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            logger.info(
                "📐 X_train : %s | y_train : %s",
                X_train.shape,
                y_train.shape,
            )

            # ───────────────────────────────────────────────
            # Step 2 : Train Model
            # ───────────────────────────────────────────────

            clf = self.train_model(
                X_train=X_train,
                y_train=y_train,
            )

            # ───────────────────────────────────────────────
            # Step 3 : Save model.pkl
            # ───────────────────────────────────────────────

            os.makedirs(
                os.path.dirname(
                    self.config.trained_model_file_path
                ),
                exist_ok=True,
            )

            with open(
                self.config.trained_model_file_path,
                "wb",
            ) as f:
                pickle.dump(clf, f)

            logger.info(
                "💾 model.pkl saved at : %s",
                self.config.trained_model_file_path,
            )

            # ───────────────────────────────────────────────
            # Step 4 : Save transformers.pkl
            # ───────────────────────────────────────────────

            transformers_src = (
                self.data_transformation_artifact
                .transformed_object_file_path
            )

            transformers_dst = os.path.join(
                os.path.dirname(
                    self.config.trained_model_file_path
                ),
                "transformers.pkl",
            )

            with open(transformers_src, "rb") as f:
                transformers = dill.load(f)

            with open(transformers_dst, "wb") as f:
                dill.dump(transformers, f)

            logger.info(
                "💾 transformers.pkl saved at : %s",
                transformers_dst,
            )

            # ───────────────────────────────────────────────
            # Step 5 : MLflow + DagShub Setup
            # ───────────────────────────────────────────────

            dagshub_token = os.getenv("DEEPSHIELD_TEST")

            if not dagshub_token:
                raise EnvironmentError(
                    "DEEPSHIELD_TEST env variable not found"
                )

            os.environ["MLFLOW_TRACKING_USERNAME"] = (
                "kaushik-chariya"
            )
            os.environ["MLFLOW_TRACKING_PASSWORD"] = (
                dagshub_token
            )

            # ✅ FIX 1: dagshub.init() must be called so that
            # the artifact store is properly configured.
            # Without this, registered model versions have
            # source='' and artifact downloads fail.
            try:
                import dagshub
                dagshub.init(
                    repo_owner="kaushik-chariya",
                    repo_name="Deep-Shield-Mail",
                    mlflow=True,
                )
                logger.info("✅ DagShub initialized successfully")
            except Exception as e:
                raise EnvironmentError(
                    f"Failed to initialize DagShub: {e}"
                )

            mlflow.set_tracking_uri(
                "https://dagshub.com/kaushik-chariya/Deep-Shield-Mail.mlflow"
            )

            logger.info("✅ MLflow tracking URI configured")

            # ✅ FIX: Experiment set karo start_run se pehle.
            # Bina iske DagShub INVALID_PARAMETER_VALUE deta
            # hai kyunki default experiment ID (0) DagShub pe
            # exist nahi karta.
            mlflow.set_experiment("Deep-Shield-Mail-Spam-Classifier")

            logger.info("✅ MLflow experiment set")

            # ───────────────────────────────────────────────
            # Step 6 : Log Model to MLflow
            # ───────────────────────────────────────────────

            with mlflow.start_run() as run:

                logger.info("🚀 MLflow run started")

                run_id = run.info.run_id

                logger.info("🔥 MLflow Run ID : %s", run_id)

                # ───────────────────────────────
                # Log Parameters
                # ───────────────────────────────

                nb_cfg = self._params["model_trainer"]["naive_bayes"]

                mlflow.log_param("alpha",     nb_cfg["alpha"])
                mlflow.log_param("fit_prior", nb_cfg["fit_prior"])

                # ───────────────────────────────
                # Log + Register Model
                # ✅ FIX 2: use MODEL_EVALUATION_MODEL_NAME
                # from constants instead of a hardcoded string,
                # so trainer and test always refer to the same
                # model name in the registry.
                # ───────────────────────────────

                mlflow.sklearn.log_model(
                    sk_model=clf,
                    artifact_path="model",
                    registered_model_name=MODEL_EVALUATION_MODEL_NAME,
                )

                logger.info(
                    "✅ MLflow model logged & registered as '%s'",
                    MODEL_EVALUATION_MODEL_NAME,
                )

                # ───────────────────────────────
                # Log Transformers
                # ───────────────────────────────

                mlflow.log_artifact(
                    local_path=transformers_dst,
                    artifact_path="transformers",
                )

                logger.info(
                    "✅ transformers.pkl logged successfully"
                )

            logger.info("=" * 60)
            logger.info("✅ MODEL TRAINER COMPLETED")
            logger.info("=" * 60)

            return ModelTrainerArtifact(
                trained_model_file_path=(
                    self.config.trained_model_file_path
                ),
                metric_artifact=None,
            )

        except Exception as e:

            logger.error(
                "❌ Model Trainer Failed : %s",
                str(e),
                exc_info=True,
            )

            raise MyException(e, sys)


# ───────────────────────────────────────────────────────────────
# Standalone Run
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    transformation_artifact = DataTransformationArtifact(
        transformed_object_file_path=(
            "artifact/data_transformation/"
            "transformed_object/preprocessing.pkl"
        ),
        transformed_train_file_path=(
            "artifact/data_transformation/"
            "transformed/train.npy"
        ),
        transformed_test_file_path=(
            "artifact/data_transformation/"
            "transformed/test.npy"
        ),
    )

    trainer = ModelTrainer(
        data_transformation_artifact=transformation_artifact,
        model_trainer_config=ModelTrainerConfig(),
    )

    artifact = trainer.initiate_model_trainer()

    print(artifact)